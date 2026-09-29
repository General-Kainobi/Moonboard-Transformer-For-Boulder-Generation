import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import math
import matplotlib.pyplot as plt
from transformer_model import MoonBoardTransformer
from config import VOCAB, INV_VOCAB, GRADES, PAD_TOKEN, START_TOKEN, END_TOKEN, hold_to_coords, EMBEDDING_DIM

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

class GradeClassificationDataset(Dataset):
    def __init__(self, json_path: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        raw_problems = data.get('data', [])
        self.problems = []
        
        for prob in raw_problems:
            grade = prob.get('grade', '')
            grade_str = f"<{grade}>"
            if grade_str not in GRADES:
                continue
                
            repeats = prob.get('repeats', 0)
            if repeats < 6:
                continue
                
            moves = prob.get('moves', [])
            holds = [m['description'] for m in moves]
            holds.sort(key=lambda h: int(h[1:]))
            
            self.problems.append({'holds': holds, 'grade_idx': GRADES.index(grade_str)})

    def __len__(self):
        return len(self.problems)
        
    def __getitem__(self, idx: int):
        holds = self.problems[idx]['holds']
        grade_idx = self.problems[idx]['grade_idx']
        
        # Input sequence WITHOUT the true grade prefix. Just start with START_TOKEN.
        tokens = [START_TOKEN]
        for h in holds:
            if h in VOCAB:
                tokens.append(VOCAB[h])
        tokens.append(END_TOKEN)
                
        # Calculate dx, dy
        coords = [hold_to_coords(INV_VOCAB.get(t, '<PAD>')) for t in tokens]
        dx_dy = [[0.0, 0.0]]
        for i in range(1, len(coords)):
            if tokens[i] == END_TOKEN:
                dx_dy.append([0.0, 0.0])
            else:
                prev_x, prev_y = coords[i-1]
                curr_x, curr_y = coords[i]
                dx_dy.append([float(curr_x - prev_x), float(curr_y - prev_y)])
            
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(dx_dy, dtype=torch.float32), torch.tensor(grade_idx, dtype=torch.long)

def collate_fn_grader(batch):
    tokens_list, dx_dy_list, grades_list = zip(*batch)
    
    tokens_padded = torch.nn.utils.rnn.pad_sequence(tokens_list, batch_first=True, padding_value=PAD_TOKEN)
    dx_dy_padded = torch.nn.utils.rnn.pad_sequence(dx_dy_list, batch_first=True, padding_value=0.0)
    grades_tensor = torch.stack(grades_list)
    
    attention_mask = (tokens_padded == PAD_TOKEN)
    
    return tokens_padded, dx_dy_padded, attention_mask, grades_tensor

class GradePredictor(nn.Module):
    def __init__(self, base_model, num_grades):
        super().__init__()
        self.base_model = base_model
        # Freeze base model to test if hidden states already encode difficulty, probably will use a linear probe
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        self.fc = nn.Linear(EMBEDDING_DIM, num_grades)
        
    def forward(self, tokens, dx_dy, src_key_padding_mask):
        hidden_states = []
        def hook_fn(module, input, output):
            hidden_states.append(output)
            
        handle = self.base_model.transformer.register_forward_hook(hook_fn)
        
        # We don't need the final logits, just the hidden states
        _ = self.base_model(tokens, dx_dy, src_key_padding_mask)
        handle.remove()
        
        out = hidden_states[0] # [batch_size, seq_len, d_model]
        
        # Mean pooling over seq_len, senza vedere i  padding
        # src mask is true for PAD_TOKEN
        mask = (~src_key_padding_mask).float().unsqueeze(-1) # [batch_size, seq_len, 1]
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        
        logits = self.fc(pooled)
        return logits

def train_grader():
    dataset = GradeClassificationDataset('problems MoonBoard Masters 2019 40.json')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn_grader)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn_grader)
    
    # Load frozen base model
    base_model = MoonBoardTransformer().to(DEVICE)
    base_model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    base_model.eval()
    
    model = GradePredictor(base_model, len(GRADES)).to(DEVICE)
    
    optimizer = optim.AdamW(model.fc.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting Training for Linear Probe Grade Classifier...")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for tokens, dx_dy, mask, grades in train_loader:
            tokens, dx_dy, mask, grades = tokens.to(DEVICE), dx_dy.to(DEVICE), mask.to(DEVICE), grades.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(tokens, dx_dy, mask)
            loss = criterion(logits, grades)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_total += grades.size(0)
            train_correct += (predicted == grades).sum().item()
            
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for tokens, dx_dy, mask, grades in val_loader:
                tokens, dx_dy, mask, grades = tokens.to(DEVICE), dx_dy.to(DEVICE), mask.to(DEVICE), grades.to(DEVICE)
                logits = model(tokens, dx_dy, mask)
                loss = criterion(logits, grades)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_total += grades.size(0)
                val_correct += (predicted == grades).sum().item()
                
        print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Acc: {100.*train_correct/train_total:.2f}% | Val Acc: {100.*val_correct/val_total:.2f}% | Val Loss: {val_loss/len(val_loader):.4f}")

if __name__ == '__main__':
    train_grader()
