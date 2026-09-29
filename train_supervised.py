import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os

from config import DEVICE, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, PAD_TOKEN, VOCAB_SIZE
from dataset import MoonBoardDataset, collate_fn
from transformer_model import MoonBoardTransformer

def train():
    print(f"Using device: {DEVICE}")
    
    # 1. Dataset & DataLoader
    dataset = MoonBoardDataset('problems MoonBoard Masters 2019 40.json')
    
    if len(dataset) == 0:
        print("Dataset is empty after filtering!")
        return
        
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # 2. Model Initialization
    model = MoonBoardTransformer().to(DEVICE)
    
    if os.path.exists('best_model.pth'):
        print("Found existing 'best_model.pth'. Resuming training from checkpoint!", flush=True)
        model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
        
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    
    # ReduceLROnPlateau scheduler helps settle into better generalized minimo
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # CrossEntropyLoss without label smoothing to enforce strict physical constraints
    # label smoothing ho provato ma giustamente fare sampling da un uniforme su dati sequenziali non funziona e "rompe" l'encoding
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    
    # 3. Training Loop con Teacher Forcing (ground truth obbligato da passo precedente al corrente)
    best_val_loss = float('inf')
    
    # Early stopping parameters, maybe switching to 3?( ferma training se passano x epoche senza differenza sostanziale nella loss)
    patience = 5
    epochs_without_improvement = 0
    
    # Store metrics for analysis
    train_losses = []
    val_losses = []
    all_y_true = []
    all_y_pred = []
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        for tokens, dx_dy, attention_mask in train_loader:
            tokens = tokens.to(DEVICE)
            dx_dy = dx_dy.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            
            # Input to model: all tokens except the last one (end token)
            # Target for loss: all tokens except the first one (start token)
            inputs = tokens[:, :-1]
            inputs_dx_dy = dx_dy[:, :-1, :]
            targets = tokens[:, 1:]
            
            # Pad mask: True where attention_mask is 0 (i.e. PAD_TOKEN)
            src_key_padding_mask = (inputs == PAD_TOKEN)
            
            optimizer.zero_grad()
            logits = model(inputs, inputs_dx_dy, src_key_padding_mask=src_key_padding_mask)
            
            # logits: [batch, seq_len, vocab_size] 
            # targets: [batch, seq_len]
            # Flatten for CrossEntropy 
            logits_flat = logits.reshape(-1, VOCAB_SIZE)
            targets_flat = targets.reshape(-1)
            
            loss = criterion(logits_flat, targets_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0 #magari vez 
        
        # Clear previous epoch predictions if we only want the final epoch's confusion matrix
        if epoch == NUM_EPOCHS:
            all_y_true.clear()
            all_y_pred.clear()
            
        with torch.no_grad():
            for tokens, dx_dy, attention_mask in val_loader:
                tokens = tokens.to(DEVICE)
                dx_dy = dx_dy.to(DEVICE)
                
                inputs = tokens[:, :-1]
                inputs_dx_dy = dx_dy[:, :-1, :]
                targets = tokens[:, 1:]
                src_key_padding_mask = (inputs == PAD_TOKEN)
                
                logits = model(inputs, inputs_dx_dy, src_key_padding_mask=src_key_padding_mask)
                loss = criterion(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
                val_loss += loss.item()
                
                if epoch == NUM_EPOCHS:
                    # Get predictions
                    preds = torch.argmax(logits, dim=-1)
                    
                    # Filter out PAD tokens from targets for confusion matrix
                    valid_mask = (targets != PAD_TOKEN)
                    valid_targets = targets[valid_mask].cpu().numpy()
                    valid_preds = preds[valid_mask].cpu().numpy()
                    
                    all_y_true.extend(valid_targets)
                    all_y_pred.extend(valid_preds)
                
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Step the scheduler
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}", flush=True)
        
        # Save checkpoint and check Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print("  --> Saved new best model", flush=True)
        else:
            epochs_without_improvement += 1
            print(f"  --> No improvement for {epochs_without_improvement} epoch(s).", flush=True)
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered! Validation loss hasn't improved for {patience} epochs.", flush=True)
                break

    return train_losses, val_losses, all_y_true, all_y_pred

if __name__ == '__main__':
    train()
