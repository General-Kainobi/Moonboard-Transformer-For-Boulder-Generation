import torch
from dataset import MoonBoardDataset, collate_fn
from torch.utils.data import DataLoader
from transformer_model import MoonBoardTransformer
from config import DEVICE
from analyze import plot_spatial_density
import os

def main():
    print("Loading dataset...")
    dataset = MoonBoardDataset('problems MoonBoard Masters 2019 40.json')
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    print("Loading model...")
    model = MoonBoardTransformer().to(DEVICE)
    model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    model.eval()
    
    y_pred = []
    print("Evaluating...")
    with torch.no_grad():
        for tokens, dx_dy, _ in val_loader:
            tokens = tokens.to(DEVICE)
            dx_dy = dx_dy.to(DEVICE)
            
            logits = model(tokens, dx_dy)
            preds = logits.argmax(dim=-1).cpu().numpy()
            y_pred.extend(preds.flatten().tolist())
            
    print("Generating spatial density plot...")
    plot_spatial_density(y_pred)
    print("Saved spatial_density.png successfully!")

if __name__ == '__main__':
    main()
