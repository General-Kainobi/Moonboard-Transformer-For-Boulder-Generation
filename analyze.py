import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import sys
import os

from train_supervised import train
from config import INV_VOCAB, VOCAB_SIZE

def plot_loss_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Cross Entropy)')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curves.png')
    plt.close()
    print("Saved loss_curves.png")

def plot_spatial_density(y_pred):
    """
    Plots a Spatial Probability Density Map of the predicted holds on the MoonBoard grid.
    """
    # 18 rows, 11 columns
    grid = np.zeros((18, 11))
    cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    
    for p in y_pred:
        hold_str = INV_VOCAB.get(p, '<PAD>')
        if hold_str in ['<PAD>', '<START>', '<END>'] or hold_str.startswith('<'):
            continue
        col_idx = cols.index(hold_str[0])
        row_num = int(hold_str[1:])
        
        x = col_idx
        y = row_num - 1
        
        grid[y, x] += 1
            
    plt.figure(figsize=(10, 14))
    
    # Use seaborn heatmap
    ax = sns.heatmap(grid, cmap='magma', cbar_kws={'label': 'Prediction Frequency'})
    ax.invert_yaxis() # Put row 1 at the bottom
    
    plt.title('Spatial Probability Density of Predicted Next Holds', fontsize=16)
    plt.xlabel('Columns', fontsize=12)
    plt.ylabel('Rows', fontsize=12)
    
    plt.xticks(ticks=np.arange(11) + 0.5, labels=['A','B','C','D','E','F','G','H','I','J','K'])
    plt.yticks(ticks=np.arange(18) + 0.5, labels=[str(i) for i in range(1, 19)])
    
    plt.tight_layout()
    plt.savefig('spatial_density.png')
    plt.close()
    print("Saved spatial_density.png")

def main():
    print("Starting training and analysis process...", flush=True)
    # NOTE: Training will take a significant amount of time depending on the device.
    train_losses, val_losses, y_true, y_pred = train()
    
    if not train_losses:
        print("Training did not produce any losses to plot.", flush=True)
        return
        
    print("Training complete. Generating plots...", flush=True)
    plot_loss_curves(train_losses, val_losses)
    plot_spatial_density(y_pred)
    print("Analysis complete.", flush=True)

if __name__ == '__main__':
    main()
