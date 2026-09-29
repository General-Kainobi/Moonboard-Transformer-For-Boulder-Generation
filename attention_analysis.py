import torch

import matplotlib.pyplot as plt
import seaborn as sns
from transformer_model import MoonBoardTransformer
from config import VOCAB, INV_VOCAB, hold_to_coords



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def analyze_attention():
    # 1. Load Model
    model = MoonBoardTransformer().to(DEVICE)
    model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    model.eval()

    # 2. Setup Hook to capture attention weights
    attention_weights = None
    
    import types
    def _sa_block_patched(self, x, attn_mask, key_padding_mask, is_causal=False):
        attn_out, attn_weights = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True, is_causal=is_causal)
        return self.dropout1(attn_out)
    model.transformer.layers[-1]._sa_block = types.MethodType(_sa_block_patched, model.transformer.layers[-1])
    
    def hook_fn(module, input, output):
        nonlocal attention_weights
        # output of MultiheadAttention is (attn_output, attn_weights)
        # attn_weights shape: (batch_size, seq_len, seq_len)
        attention_weights = output[1].detach().cpu()
        
    # Register hook on the LAST transformer layer's self-attention module
    hook_handle = model.transformer.layers[-1].self_attn.register_forward_hook(hook_fn)

    # 3. Prepare a sample sequence
    # Let's use a 7C problem: J3 -> K4 -> H10 -> H14 -> J16 -> F18
    sequence_str = ['<7C>', 'J3', 'K4', 'H10', 'H14', 'J16', 'F18']
    
    tokens = [VOCAB[t] for t in sequence_str]
    tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(DEVICE)
    
    # Compute dx_dy
    dx_dy = []
    for i, tok_idx in enumerate(tokens):
        if INV_VOCAB[tok_idx] in ['<PAD>', '<START>', '<END>'] or INV_VOCAB[tok_idx].startswith('<'):
            dx_dy.append([0.0, 0.0])
        else:
            if i == 0 or INV_VOCAB[tokens[i-1]] in ['<PAD>', '<START>', '<END>'] or INV_VOCAB[tokens[i-1]].startswith('<'):
                dx_dy.append([0.0, 0.0])
            else:
                cx, cy = hold_to_coords(INV_VOCAB[tok_idx])
                px, py = hold_to_coords(INV_VOCAB[tokens[i-1]])
                dx_dy.append([float(cx - px), float(cy - py)])
                
    dx_dy_tensor = torch.tensor([dx_dy], dtype=torch.float32).to(DEVICE)
    
    # Create mask (causal)
    seq_len = len(tokens)
    
    # 4. Forward Pass (this triggers the hook)
    with torch.no_grad():
        _ = model(tokens_tensor, dx_dy_tensor)
        
    hook_handle.remove()
    
    # 5. Plot the heatmap
    attn_matrix = attention_weights[0].numpy() # [seq_len, seq_len]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_matrix, xticklabels=sequence_str, yticklabels=sequence_str, cmap="viridis", annot=True, fmt=".2f")
    plt.title("Attention Weights (Final Layer) for a 7C Problem")
    plt.xlabel("Key (Token Being Attended To)")
    plt.ylabel("Query (Current Token Predicting Next)")
    
    output_path = r'C:\Users\natan\.gemini\antigravity\brain\974986b0-f0ce-4977-b6d1-99e0b9c2f2d5\attention_heatmap.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved attention heatmap to {output_path}")

if __name__ == '__main__':
    analyze_attention()
