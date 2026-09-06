import torch
import os

def patch_model():
    if not os.path.exists('best_model.pth'):
        print("No model to patch.")
        return
        
    state_dict = torch.load('best_model.pth')
    
    # Old vocab size = 201, New = 216
    # 15 new grade tokens added starting at index 3.
    # Indices 0, 1, 2 (PAD, START, END) remain the same.
    # Old indices 3 to 200 (the physical holds) shift to 18 to 215.
    
    patched = False
    for key in ['token_embedding.weight', 'fc_out.weight', 'fc_out.bias']:
        if key not in state_dict:
            continue
            
        old_tensor = state_dict[key]
        if old_tensor.shape[0] == 201:
            print(f"Patching {key} from 201 to 216...")
            if len(old_tensor.shape) == 2:
                new_tensor = torch.zeros(216, old_tensor.shape[1], dtype=old_tensor.dtype, device=old_tensor.device)
                new_tensor[0:3] = old_tensor[0:3]
                new_tensor[18:216] = old_tensor[3:201]
                # Initialize new grade tokens randomly
                torch.nn.init.normal_(new_tensor[3:18], mean=0, std=0.02)
                state_dict[key] = new_tensor
                patched = True
            else:
                new_tensor = torch.zeros(216, dtype=old_tensor.dtype, device=old_tensor.device)
                new_tensor[0:3] = old_tensor[0:3]
                new_tensor[18:216] = old_tensor[3:201]
                state_dict[key] = new_tensor
                patched = True
        elif old_tensor.shape[0] == 216:
            print(f"{key} is already size 216.")
            
    if patched:
        torch.save(state_dict, 'best_model.pth')
        print("Patching complete. 15 grade tokens successfully injected.")
    else:
        print("No patching was needed.")

if __name__ == "__main__":
    patch_model()
