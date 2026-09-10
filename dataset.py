import json
import math
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple
from config import VOCAB, PAD_TOKEN, START_TOKEN, END_TOKEN, hold_to_coords

class MoonBoardDataset(Dataset):
    """
    PyTorch Dataset for MoonBoard 2019 problems.
    Filters:
      - Less than 6 ascents ('repeats' < 6) are removed.
      - 8B and 8B+ grades are removed.
    Weighting:
      - Problems are duplicated/weighted based on 'userRating' (stars).
    Provides:
      - token_ids: Sequence of discrete hold tokens.
      - dx_dy: Sequence of continuous spatial displacements.
    """
    def __init__(self, json_path: str):
        self.problems = self._load_and_filter_data(json_path)
        
    def _load_and_filter_data(self, json_path: str) -> List[Dict[str, Any]]:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        raw_problems = data.get('data', [])
        filtered = []
        
        for prob in raw_problems:
            # Discard 8B and 8B+ entirely
            grade = prob.get('grade', '')
            if grade in ['8B', '8B+']:
                continue
                
            # Eliminate problems with less than 6 ascents
            repeats = prob.get('repeats', 0)
            if repeats < 6:
                continue
                
            # Weight problems based on a combination of stars (userRating) and ascents (repeats).
            # We use a logarithmic scale for repeats to prevent highly popular problems from dominating,
            # multiplied by the user rating to favor high-quality, well-tested problems.
            rating = prob.get('userRating', 1)
            
            # Calculate weight: e.g., (rating / 5.0) * log2(repeats).
            # We ensure a minimum weight of 1 for any problem that passed the filters.
            weight_val = (rating / 5.0) * math.log2(max(1, repeats))
            weight = max(1, int(round(weight_val)))
            
            # Sort moves by isStart, then keep original order or by problemId?
            # Actually, the sequence of moves in the JSON is usually unordered or ordered by hold.
            # Realistically, sequences should be ordered by y-coordinate (bottom to top).
            moves = prob.get('moves', [])
            
            # Get hold descriptions (e.g. 'J3')
            holds = [m['description'] for m in moves]
            # Order them heuristically by y-coordinate (row number)
            holds.sort(key=lambda h: int(h[1:]))
            
            for _ in range(weight):
                filtered.append({'holds': holds, 'grade': f"<{grade}>"})
                
        return filtered

    def __len__(self):
        return len(self.problems)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        holds = self.problems[idx]['holds']
        grade_token_str = self.problems[idx]['grade']
        
        tokens = []
        if grade_token_str in VOCAB:
            tokens.append(VOCAB[grade_token_str])
        
        tokens.append(START_TOKEN)
        for h in holds:
            if h in VOCAB:
                tokens.append(VOCAB[h])
                
        tokens.append(END_TOKEN)
                
        # Calculate dx, dy
        coords = [hold_to_coords(INV_VOCAB.get(t, '<PAD>')) for t in tokens]
        dx_dy = []
        
        # the displacement from the previous hold
        # For START_TOKEN, displacement is (0,0)
        dx_dy.append([0.0, 0.0])
        
        for i in range(1, len(coords)):
            if tokens[i] == END_TOKEN:
                dx_dy.append([0.0, 0.0])
            else:
                prev_x, prev_y = coords[i-1]
                curr_x, curr_y = coords[i]
                dx_dy.append([float(curr_x - prev_x), float(curr_y - prev_y)])
            
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(dx_dy, dtype=torch.float32)

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pads the sequences to the maximum length in the batch.
    Returns (tokens, dx_dy, attention_mask)
    """
    tokens_list, dx_dy_list = zip(*batch)
    
    # Pad tokens
    tokens_padded = torch.nn.utils.rnn.pad_sequence(
        tokens_list, batch_first=True, padding_value=PAD_TOKEN
    )
    
    # Pad dx_dy
    dx_dy_padded = torch.nn.utils.rnn.pad_sequence(
        dx_dy_list, batch_first=True, padding_value=0.0
    )
    
    # Create attention mask (1 for real tokens, 0 for PAD)
    attention_mask = (tokens_padded != PAD_TOKEN).float()
    
    return tokens_padded, dx_dy_padded, attention_mask

# Helper to avoid circular imports in some cases, though INV_VOCAB is in config
from config import INV_VOCAB
