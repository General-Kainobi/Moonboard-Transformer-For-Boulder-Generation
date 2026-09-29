import torch
import math
from typing import List, Tuple
from config import VOCAB_SIZE, PAD_TOKEN, START_TOKEN, END_TOKEN, GRADES, MAX_SPAN_UNITS, token_to_coords

def apply_csp_mask(logits: torch.Tensor, current_token_id: int, previous_tokens: List[int]) -> torch.Tensor:
    """
    Applies Constraint-Satisfaction (Neuro-Symbolic) masking to the logits.
    
    Args:
        logits: Tensor of shape [vocab_size] representing the un-normalized predictions for the NEXT token.
        current_token_id: The ID of the current token we are transitioning FROM.
        previous_tokens: List of all previous token IDs in the current sequence.
        
    Returns:
        torch.Tensor: Masked logits.
    """
    masked_logits = logits.clone()
    
    # No Repetition: Set logit of the immediately previous to -inf.
    if current_token_id not in [PAD_TOKEN, START_TOKEN]:
        masked_logits[current_token_id] = float('-inf')
    
    for pt in previous_tokens:
        if pt not in [PAD_TOKEN, START_TOKEN]:
            masked_logits[pt] = float('-inf')
            
    # Also mask PAD and start tokens, they shouldn't be generated in the middle of a sequence.
    masked_logits[PAD_TOKEN] = float('-inf')
    masked_logits[START_TOKEN] = float('-inf')
    
    #Maximum Anatomical Span
    # Euclidean distance for every token in vocabulary.
    if current_token_id not in [PAD_TOKEN, START_TOKEN]:
        cx, cy = token_to_coords(current_token_id)
        
        for token_idx in range(VOCAB_SIZE):
            if token_idx in [PAD_TOKEN, START_TOKEN, END_TOKEN] or token_idx >= 3 and token_idx < 3 + len(GRADES):
                continue
                
            # If already masked, skip
            if masked_logits[token_idx] == float('-inf'):
                continue
                
            tx, ty = token_to_coords(token_idx)
            dist = math.sqrt((tx - cx)**2 + (ty - cy)**2)
            
            if dist >= MAX_SPAN_UNITS:
                masked_logits[token_idx] = float('-inf')
                
    return masked_logits
