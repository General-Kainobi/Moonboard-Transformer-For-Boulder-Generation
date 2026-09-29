import torch
from typing import Dict, Tuple

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Hyperparameters
EMBEDDING_DIM = 256
NUM_LAYERS = 4
NUM_HEADS = 8
DROPOUT = 0.3
MAX_SEQ_LEN = 25  # A typical bouldering problem is never longer than this, forse mettere 15 anche gia vaben?
MAX_SPAN_UNITS = 7.0

# Training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50

# Special Tokens
PAD_TOKEN = 0
START_TOKEN = 1
END_TOKEN = 2

GRADES = ['<6A+>', '<6B>', '<6B+>', '<6C>', '<6C+>', '<7A>', '<7A+>', '<7B>', '<7B+>', '<7C>', '<7C+>', '<8A>', '<8A+>', '<8B>', '<8B+>']

# Vocabulary (A1 to K18) + Special tokens
# Token IDs will start from 3 + len(GRADES)

def generate_vocab() -> Dict[str, int]:
    """Generates a mapping from hold description (e.g. 'A1') to a discrete token ID."""
    vocab = {'<PAD>': PAD_TOKEN, '<START>': START_TOKEN, '<END>': END_TOKEN}
    
    token_id = 3
    for g in GRADES:
        vocab[g] = token_id
        token_id += 1
    columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    for col in columns:
        for row in range(1, 19):
            hold = f"{col}{row}"
            vocab[hold] = token_id
            token_id += 1
    return vocab

VOCAB = generate_vocab()
VOCAB_SIZE = len(VOCAB)
INV_VOCAB = {v: k for k, v in VOCAB.items()}

def hold_to_coords(hold: str) -> Tuple[int, int]:
    """
    Converts a hold string (e.g. 'A5') to (x, y) coordinates.
    A -> x=0, K -> x=10.
    1 -> y=0, 18 -> y=17.
    Returns (0, 0) for special tokens.
    """
    if hold in ['<PAD>', '<START>', '<END>'] or hold in GRADES:
        return (0, 0)
    
    col_str = hold[0]
    row_str = hold[1:]
    
    x = ord(col_str.upper()) - ord('A')
    y = int(row_str) - 1
    
    return (x, y)

def token_to_coords(token_id: int) -> Tuple[int, int]:
    """Converts a token ID back to (x, y) coordinates."""
    hold = INV_VOCAB.get(token_id, '<PAD>')
    if token_id in [PAD_TOKEN, START_TOKEN, END_TOKEN] or hold in GRADES:
        return (0, 0)
    return hold_to_coords(hold)
