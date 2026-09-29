import torch
import torch.nn.functional as F
from config import DEVICE, VOCAB, INV_VOCAB, START_TOKEN, PAD_TOKEN, END_TOKEN, MAX_SEQ_LEN, hold_to_coords
from transformer_model import MoonBoardTransformer
from csp_decoding import apply_csp_mask
import os
import numpy as np
import os
import copy
import pickle
import heapq
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.cbook as cbook
import re 
import PIL
def generate_problems(num_problems=5, temperature=1.0, grade='<7A>'):
    cwd = os.getcwd()
    parent_wd = cwd.replace('/model', '')
    model = MoonBoardTransformer().to(DEVICE)
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
        print("Loaded best_model.pth successfully!\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    
    for i in range(num_problems):
        tokens = []
        if grade in VOCAB:
            tokens.append(VOCAB[grade])
        tokens.append(START_TOKEN)
        
        with torch.no_grad():
            for step in range(MAX_SEQ_LEN):
                # Calculate dx, dy sequence
                coords = [hold_to_coords(INV_VOCAB.get(t, '<PAD>')) for t in tokens]
                dx_dy = [[0.0, 0.0]]
                for j in range(1, len(coords)):
                    prev_x, prev_y = coords[j-1]
                    curr_x, curr_y = coords[j]
                    dx_dy.append([float(curr_x - prev_x), float(curr_y - prev_y)])
                
                # Convert to tensors
                tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)
                dx_dy_tensor = torch.tensor(dx_dy, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                
                # No padding mask needed since batch size is 1 and we have no PAD tokens inside the sequence
                logits = model(tokens_tensor, dx_dy_tensor)
                
                # Get logits for the next token prediction (the last element in sequence)
                next_token_logits = logits[0, -1, :]
                
                # Apply CSP Mask to prevent repeats and out-of-span holds
                masked_logits = apply_csp_mask(next_token_logits, tokens[-1], tokens)
                
                # Apply temperature
                masked_logits = masked_logits / temperature
                
                # Convert to probabilities
                probs = F.softmax(masked_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                if next_token == PAD_TOKEN or next_token == END_TOKEN:
                    break
                    
                tokens.append(next_token)
                
        # Convert tokens to holds, ignore special tokens and grade tokens
        from config import GRADES
        holds = [INV_VOCAB[t] for t in tokens if t not in [START_TOKEN, PAD_TOKEN, END_TOKEN] and INV_VOCAB.get(t, '') not in GRADES]
        print(f"Problem {i+1} ({grade}): {' -> '.join(holds)}")
        print(holds, "\n")
        print(tokens, "\n")


""" Draw a moonboard problem on the layout"""
def plotAProblem(stringList, title = None, key = None):
    cwd = os.getcwd()
    parent_wd = cwd.replace('/model', '')
    image_file = cbook.get_sample_data(parent_wd + "/raw_data/upscalemedia-transformedcut.jpg")
    img = plt.imread(image_file)
    x = []
    y = []
    for hold in stringList:
        # Using re.findall() 
        # Splitting text and number in string  
        res = [re.findall(r'(\w+?)(\d+)', hold.split("-")[0])[0]] 
        
        alphabateList = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"] 
        ixInXAxis = alphabateList.index(res[0][0]) 
     
        x = x + [(90 + 52 * ixInXAxis)]# * img.shape[0] / 1024]
        y = y + [(1020 - 52 * int(res[0][1]))]# * img.shape[1] / 1024]

    # Create a figure. Equal aspect so circles look circular
    plt.rcParams["figure.figsize"] = (10,20)
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')

    # Show the image
    ax.imshow(img)
    #circle markers, green for start, blue for middle, red for end
    # Now, loop through coord arrays, and create a circle at each x,y pair
    count = 0
    for xx,yy in zip(x,y):
        if yy == 84:
            circ = plt.Circle((xx,yy), 30, color = 'r', fill=False, linewidth = 2)
        elif count < 2:
            circ = plt.Circle((xx,yy), 30, color = 'g', fill=False, linewidth = 2)
        else:
            circ = plt.Circle((xx,yy), 30, color = 'b', fill=False, linewidth = 2)
        ax.add_patch(circ)
        count = count + 1

    # Show the image
    if title:
        plt.title(title)
        # plt.savefig(key + '.jpg', dpi = 200)
    plt.show()
    # plt.savefig("generated_problem %i.jpg" %title)



if __name__ == '__main__':
    
    print("\nGenerating Extrapolated 8B+ problems...")
    generate_problems(num_problems=5, temperature=1.0, grade='<7A>')
