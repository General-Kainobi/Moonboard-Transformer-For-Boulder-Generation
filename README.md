# MoonBoard Transformer for Boulder Generation

This repository contains a PyTorch-based Transformer model designed to generate bouldering problems for the **MoonBoard (Masters 2019 40° layout)**.

The model learns from a dataset of existing MoonBoard problems and generates new problems conditioned on a target grade (e.g., `<7A>`, `<8B+>`). It features a unique **Constraint-Satisfaction Problem (CSP) decoding** mechanism that ensures the generated problems respect real-world physical constraints.

## Key Features

- **Transformer Architecture**: Uses a PyTorch Transformer with positional encodings and relative coordinate tracking (`dx`, `dy`) between holds to model movement.
- **Conditional Generation**: Start generation by prompting the model with a specific difficulty grade.
- **CSP Decoding (Constraint-Satisfaction)**: Enforces physical climbing rules during inference. 
  - Prevents the generation of the same hold multiple times.
  - Prevents physically impossible spans between holds based on Euclidean distance limits.
  - Masks out special padding/start tokens during generation.
- **Attention Analysis**: Includes tools (`attention_analysis.py`) to visualize the model's self-attention weights, helping to interpret which previous holds the model considers most important when choosing the next hold.

## File Structure

- `transformer_model.py`: Defines the `MoonBoardTransformer` architecture.
- `train_supervised.py`: Script to train the model using cross-entropy loss with early stopping and learning rate scheduling.
- `dataset.py`: Defines the `MoonBoardDataset` for loading, parsing, and tokenizing JSON data of existing problems.
- `generate.py`: Script for generating new problems using the trained model (`best_model.pth`).
- `csp_decoding.py`: Contains the logic for the `apply_csp_mask` function, which filters out physically impossible or invalid next tokens.
- `config.py`: Central configuration file housing hyperparameters, vocabulary mapping, and token definitions.
- `attention_analysis.py`: Tooling to extract and visualize the attention matrices for interpretability.
- `train_grader.py`: (Optional/WIP) Contains logic to potentially train a network for grading existing problems.

## Setup and Usage

### Prerequisites
- Python 3.8+
- PyTorch
- Scikit-learn, Matplotlib, Seaborn, Pandas (for analysis and visualizations)

### Training
To train the model from scratch on the dataset (`problems MoonBoard Masters 2019 40.json`), run:
```bash
python train_supervised.py
```
This will train the model and save the best performing checkpoint as `best_model.pth`.

### Generation
To generate new problems, ensure you have a trained `best_model.pth` and run:
```bash
python generate.py
```
By default, this will sample new problems from the model, applying the CSP mask to ensure the generation is anatomically valid. You can adjust the `temperature` and `grade` parameters within the script.

## How the CSP Masking Works
During autoregressive generation, the `apply_csp_mask` function intercepts the raw logits output by the transformer before they are passed into the Softmax sampling step. It forcefully sets the logits of invalid holds to `-inf`, guaranteeing the model will not sample them. This prevents trivial errors such as generating the same hold twice or generating a hold that is mathematically too far from the previous holds (exceeding maximum anatomical span).
