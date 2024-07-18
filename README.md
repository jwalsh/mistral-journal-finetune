# Mistral Journal Fine-tuning

This project fine-tunes the Mistral 7B model on personal journal entries using QLoRA (Quantized Low-Rank Adaptation).

## Setup

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Prepare your data:
   - Place your training data in `data/notes.jsonl`
   - Place your validation data in `data/notes_validation.jsonl`

3. Run the training script:
   ```
   python scripts/train.py
   ```

## Project Structure

- `data/`: Contains the training and validation datasets
- `notebooks/`: Jupyter notebooks for exploration and analysis
- `scripts/`: Python scripts for training and inference
- `models/`: Directory to store fine-tuned models
- `logs/`: Training logs

## License

This project is licensed under the MIT License.
