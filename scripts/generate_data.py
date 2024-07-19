import json
import os
from typing import List, Dict

import click
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

def generate_journal_entry() -> str:
    """
    Generate a single journal entry using OpenAI's API.

    Returns:
        str: A synthetic journal entry.
    """
    prompt = "Write a short, personal journal entry about a day in the life of a machine learning / Python developer. Include technical notes, research, thoughts, feelings, and activities."
    
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",  # Using GPT-3.5-turbo-instruct as a replacement for text-davinci-002
        prompt=prompt,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.8,
    )
    
    return response.choices[0].text.strip()

def create_dataset(num_entries: int) -> List[Dict[str, str]]:
    """
    Create a dataset of journal entries.

    Args:
        num_entries (int): The number of journal entries to generate.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing a journal entry.
    """
    dataset = []
    with click.progressbar(range(num_entries), label='Generating entries') as bar:
        for _ in bar:
            entry = generate_journal_entry()
            dataset.append({"note": entry})
    return dataset

def save_jsonl(data: List[Dict[str, str]], filename: str) -> None:
    """
    Save data to a JSONL file.

    Args:
        data (List[Dict[str, str]]): The data to save.
        filename (str): The name of the file to save the data to.
    """
    with open(filename, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')

@click.command()
@click.option('--train-size', default=150, help='Number of entries in the training dataset.')
@click.option('--val-size', default=50, help='Number of entries in the validation dataset.')
@click.option('--output-dir', default='data', help='Directory to save the generated datasets.')
def main(train_size: int, val_size: int, output_dir: str) -> None:
    """
    Generate synthetic journal entries and save them as training and validation datasets.

    This script uses the OpenAI API to generate journal entries and saves them in JSONL format.
    The datasets are split into training and validation sets.

    Args:
        train_size (int): Number of entries in the training dataset.
        val_size (int): Number of entries in the validation dataset.
        output_dir (str): Directory to save the generated datasets.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate and save training data
    train_data = create_dataset(train_size)
    train_file = os.path.join(output_dir, 'notes.jsonl')
    save_jsonl(train_data, train_file)
    click.echo(f"Created training dataset with {len(train_data)} entries in {train_file}")

    # Generate and save validation data
    val_data = create_dataset(val_size)
    val_file = os.path.join(output_dir, 'notes_validation.jsonl')
    save_jsonl(val_data, val_file)
    click.echo(f"Created validation dataset with {len(val_data)} entries in {val_file}")

if __name__ == "__main__":
    main()
