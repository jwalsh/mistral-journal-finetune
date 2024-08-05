import json
import random
from typing import Dict, Any
from uuid import uuid4
from datetime import datetime
import click
from faker import Faker

fake = Faker()

def generate_random_prompt_response() -> Dict[str, Dict[str, str]]:
    """
    Generate a random prompt and response for an LLM dataset entry.
    
    Returns:
        Dict[str, Dict[str, str]]: A dictionary with 'inputs' and 'outputs' keys.
    """
    prompt = fake.sentence(nb_words=10)
    response = fake.sentence(nb_words=15)
    
    return {
        "inputs": {"input": prompt},
        "outputs": {"output": response}
    }

def generate_dataset_entry() -> Dict[str, Any]:
    """
    Generate a single dataset entry for an LLM dataset.
    
    Returns:
        Dict[str, Any]: A dictionary representing the dataset entry.
    """
    return {
        "name": fake.word(),
        "description": fake.sentence(),
        "created_at": datetime.utcnow().isoformat(),
        "data_type": "llm",
        "inputs_schema_definition": None,
        "outputs_schema_definition": None,
        "externally_managed": False,
        "id": str(uuid4()),
        "tenant_id": str(uuid4()),
        "example_count": random.randint(1, 1000),
        "session_count": random.randint(1, 100),
        "modified_at": datetime.utcnow().isoformat(),
        "last_session_start_time": datetime.utcnow().isoformat(),
        **generate_random_prompt_response()
    }

@click.command()
@click.argument('output_file', type=click.Path())
@click.option('--num_entries', default=10, help='Number of entries to generate.')
def generate_jsonl(output_file: str, num_entries: int):
    """
    Generate a JSONL file with randomly generated LLM dataset entries.
    
    Args:
        output_file (str): The path to the output JSONL file.
        num_entries (int): The number of entries to generate.
    """
    with open(output_file, 'w') as f:
        for _ in range(num_entries):
            entry = generate_dataset_entry()
            f.write(json.dumps(entry) + '\n')

if __name__ == '__main__':
    generate_jsonl()
