import json
import random
from typing import Dict, Any, List
from uuid import uuid4
from datetime import datetime
import click
from faker import Faker

fake = Faker()

def generate_random_chat_message() -> str:
    """
    Generate a random chat message.
    
    Returns:
        str: A serialized chat message.
    """
    num_messages = random.randint(1, 5)
    messages = []
    for _ in range(num_messages):
        sender = fake.first_name()
        timestamp = fake.date_time_this_year().isoformat()
        message = fake.sentence(nb_words=12)
        messages.append(f"[{timestamp}] {sender}: {message}")
    return " ".join(messages)

def generate_random_chat_entry() -> Dict[str, Dict[str, str]]:
    """
    Generate a random chat input and output for a chat dataset entry.
    
    Returns:
        Dict[str, Dict[str, str]]: A dictionary with 'inputs' and 'outputs' keys.
    """
    input_chat = generate_random_chat_message()
    output_chat = generate_random_chat_message()
    
    return {
        "inputs": {"input": input_chat},
        "outputs": {"output": output_chat}
    }

def generate_dataset_entry() -> Dict[str, Any]:
    """
    Generate a single dataset entry for a chat dataset.
    
    Returns:
        Dict[str, Any]: A dictionary representing the dataset entry.
    """
    return {
        "name": fake.word(),
        "description": fake.sentence(),
        "created_at": datetime.utcnow().isoformat(),
        "data_type": "chat",
        "inputs_schema_definition": None,
        "outputs_schema_definition": None,
        "externally_managed": False,
        "id": str(uuid4()),
        "tenant_id": str(uuid4()),
        "example_count": random.randint(1, 1000),
        "session_count": random.randint(1, 100),
        "modified_at": datetime.utcnow().isoformat(),
        "last_session_start_time": datetime.utcnow().isoformat(),
        **generate_random_chat_entry()
    }

@click.command()
@click.argument('output_file', type=click.Path())
@click.option('--num_entries', default=10, help='Number of entries to generate.')
def generate_jsonl(output_file: str, num_entries: int):
    """
    Generate a JSONL file with randomly generated chat dataset entries.
    
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
