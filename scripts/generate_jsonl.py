import json
import random
from typing import List, Dict, Any
from uuid import uuid4
from datetime import datetime
import click
from faker import Faker

fake = Faker()

OPERATORS = ['+', '-', '*', '/']

def generate_random_expression() -> Dict[str, Any]:
    """Generate a random mathematical expression and its result."""
    operand1 = random.randint(1, 100)
    operand2 = random.randint(1, 100)
    operator = random.choice(OPERATORS)
    expression = f"({operator} {operand1} {operand2})"
    
    if operator == '+':
        result = operand1 + operand2
    elif operator == '-':
        result = operand1 - operand2
    elif operator == '*':
        result = operand1 * operand2
    elif operator == '/':
        result = operand1 / operand2 if operand2 != 0 else None  # Avoid division by zero
    
    return {
        "expression": expression,
        "result": result
    }

def generate_dataset_entry() -> Dict[str, Any]:
    """Generate a single dataset entry."""
    return {
        "name": fake.word(),
        "description": fake.sentence(),
        "created_at": datetime.utcnow().isoformat(),
        "data_type": "kv",
        "inputs_schema_definition": None,
        "outputs_schema_definition": None,
        "externally_managed": False,
        "id": str(uuid4()),
        "tenant_id": str(uuid4()),
        "example_count": random.randint(1, 1000),
        "session_count": random.randint(1, 100),
        "modified_at": datetime.utcnow().isoformat(),
        "last_session_start_time": datetime.utcnow().isoformat(),
        "random_expression": generate_random_expression()
    }

@click.command()
@click.argument('output_file', type=click.Path())
@click.option('--num_entries', default=10, help='Number of entries to generate.')
def generate_jsonl(output_file: str, num_entries: int):
    """
    Generate a JSONL file with randomly generated dataset entries.
    
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
