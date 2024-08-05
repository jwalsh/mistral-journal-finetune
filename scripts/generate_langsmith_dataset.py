import json
import csv
import random
from typing import Dict, Any
from uuid import uuid4
from datetime import datetime
import click
from faker import Faker

fake = Faker()

def generate_random_math_expression() -> str:
    num_operands = random.randint(1, 9)
    operator_list = ['+', '-', '*', '/']
    expression = f"{random.choice(operator_list)} " + " ".join(str(random.randint(-9, 9)) for _ in range(num_operands))
    return expression

def calculate_expression(expression: str) -> Any:
    tokens = expression.split()
    op = tokens[0]
    operands = list(map(int, tokens[1:]))
    
    try:
        if op == '+':
            result = sum(operands)
        elif op == '-':
            result = operands[0]
            for num in operands[1:]:
                result -= num
        elif op == '*':
            result = 1
            for num in operands:
                result *= num
        elif op == '/':
            result = operands[0]
            for num in operands[1:]:
                if num == 0:
                    return None  # Represents undefined in this context
                result /= num
        return round(result, 2) if result is not None else None
    except (ZeroDivisionError, ValueError):
        return None

def generate_random_math_entry() -> Dict[str, Any]:
    expression = generate_random_math_expression()
    result = calculate_expression(expression)
    return {
        "inputs": {"input": expression},
        "outputs": {"output": result if result is not None else "null"}
    }

def write_csv(output_file: str, num_entries: int):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f, quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Input", "Output"])
        for _ in range(num_entries):
            entry = generate_random_math_entry()
            # Ensure that the expression is quoted properly
            writer.writerow([f'"{entry["inputs"]["input"]}"', entry["outputs"]["output"]])

def write_jsonl(output_file: str, num_entries: int):
    with open(output_file, 'w') as f:
        for _ in range(num_entries):
            entry = generate_random_math_entry()
            f.write(json.dumps(entry) + '\n')

@click.command()
@click.argument('output_file', type=click.Path())
@click.option('--num_entries', default=100, help='Number of entries to generate.')
@click.option('--format', default='csv', type=click.Choice(['csv', 'jsonl']), help='Output format.')
def generate_dataset(output_file: str, num_entries: int, format: str):
    """
    Generate a dataset in the specified format.
    
    Args:
        output_file (str): The path to the output file.
        num_entries (int): The number of entries to generate.
        format (str): The output format, either 'csv' or 'jsonl'.
    """
    if format == 'csv':
        write_csv(output_file, num_entries)
    elif format == 'jsonl':
        write_jsonl(output_file, num_entries)

if __name__ == '__main__':
    generate_dataset()
