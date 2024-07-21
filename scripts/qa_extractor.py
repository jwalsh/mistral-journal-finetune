import os
import json
from typing import List, Dict, Union
import click
import nltk
from nltk.tokenize import sent_tokenize
import requests
from pypdf import PdfReader

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_txt(txt_path: str) -> str:
    """
    Extract text from a TXT file.

    Args:
        txt_path (str): Path to the TXT file.

    Returns:
        str: Content of the TXT file.
    """
    with open(txt_path, 'r', encoding='utf-8') as file:
        return file.read()

def segment_text(text: str) -> List[str]:
    """
    Segment the input text into sentences.

    Args:
        text (str): Input text to be segmented.

    Returns:
        List[str]: List of segmented sentences.
    """
    return sent_tokenize(text)

def extract_qa_pairs(segments: List[str]) -> List[Dict[str, str]]:
    """
    Extract question-answer pairs from segmented text.

    Args:
        segments (List[str]): List of text segments.

    Returns:
        List[Dict[str, str]]: List of question-answer pairs.
    """
    qa_pairs = []
    for i in range(0, len(segments) - 1, 2):
        question = segments[i].strip()
        answer = segments[i+1].strip()
        if question.endswith('?'):
            qa_pairs.append({"input": question, "output": answer})
    return qa_pairs

def process_file(file_path: str) -> List[Dict[str, str]]:
    """
    Process a single file (PDF or TXT) and extract question-answer pairs.

    Args:
        file_path (str): Path to the file to be processed.

    Returns:
        List[Dict[str, str]]: List of extracted question-answer pairs.

    Raises:
        ValueError: If the file format is not supported.
    """
    if file_path.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.txt'):
        text = extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    segments = segment_text(text)
    return extract_qa_pairs(segments)

def send_to_claude_api(qa_pairs: List[Dict[str, str]], api_key: str, model: str) -> None:
    """
    Send extracted question-answer pairs to the Claude API.

    Args:
        qa_pairs (List[Dict[str, str]]): List of question-answer pairs.
        api_key (str): Claude API key.
        model (str): Claude model to use.
    """
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key
    }
    
    for qa_pair in qa_pairs:
        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(qa_pair)
                }
            ]
        }
        response = requests.post(url, headers=headers, json=data)
        click.echo(f"Sent: {qa_pair}")
        click.echo(f"Response: {response.status_code}")
        click.echo(response.json())

@click.command()
@click.option('--input-dir', type=click.Path(exists=True), help='Path to the input directory containing PDF and TXT files.')
@click.option('--api-key', envvar='CLAUDE_API_KEY', help='Claude API key. Can also be set via CLAUDE_API_KEY environment variable.')
@click.option('--model', default='claude-3-sonnet-20240229', help='Claude model to use.')
def main(input_dir: str, api_key: str, model: str) -> None:
    """
    Process PDF and TXT files in the input directory, extract question-answer pairs,
    and send them to the Claude API.
    """
    if not api_key:
        raise click.UsageError("API key must be provided either via --api-key option or CLAUDE_API_KEY environment variable.")

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        try:
            click.echo(f"Processing {filename}...")
            qa_pairs = process_file(file_path)
            send_to_claude_api(qa_pairs, api_key, model)
        except Exception as e:
            click.echo(f"Error processing {filename}: {str(e)}", err=True)

if __name__ == "__main__":
    main()
