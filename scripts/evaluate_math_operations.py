import click
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from typing import List

@click.command()
@click.option("--data", default="random_math_operations_dataset", help="Dataset to evaluate on")
@click.option("--experiment-prefix", default="random_math_operations_dataset", help="Experiment prefix")
def evaluate_math_operations(data: str, experiment_prefix: str) -> None:
    """
    Evaluates the math operations task on the specified dataset.

    Args:
        data (str): Dataset to evaluate on.
        experiment_prefix (str): Experiment prefix.
    """
    # Define the prompt template for evaluating math operations
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a math expert. Please process the following mathematical expression and provide the result. If the expression is invalid or cannot be computed, respond with 'undefined'."),
        ("user", "{input}")
    ])

    # Initialize the chat model and output parser
    chat_model = ChatOpenAI()
    output_parser = StrOutputParser()

    # Create the evaluation chain
    chain = prompt | chat_model | output_parser

    # Define the evaluators for the target task
    evaluators: List[LangChainStringEvaluator] = [
        LangChainStringEvaluator("cot_qa"),
        LangChainStringEvaluator("labeled_criteria", config={"criteria": "conciseness"})
    ]

    # Run the evaluation
    results = evaluate(
        chain.invoke,
        data=data,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
    )

if __name__ == "__main__":
    evaluate_math_operations()
