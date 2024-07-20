.PHONY: all clean data train test setup lint pytest docs

# Default target
all: data train

# Setup Poetry environment and install dependencies
setup:
	poetry run pip install --upgrade pip
	poetry install
	mkdir -p data models logs

lock:
	poetry lock

# Generate JSONL files for training and testing
data: check-api-key
	poetry run python scripts/generate_data.py

# Train the model
train:
	poetry run python scripts/train.py

# Test the model
test: train
	poetry run python scripts/test.py

# Clean up generated files
clean:
	rm -rf data/*.jsonl
	rm -rf models/*
	rm -rf logs/*

# Run linter
lint: setup
	poetry run flake8 scripts/*.py

# Run tests
pytest: setup
	poetry run pytest tests/

# Generate documentation
docs: setup
	poetry run sphinx-apidoc -o docs scripts
	cd docs && poetry run make html

# Set OpenAI API key
set-api-key:
	@read -p "Enter your OpenAI API key: " api_key; \
	echo "export OPENAI_API_KEY=$$api_key" >> ~/.bashrc; \
	echo "API key has been added to ~/.bashrc. Please run 'source ~/.bashrc' to apply changes."

# Check if OPENAI_API_KEY is set
check-api-key:
	@if [ -z "$$OPENAI_API_KEY" ]; then \
		echo "OPENAI_API_KEY is not set. Please run 'make set-api-key' to set it."; \
		exit 1; \
	fi
