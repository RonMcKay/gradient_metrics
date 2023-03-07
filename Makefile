.PHONY: all setup install test clean build docs test test-flake8 test-black test-isort format

setup: install

install:
	@echo "Installing dependencies..."
	poetry install

all: test build

build: docs
	poetry build

docs:
	(cd docs/; poetry run make html)

test: test-flake8 test-black test-isort test-mypy test-pytest
	@echo "All tests passed successfully!"

test-mypy:
	@echo "Testing with mypy..."
	poetry run mypy gradient_metrics

test-pytest:
	@echo "Testing with pytest..."
	poetry run pytest --cov

test-flake8:
	@echo "Checking format with flake8..."
	poetry run flake8 . --count --statistics

test-black:
	@echo "Checking format with black..."
	poetry run black --check gradient_metrics tests

test-isort:
	@echo "Checking format with isort..."
	poetry run isort --check --settings-path pyproject.toml gradient_metrics tests

format:
	@echo "Formatting with black and isort..."
	poetry run black tests gradient_metrics examples
	poetry run isort --settings-path pyproject.toml tests gradient_metrics examples

clean:
	rm -rfv dist/
	(cd docs/; poetry run make clean)
