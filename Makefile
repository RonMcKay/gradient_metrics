.PHONY: all setup pre-commit install test clean build docs test-format test-format-flake8 test-format-black test-format-isort format

setup: install pre-commit

install:
	@echo "Installing dependencies..."
	poetry install

pre-commit: install
	@echo "Setting up pre-commit..."
	poetry run pre-commit install -t commit-msg -t pre-commit

all: test build

build: docs
	poetry build

docs:
	(cd docs/; poetry run make html)

test: test-format
	poetry run mypy gradient_metrics
	poetry run pytest --cov=gradient_metrics/ --cov-report=term-missing --cov-fail-under=90 tests

test-format: test-format-flake8 test-format-black test-format-isort

test-format-flake8:
	@echo "Checking format with flake8..."
	poetry run flake8 . --count --statistics

test-format-black:
	@echo "Checking format with black..."
	poetry run black --check gradient_metrics tests

test-format-isort:
	@echo "Checking format with isort..."
	poetry run isort --check --settings-path pyproject.toml gradient_metrics tests

format:
	@echo "Formatting with black and isort..."
	poetry run black
	poetry run isort --settings-path pyproject.toml

clean:
	rm -rfv dist/
	(cd docs/; poetry run make clean)
