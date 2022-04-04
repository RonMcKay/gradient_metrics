all: test build

build:
	poetry build

test: lint
	poetry run mypy gradient_metrics
	poetry run pytest --cov=gradient_metrics/ --cov-report=term-missing --cov-fail-under=90 tests

lint:
	poetry run flake8 . --max-line-length=88 --extend-exclude=.venv,venv --count --statistics

