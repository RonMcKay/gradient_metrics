.PHONY: all test clean build docs

all: test build

build: docs
	poetry build

docs:
	(cd docs/; poetry run make html)

test: lint
	poetry run mypy gradient_metrics
	poetry run pytest --cov=gradient_metrics/ --cov-report=term-missing --cov-fail-under=90 tests

lint:
	poetry run flake8 . --max-line-length=88 --extend-exclude=.venv,venv --count --statistics

clean:
	rm -rfv dist/
	(cd docs/; poetry run make clean)
