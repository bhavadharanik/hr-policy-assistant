.PHONY: install dev eval test clean

install:
	pip3 install -r requirements.txt

dev:
	python3 -m src.main "$(QUESTION)"

eval:
	python3 -m evals.run_evals

test:
	python3 -m pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
