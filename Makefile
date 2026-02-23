PYTHON ?= /Library/Frameworks/Python.framework/Versions/3.11/bin/python3
CONFIG ?= config.json

.PHONY: run test clean

run:
	$(PYTHON) run_pipeline.py --config $(CONFIG)

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py" -v

clean:
	rm -rf __pycache__ src/ld_preprocessing/__pycache__ tests/__pycache__
