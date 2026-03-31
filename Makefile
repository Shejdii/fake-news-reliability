install:

	pip install -r requirements.txt

data:
	python -m src.pipeline.run_data_pipeline

baseline:
	python -m src.pipeline.train_baseline

test:
	pytest -v -q

test-unit:
	pytest -q tests/test_label_mapping.py

test-smoke:
	pytest -q tests/test_data_smoke.py

test-contract:
	pytest -q tests/test_data_contract.py

test-data:
	pytest -q tests/test_data_smoke.py tests/test_data_contract.py

lint:
	pylint src || true

format:
	black .

check: format lint test

pipeline: check