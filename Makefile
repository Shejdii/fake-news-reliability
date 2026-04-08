install:
	pip install -r requirements.txt

data:
	python -m src.pipeline.run_data_pipeline

baseline:
	python -m src.pipeline.train_baseline

distilbert:
	python -m src.pipeline.train_distilbert

demo-confidence:
	python -m src.pipeline.demo_confidence

compare-confidence:
	python -m src.pipeline.compare_confidence

test:
	pytest -v -q -m "not integration"

test-unit:
	pytest -q tests/test_label_mapping.py

test-smoke:
	pytest -q tests/test_data_smoke.py

test-contract:
	pytest -q tests/test_data_contract.py

test-data:
	pytest -q tests/test_data_smoke.py tests/test_data_contract.py

test-confidence:
	pytest -q tests/test_confidence_utils.py tests/test_baseline_confidence.py tests/test_distilbert_confidence.py

debug-data:
	python -m src.pipeline.debug_data

lint:
	-pylint src

format:
	black .

format-check:
	black --check .

check: format-check lint test

pipeline: check data baseline