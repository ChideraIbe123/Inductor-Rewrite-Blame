PY := .venv/bin/python
DEPS := torch==2.14.1 torchvision transformers scikit-learn numpy matplotlib tabulate pytest

setup:
	uv venv -p 3.12 .venv
	uv pip install -p $(PY) $(DEPS)
	$(PY) -m rewrite_blame discover

test-fast:
	$(PY) -m pytest -m "unit or inductor" -q

test-unit:
	$(PY) -m pytest -m unit -q

test-all:
	$(PY) -m pytest -q

discover:
	$(PY) -m rewrite_blame discover

.PHONY: setup test-fast test-unit test-all discover
