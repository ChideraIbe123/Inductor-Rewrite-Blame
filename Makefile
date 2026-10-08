PY := .venv/bin/python

test-fast:
	$(PY) -m pytest -m "unit or inductor" -q

test-unit:
	$(PY) -m pytest -m unit -q

test-all:
	$(PY) -m pytest -q

discover:
	$(PY) -m rewrite_blame discover

report:
	cd report/midterm && tectonic midterm.tex

.PHONY: test-fast test-unit test-all discover report
