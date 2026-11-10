PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
DBT ?= .venv/bin/dbt
export PATH := /home/box/.local/bin:$(PATH)

.PHONY: setup data check bronze dbt quality train api dashboard

setup:
	python3 -m venv .venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) scripts/generate_data.py --seed 42

check:
	$(PYTHON) -m ruff check scripts tests
	$(PYTHON) -m pytest

bronze:
	$(PYTHON) scripts/load_bronze.py

# Build bronze + silver and run their tests. Gold/metrics are owned by
# Analytics Engineering: we attempt the build but do not fail this target
# or patch AE files if they error.
dbt: bronze
	cd dbt && ../$(DBT) seed --full-refresh --profiles-dir .
	cd dbt && ../$(DBT) run --select bronze silver --full-refresh --profiles-dir .
	cd dbt && ../$(DBT) test --select bronze silver path:tests/silver --exclude path:tests/gold gold metrics --profiles-dir .
	@echo "---- Attempting Analytics Engineering gold/metrics (best effort) ----"
	-cd dbt && ../$(DBT) run --select gold metrics --profiles-dir .
	-cd dbt && ../$(DBT) test --select gold metrics path:tests/gold --profiles-dir .
	@echo "---- End of gold/metrics attempt ----"

quality:
	$(PYTHON) scripts/quality_report.py

train:
	@echo "Placeholder: model training and scoring are scheduled for milestone M3."

api:
	@echo "Placeholder: scoring API is scheduled for milestone M4."

dashboard:
	$(PIP) install -r dashboard/requirements.txt
	MC_DUCKDB_PATH=$(CURDIR)/data/warehouse.duckdb $(PYTHON) -m streamlit run dashboard/app.py
