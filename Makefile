# Usage: make data | make train CONFIG=configs/b1_resnet50_wce.yaml | make debug CONFIG=... | make eval CONFIG=... | make table | make test
RUN = uv run

.PHONY: data train debug eval table test lint
data:
	$(RUN) python scripts/download_data.py
	$(RUN) python -m galaxy10.data

train:
	$(RUN) python -m galaxy10.train $(CONFIG)

debug:
	$(RUN) python -m galaxy10.train $(CONFIG) --debug --seeds 0

eval:
	$(RUN) python -m galaxy10.evaluate $(CONFIG) --figures $(if $(RESULTS),--results $(RESULTS))

table:
	$(RUN) python scripts/make_results_table.py

test:
	$(RUN) pytest -q

lint:
	$(RUN) ruff check
