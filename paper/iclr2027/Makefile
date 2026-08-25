SHELL := /bin/bash
.SHELLFLAGS := -Eeuo pipefail -c

PYTHON ?= python3
LATEXMK ?= latexmk
PDFINFO ?= pdfinfo

export SOURCE_DATE_EPOCH := 1787616000
export FORCE_SOURCE_DATE := 1
export TZ := UTC
export LC_ALL := C.UTF-8

BUILD := build
STAGE := $(BUILD)/stage
PDF := $(BUILD)/flavourbench-iclr2027-anonymous.pdf

GENERATED := \
	complete-core-macros.tex \
	complete-core-stability-macros.tex \
	complete-core-robustness-macros.tex \
	complete-core-public-scorer-macros.tex \
	complete-core-external-substitution-validation-macros.tex \
	complete-core-leaderboard-table.tex \
	complete-core-route-table.tex \
	complete-core-task-table.tex \
	complete-core-family-table.tex \
	complete-core-score-sensitivity-table.tex \
	complete-core-public-scorer-table.tex \
	complete-core-external-substitution-validation-table.tex \
	complete-core-examples-table.tex

FIGURES := \
	complete-core-leaderboard-forest.pdf \
	complete-core-selection-robustness.pdf \
	complete-core-public-scorer-sensitivity.pdf \
	complete-core-external-substitution-validation.pdf \
	complete-core-panel-replication.pdf \
	complete-core-task-count-stability.pdf \
	complete-core-pairwise-matrix.pdf \
	complete-core-family-heatmap.pdf \
	complete-core-rank-intervals.pdf

STYLE := iclr2027_conference.sty iclr2027_conference.bst natbib.sty fancyhdr.sty

.PHONY: all prepare pdf verify package clean

all: verify

prepare:
	mkdir -p $(STAGE)/generated $(STAGE)/figures
	install -m 0644 main.tex $(STYLE) $(STAGE)/
	install -m 0644 ../references.bib $(STAGE)/references.bib
	for asset in $(GENERATED); do \
		install -m 0644 "../generated/complete-core/$$asset" "$(STAGE)/generated/$$asset"; \
	done
	for figure in $(FIGURES); do \
		install -m 0644 "../figures/complete-core/$$figure" "$(STAGE)/figures/$$figure"; \
	done

pdf: prepare
	cd $(STAGE) && $(LATEXMK) -g -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex
	install -m 0644 $(STAGE)/main.pdf $(PDF)

verify: pdf
	$(PYTHON) verify_submission.py

package: verify
	$(PYTHON) package_submission.py

clean:
	rm -rf $(BUILD)
