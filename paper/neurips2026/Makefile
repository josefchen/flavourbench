SHELL := /bin/bash
.SHELLFLAGS := -Eeuo pipefail -c

PYTHON ?= python3
LATEXMK ?= latexmk
PDFINFO ?= pdfinfo
PDFTOTEXT ?= pdftotext

export SOURCE_DATE_EPOCH := 1786217820
export FORCE_SOURCE_DATE := 1
export TZ := UTC
export LC_ALL := C.UTF-8

MAIN := main
BUILD := build
PDF := $(BUILD)/flavourbench-neurips2026.pdf

.PHONY: all sync checklist pdf verify package clean

all: verify

sync:
	$(PYTHON) sync_from_arxiv.py

checklist: checklist-answers.tex

checklist-answers.tex: checklist.tex prepare_checklist.py
	$(PYTHON) prepare_checklist.py

pdf: checklist
	mkdir -p $(BUILD)
	BIBINPUTS="$(abspath ..):" $(LATEXMK) -g -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
	  -outdir=$(BUILD) $(MAIN).tex
	cp $(BUILD)/$(MAIN).pdf $(PDF)

verify: pdf
	$(PYTHON) sync_from_arxiv.py --check
	test "$$($(PDFINFO) $(PDF) | awk '/^Pages:/ {print $$2}')" -ge 1
	test "$$($(PDFINFO) $(PDF) | awk '/^File size:/ {print $$3}')" -lt 52428800
	! $(PDFTOTEXT) $(PDF) - | grep -F 'Anonymous Authors'
	! grep -R -n -E '\\answerTODO|\\justificationTODO' checklist-answers.tex
	$(PYTHON) verify_submission.py

package: verify
	$(PYTHON) package_submission.py
	cd $(BUILD) && sha256sum --check SUBMISSION-MANIFEST.sha256

clean:
	rm -rf $(BUILD) checklist-answers.tex
