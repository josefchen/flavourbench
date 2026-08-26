SHELL := /bin/bash
.SHELLFLAGS := -Eeuo pipefail -c

LATEXMK ?= latexmk

export SOURCE_DATE_EPOCH := 1787616000
export FORCE_SOURCE_DATE := 1
export TZ := UTC
export LC_ALL := C.UTF-8

.PHONY: all pdf verify clean

all: verify

pdf:
	$(LATEXMK) -g -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex

verify: pdf
	test "$$(pdfinfo main.pdf | awk '/^Pages:/ {print $$2}')" -eq 21
	test "$$(pdfinfo main.pdf | awk -F': +' '/^Page size:/ {print $$2}')" = '612 x 792 pts (letter)'
	! grep -E -i 'Overfull \\hbox|Overfull \\vbox|undefined references|Citation .* undefined' main.log
	pdffonts main.pdf | awk 'NR > 2 && $$(NF-4) != "yes" {exit 1}'

clean:
	$(LATEXMK) -C main.tex
