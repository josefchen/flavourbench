SHELL := /bin/bash
.SHELLFLAGS := -Eeuo pipefail -c
.ONESHELL:

export LC_ALL := C
export TZ := UTC
export SOURCE_DATE_EPOCH := 1786217820
export FORCE_SOURCE_DATE := 1

PYTHON ?= python3
RUFF ?= ruff
LATEXMK ?= latexmk
LOCK_FILE ?= /tmp/flavourbench-paper-$(shell id -u).lock

ifneq ($(wildcard ../src/flavourbench/epicure_native_taskset.py),)
SOURCE_ROOT := ../src/flavourbench
TEST_FILE := ../tests/epicure_native_taskset_test.py
else
SOURCE_ROOT := analysis/flavourbench
TEST_FILE := analysis/tests/epicure_native_taskset_test.py
endif

RELEASE := generated/epicure-native/epicure-native-release.json
GENERATED := generated/epicure-native
FIGURES := figures/epicure-native
ASSET_BUILDER := build_epicure_native_assets.py
REPLAY := reproduce_epicure_native.py

TABLE_ASSETS := \
	$(GENERATED)/epicure-native-macros.tex \
	$(GENERATED)/epicure-native-leaderboard-table.tex \
	$(GENERATED)/epicure-native-route-table.tex \
	$(GENERATED)/epicure-native-family-table.tex \
	$(GENERATED)/epicure-native-case-studies.tex \
	$(GENERATED)/epicure-native-case-studies.json

FIGURE_ASSETS := \
	$(FIGURES)/frontier-score-forest.pdf \
	$(FIGURES)/frontier-score-dumbbell.pdf \
	$(FIGURES)/frontier-family-heatmap.pdf \
	$(FIGURES)/frontier-paired-outcome-matrix.pdf \
	$(FIGURES)/frontier-latency-uplift.pdf \
	$(FIGURES)/frontier-social-summary.pdf

ANALYSIS_FILES := \
	$(SOURCE_ROOT)/__init__.py \
	$(SOURCE_ROOT)/epicure_native_taskset.py \
	$(SOURCE_ROOT)/epicure_native_leaderboard.py \
	$(SOURCE_ROOT)/epicure_native_release.py \
	$(SOURCE_ROOT)/frontier_manifest.py \
	$(TEST_FILE)

CORE_INPUTS := \
	main.tex references.bib README.md LICENSES.md Makefile \
	$(REPLAY) $(ASSET_BUILDER) $(RELEASE) $(TABLE_ASSETS) $(FIGURE_ASSETS) \
	$(ANALYSIS_FILES)

.PHONY: all assets verify paper _paper arxiv _arxiv clean

all: paper

assets:
	$(PYTHON) $(ASSET_BUILDER) \
		--release $(RELEASE) --generated $(GENERATED) --figures $(FIGURES)

verify: assets
	for file in $(CORE_INPUTS); do
		test -f "$$file"
		test ! -L "$$file"
	done
	/usr/bin/python3 -I $(REPLAY) --release $(RELEASE) >/dev/null
	$(PYTHON) -m pytest -q $(TEST_FILE)
	$(RUFF) check \
		$(SOURCE_ROOT)/epicure_native_taskset.py \
		$(SOURCE_ROOT)/epicure_native_leaderboard.py \
		$(SOURCE_ROOT)/epicure_native_release.py \
		$(REPLAY) $(ASSET_BUILDER)
	$(RUFF) format --check \
		$(SOURCE_ROOT)/epicure_native_taskset.py \
		$(SOURCE_ROOT)/epicure_native_leaderboard.py \
		$(SOURCE_ROOT)/epicure_native_release.py \
		$(REPLAY) $(ASSET_BUILDER)

paper:
	flock -x "$(LOCK_FILE)" $(MAKE) --no-print-directory _paper

_paper: verify
	mkdir -p build
	$(LATEXMK) -g -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
		-outdir=build main.tex
	install -m 0644 build/main.pdf build/flavourbench.pdf
	! grep -Fq 'Overfull \hbox' build/main.log
	test "$$(pdffonts build/flavourbench.pdf | awk 'NR > 2 && NF && $$(NF - 4) != "yes" {bad=1} END {print bad+0}')" = 0

arxiv:
	flock -x "$(LOCK_FILE)" $(MAKE) --no-print-directory _arxiv

_arxiv: _paper
	build_dir="$(CURDIR)/build"
	stage=$$(mktemp -d "$$build_dir/.arxiv-stage.XXXXXX")
	check=$$(mktemp -d "$$build_dir/.arxiv-check.XXXXXX")
	archive=$$(mktemp "$$build_dir/.flavourbench-arxiv-source.XXXXXX.tar.gz")
	cleanup() { rm -rf -- "$$stage" "$$check"; rm -f -- "$$archive"; }
	trap cleanup EXIT HUP INT TERM

	install -m 0644 main.tex references.bib README.md LICENSES.md Makefile \
		$(REPLAY) $(ASSET_BUILDER) "$$stage/"
	install -d -m 0755 \
		"$$stage/generated/epicure-native" \
		"$$stage/figures/epicure-native" \
		"$$stage/analysis/flavourbench" \
		"$$stage/analysis/tests"
	install -m 0644 $(RELEASE) $(TABLE_ASSETS) "$$stage/generated/epicure-native/"
	install -m 0644 $(FIGURE_ASSETS) "$$stage/figures/epicure-native/"
	install -m 0644 \
		$(SOURCE_ROOT)/__init__.py \
		$(SOURCE_ROOT)/epicure_native_taskset.py \
		$(SOURCE_ROOT)/epicure_native_leaderboard.py \
		$(SOURCE_ROOT)/epicure_native_release.py \
		$(SOURCE_ROOT)/frontier_manifest.py \
		"$$stage/analysis/flavourbench/"
	install -m 0644 $(TEST_FILE) \
		"$$stage/analysis/tests/"

	(
		cd "$$stage"
		find . -type f ! -name SOURCE_MANIFEST.sha256 -printf '%P\0' \
			| sort -z | xargs -0 sha256sum
	) > "$$stage/SOURCE_MANIFEST.sha256"
	if rg -l -I --pcre2 \
		"/hom[e]/|remy-simpc[4]|(?:AKIA|ASIA)[0-9A-Z]{16}|sk-(?:or-v1|kimi)-[A-Za-z0-9_-]{20,}|sk-(?:sp|ws)-[A-Za-z0-9._-]{20,}|[0-9a-f]{32}\.[A-Za-z0-9_-]{16,}|BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY" \
		"$$stage"; then
		echo "arXiv package refused: local path or credential marker present" >&2
		exit 1
	fi

	(
		cd "$$stage"
		tar --sort=name --format=ustar --mtime="@$(SOURCE_DATE_EPOCH)" \
			--owner=0 --group=0 --numeric-owner --mode='u+rwX,go+rX,go-w' \
			-cf - .
	) | gzip -n -9 > "$$archive"
	gzip -t "$$archive"
	tar -xzf "$$archive" -C "$$check"
	(
		cd "$$check"
		sha256sum --check SOURCE_MANIFEST.sha256
		mkdir -p build
		$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
			-outdir=build main.tex >/dev/null
		/usr/bin/python3 -I $(REPLAY) --release $(RELEASE) >/dev/null
	)
	cmp -s build/flavourbench.pdf "$$check/build/main.pdf"
	test "$$(pdffonts "$$check/build/main.pdf" | awk 'NR > 2 && NF && $$(NF - 4) != "yes" {bad=1} END {print bad+0}')" = 0

	chmod 0644 "$$archive"
	mv -f -- "$$archive" build/flavourbench-arxiv-source.tar.gz
	(
		cd build
		sha256sum flavourbench.pdf flavourbench-arxiv-source.tar.gz
	) > build/ARTIFACTS.sha256
	(
		cd build
		sha256sum --check ARTIFACTS.sha256
	)
	trap - EXIT HUP INT TERM

clean:
	rm -rf build
