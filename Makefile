PYTHON ?= python
PYTEST ?= pytest
RUFF ?= ruff

RELEASE := paper/$(shell awk '$$1 == "RELEASE" && $$2 == ":=" {print $$3}' paper/Makefile.powered)

PUBLIC_TEST_FILES := \
	tests/epicure_native_powered*_test.py \
	tests/epicure_native_taskset_v*_test.py \
	tests/epicure_selection_*_test.py \
	tests/frontier_refresh_*_test.py \
	tests/build_powered*_test.py \
	tests/build_joint_powered_dataset_test.py \
	tests/build_lab_dataset_test.py \
	tests/reward_transfer_plan_test.py \
	tests/reward_transfer_execution_test.py \
	tests/lab_test.py \
	tests/lab_cli_test.py \
	tests/lab_runner_test.py \
	tests/hf_lab_space_api_test.py \
	tests/hf_powered_space_app_test.py \
	tests/reproduce_powered_release_test.py \
	tests/restore_powered_runs_test.py \
	tests/restore_joint_powered_runs_test.py \
	tests/external_substitution_validation_artifact_test.py \
	tests/public_scorer_sensitivity_artifact_test.py \
	tests/selection_robustness_artifact_test.py \
	tests/selection_response_parser_v3_test.py \
	tests/test_service_internals.py \
	tests/test_service_zai.py

PUBLIC_SOURCE_FILES := \
	src/flavourbench/epicure_native_powered*.py \
	src/flavourbench/epicure_native_taskset_v*.py \
	src/flavourbench/epicure_selection_*.py \
	src/flavourbench/frontier_refresh_*.py \
	paper/build_complete_core_assets.py \
	paper/build_external_substitution_validation_assets.py \
	paper/build_public_scorer_sensitivity_assets.py \
	paper/build_powered_selection_assets.py \
	paper/build_selection_robustness_assets.py \
	paper/build_stability_assets.py \
	paper/reproduce_complete_core_release.py \
	paper/reproduce_powered_release.py \
	paper/verify_complete_core_release.py \
	paper/iclr2027/package_submission.py \
	paper/iclr2027/verify_submission.py \
	paper/iclr2027/supplement/rebuild_summary.py \
	hf/dataset/build_complete_core_dataset.py \
	hf/dataset/build_lab_dataset.py \
	hf/dataset/build_joint_powered_dataset.py \
	hf/dataset/build_powered_dataset.py \
	hf/dataset/restore_complete_core_sources.py \
	hf/dataset/restore_joint_powered_runs.py \
	hf/dataset/restore_powered_runs.py \
	hf/dataset/verify_complete_core_dataset.py \
	hf/space/build_complete_core_space_bundle.py \
	hf/space/build_powered_space_bundle.py \
	hf/space/lab_api.py \
	hf/space/app.py \
	src/flavourbench/lab.py \
	src/flavourbench/lab_cli.py \
	src/flavourbench/lab_runner.py \
	examples/lab/train_sft.py \
	examples/lab/train_dpo.py \
	examples/lab/train_grpo.py \
	experiments/reward_transfer/audit_data.py \
	experiments/reward_transfer/unlock_evaluation.py \
	experiments/reward_transfer/evaluate.py \
	experiments/reward_transfer/analyze.py \
	experiments/reward_transfer/release_results.py \
	experiments/reward_transfer/train_sft.py

PUBLIC_LINT_FILES := $(PUBLIC_SOURCE_FILES) scripts/scan_public_release.py $(PUBLIC_TEST_FILES)

.PHONY: ci format lab-data lint reward-transfer-audit scan test verify-artifacts verify-python verify-release

ci: verify-release test verify-python lab-data reward-transfer-audit verify-artifacts lint scan

verify-release:
	test -n "$(RELEASE)"
	test -f "$(RELEASE)"
	test ! -L "$(RELEASE)"
	$(PYTHON) -I paper/verify_complete_core_release.py --release "$(RELEASE)"

test:
	$(PYTEST) -q $(PUBLIC_TEST_FILES)

verify-python:
	$(PYTHON) -m py_compile $(PUBLIC_SOURCE_FILES)

lab-data:
	$(PYTHON) hf/dataset/build_lab_dataset.py --check

reward-transfer-audit:
	$(PYTHON) experiments/reward_transfer/audit_data.py >/dev/null

verify-artifacts:
	(cd paper/build && sha256sum --check ARTIFACTS.sha256)
	gzip -t paper/build/flavourbench-arxiv-source.tar.gz
	stage="$$(mktemp -d)"; \
	trap 'rm -rf -- "$$stage"' EXIT; \
	tar -xzf paper/build/flavourbench-arxiv-source.tar.gz -C "$$stage"; \
	(cd "$$stage" && sha256sum --check SOURCE_MANIFEST.sha256)
	(cd paper/iclr2027/build && sha256sum --check ICLR2027-MANIFEST.sha256)
	gzip -t paper/iclr2027/build/flavourbench-iclr2027-anonymous-source.tar.gz
	unzip -q -t paper/iclr2027/build/flavourbench-iclr2027-anonymous-supplement.zip
	stage="$$(mktemp -d)"; \
	trap 'rm -rf -- "$$stage"' EXIT; \
	tar -xzf paper/iclr2027/build/flavourbench-iclr2027-anonymous-source.tar.gz \
		-C "$$stage"; \
	root="$$stage/flavourbench-iclr2027-anonymous-source"; \
	(cd "$$root" && sha256sum --check MANIFEST.sha256 && $(MAKE) verify); \
	cmp "$$root/main.pdf" paper/iclr2027/build/flavourbench-iclr2027-anonymous.pdf
	stage="$$(mktemp -d)"; \
	trap 'rm -rf -- "$$stage"' EXIT; \
	unzip -q paper/iclr2027/build/flavourbench-iclr2027-anonymous-supplement.zip \
		-d "$$stage"; \
	root="$$stage/flavourbench-iclr2027-anonymous-supplement"; \
	(cd "$$root" && sha256sum --check MANIFEST.sha256); \
	$(PYTHON) "$$root/code/verify_dataset.py" \
		--dataset-directory "$$root/data/complete-core"; \
	$(PYTHON) "$$root/code/rebuild_summary.py" \
		--dataset-directory "$$root/data/complete-core" \
		--output-directory "$$root/reconstructed"

lint:
	$(RUFF) check $(PUBLIC_LINT_FILES)
	$(RUFF) format --check $(PUBLIC_LINT_FILES)

format:
	$(RUFF) format $(PUBLIC_LINT_FILES)

scan:
	$(PYTHON) scripts/scan_public_release.py
