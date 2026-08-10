.PHONY: check plan

check:
	PYTHONPYCACHEPREFIX=/tmp/flavourbench-gpu-pyc python3 -m compileall -q common modal lambda_cloud
	python3 -m json.tool manifests/example-modal-qwen2.5-0.5b.json >/dev/null
	python3 -m json.tool manifests/example-lambda-qwen2.5-0.5b.json >/dev/null
	python3 common/manifest.py validate manifests/example-modal-qwen2.5-0.5b.json
	python3 common/manifest.py validate manifests/example-lambda-qwen2.5-0.5b.json
	python3 -m unittest discover -s tests -v

plan:
	python3 modal/deploy.py plan --manifest manifests/example-modal-qwen2.5-0.5b.json
	python3 lambda_cloud/controller.py plan \
		--manifest manifests/example-lambda-qwen2.5-0.5b.json \
		--max-runtime-minutes 30
