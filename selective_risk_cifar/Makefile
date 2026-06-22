.PHONY: test smoke-cert docker-build docker-test docker-smoke docker-cifar-smoke preflight git-start git-end

test:
	python tests/test_certify.py

# This does not train; it only validates the certification script on fake logits.
smoke-cert:
	python scripts/smoke_certification_with_fake_logits.py

docker-build:
	bash scripts/docker_build.sh

docker-test:
	bash scripts/docker_test.sh

docker-smoke:
	bash scripts/docker_smoke.sh

docker-cifar-smoke:
	bash scripts/docker_cifar_smoke_train.sh

preflight:
	bash scripts/preflight_repo.sh

git-start:
	bash scripts/git_start_day.sh

git-end:
	bash scripts/git_end_day.sh "daily srcc experiment update"
