.PHONY: test inspect
test:
	python -m pytest -q
inspect:
	python -m applied_ai.cli inspect --input $(INPUT)
