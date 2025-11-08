format:
	uv run ruff check --select I --fix .
	uv run ruff format .

run_benchmark:
	uv run python main.py

test:
	pytest test_implementations.py -v

test_quick:
	pytest test_implementations.py -v -k "basic"
