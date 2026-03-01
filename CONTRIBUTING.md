# Contributing

Thanks for your interest in contributing to `litellm-a2a-settlement`.

This package integrates [LiteLLM](https://github.com/BerriAI/litellm) with the [A2A Settlement Exchange](https://github.com/a2a-settlement/a2a-settlement). It follows the same governance and contribution patterns as the core a2a-settlement org.

## Where to contribute

- **Bugs**: open an Issue with reproduction steps.
- **Feature requests**: open a Discussion first if it affects the API surface.
- **Integration improvements**: PRs welcome for `litellm_a2a_settlement/`.

## How to propose a new feature

1. Open a GitHub Discussion with a description and motivation.
2. Get feedback from maintainers.
3. Submit an implementation PR with tests.
4. Link to any related a2a-settlement spec changes if applicable.

## Development setup

Use a clean Python environment (venv/conda):

```bash
pip install -e ".[dev]"
```

## Running tests

```bash
python -m pytest -q
```

With coverage:

```bash
python -m pytest --cov=litellm_a2a_settlement --cov-report=xml --cov-report=term-missing -q
```

To mirror the GitHub Actions CI job locally (including the a2a-settlement SDK Git install):

```bash
./ci-local.sh
```

## Code style

- Keep code small and readable.
- Avoid heavy dependencies unless they materially improve clarity or correctness.
