# Contributing to geometer

Thank you for your interest in contributing to **geometer**! This project is a Python library for computational geometry, leveraging projective geometry and NumPy for fast geometric computations. Your contributions help improve the library and make it more robust and versatile.

## Table of Contents

* [Getting Started](#getting-started)
* [Installing Dependencies](#installing-dependencies)
* [Running Tests](#running-tests)
* [Code Style and Linting](#code-style-and-linting)
* [Submitting Changes](#submitting-changes)
* [Additional Resources](#additional-resources)

## Getting Started

To contribute to geometer, please follow these steps:

1. **Fork the Repository**: Click the "Fork" button on the [Geometer GitHub page](https://github.com/jan-mue/geometer) to create your own copy of the repository.

2. **Clone Your Fork**:

   ```bash
   git clone https://github.com/your-username/geometer.git
   cd geometer
   ```

3. **Add Upstream Remote**:

   ```bash
   git remote add upstream https://github.com/jan-mue/geometer.git
   ```

## Installing Dependencies

Geometer uses [uv](https://astral.sh/blog/uv/) for managing dependencies. To set up your development environment:

1. **Install `uv`**: Follow the instructions on the [uv installation guide](https://astral.sh/blog/uv/).

2. **Sync Dependencies**:

   ```bash
   uv sync
   ```

   This command will install all necessary dependencies as specified in `pyproject.toml`.

## Running Tests

Geometer uses `pytest` for testing. To run the test suite:

```bash
uv run pytest
```

Ensure that all tests pass before submitting your changes.

## Code Style and Linting

Geometer uses [Ruff](https://beta.ruff.rs/docs/) for code style enforcement and linting, configured as a pre-commit hook.

1. **Install Pre-commit Hooks**:

   ```bash
   pre-commit install
   ```

2. **Run Pre-commit Hooks**:

   ```bash
   pre-commit run --all-files
   ```

This will automatically format your code and check for linting issues.

## Submitting Changes

1. **Create a New Branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**: Implement your feature or bug fix.

3. **Commit Your Changes**:

   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

   Use clear and descriptive commit messages.

4. **Push to Your Fork**:

   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request**: Navigate to your fork on GitHub and open a pull request against the `main` branch of the original repository.

## Additional Resources

* [Geometer Documentation](https://geometer.readthedocs.io/)

We appreciate your contributions and look forward to your pull requests!
