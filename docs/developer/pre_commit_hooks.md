# Pre-commit hooks

!!! note
    See [Git Hooks](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks) and [pre-commit](https://pre-commit.com/) for more details.

Pre-commit framework is a tool for managing pre-commit hooks. It's especially helpful to check your code before committing. It can be used to check code style, run tests, etc.

The general pipeline of the pre-commit framework looks like the figure below. Notice that we don't use `flake8` in this project, check [Hooks In Use](#hooks-in-use) for the available hooks.

![precommit pipeline](https://ljvmiranda921.github.io/assets/png/tuts/precommit_pipeline.png)

> Source: [Automate Python workflow using pre-commits: black and flake8](https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/)

To use pre-commit hook, you need to install it in your environment and set up the git hook scripts.

Install `pre-commit` in your environment.

```bash
pip install pre-commit
```

Run `pre-commit install` to set up the git hook scripts. After this command, `pre-commit` will run automatically on `git commit`.

```bash
pre-commit install
```

It's usually a good idea to run the hooks against all of the files while adding new hooks (usually `pre-commit` will only run on the changed files during git hooks)

```bash
pre-commit run --all-files
```

## Hooks In Use

- `trailing-whitespace`: Trims trailing whitespace.
- `end-of-file-fixer`: Makes sure files end in a newline and only a newline.
- `requirements-txt-fixer`: Sorts entries in requirements.txt and constraints.txt and removes incorrect entry for `pkg-resources==0.0.0`.
- `black`: Python code formatter. A tool to enforce a consistent coding style throughout a codebase.
- `ruff`: Python linter. A tool that examines the source code for potential issues, violations of coding standards, and possible errors.
- `mypy`: Python static type checker. A tool that helps identify and prevent type-related errors.
