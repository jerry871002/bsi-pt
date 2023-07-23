# Pre-commit hook

!!! note
    See [pre-commit](https://pre-commit.com/) for more details.

Pre-commit hook is a tool that helps you to check your code before committing. It can be used to check code style, run tests, etc.

To use pre-commit hook, you need to install it in your environment and set up the git hook scripts.

Install `pre-commit` in your environment.

```bash
pip install pre-commit
```

Run `pre-commit install` to set up the git hook scripts. After this command, `pre-commit` will run automatically on `git commit`.

```bash
pre-commit install
```

It's usually a good idea to run the hooks against all of the files for the first time or when adding new hooks (usually `pre-commit` will only run on the changed files during git hooks)

```bash
pre-commit run --all-files
```
