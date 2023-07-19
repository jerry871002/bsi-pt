# Git Workflow

## Pull Request

The `master` branch is protected and commits can only be added via a reviewed PR.

## Commit Message

> See [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for more details

Use all-lower-cased messages with a prefix. The prefix can be one of the following

- `fix`: bug fix
- `feat`: new feature
- `doc`: add/edit documents
- `ci`: construct CI pipeline
- `style`: adjust code style
- `refactor`: code refactor
- `perf`: tune performance
- `test`: test
- `ci`: CI pipeline
- `chore`: other stuff

The title should be a manageable length. Details can be added in the body. For example, a commit message like this is probably a bad idea.

```
feat: added new features for user authentication and authorization, including login, registration, password reset, and role-based access control
```

And it can be improved as

```
feat: implement user authentication and authorization

- add login, registration, and password reset
- enforce role-based access control
```
