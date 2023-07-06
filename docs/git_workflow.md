# Git Workflow

## Commit Message

Use all-lower-cased messages with a prefix. The prefix could be one of the following

- `fix`: bug fix
- `feat`: new feature
- `docs`: documents related
- `style`: code style related
- `refactor`: code refactor
- `perf`: tune performance
- `test`: test
- `chore`: other stuff

The title should be a manageable length. Details can be added in the body. For example, a commit message like this

```
feat: added new features for user authentication and authorization, including login, registration, password reset, and role-based access control
```

is probably a bad idea. And it could be improved as

```
feat: implement user authentication and authorization

- add login, registration, and password reset
- enforce role-based access control
```
