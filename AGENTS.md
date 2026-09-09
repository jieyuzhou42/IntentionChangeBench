# General Agent Rules

These rules apply to every agent working on the IntentionChange project and its related repositories.

## 1. Compute job names must not identify the project

- When requesting compute resources or submitting scheduler jobs, including CPU or GPU jobs, use a generic job name such as `sihengx-p4-dev`.
- Never include project, repository, experiment, dataset, customer, or other confidential identifiers in a job name.

## 2. Never push code

- Never run `git push` under any circumstances.
- This project contains confidential company code and must never be uploaded to a public repository or any unauthorized remote service.
- Do not publish, mirror, paste, or otherwise transmit project code, credentials, logs, or artifacts to public or unapproved external services.
- Local Git operations such as `git status`, `git diff`, and local commits are allowed when needed, but all remote uploads are prohibited.

## 3. Prefer disconnect-resilient background execution

- Assume that the user's network connection and interactive session may be interrupted.
- Run long-running builds, tests, downloads, analyses, simulations, and other suitable tasks in the background whenever practical.
- Prefer resilient mechanisms such as a cluster scheduler, `nohup`, `tmux`, or an equivalent job runner, and write output to a local log file.
- Record enough information to resume or inspect the task later, including the command, log path, process or job identifier, and current status.
- Do not background short commands when doing so would add unnecessary complexity or hide failures.
