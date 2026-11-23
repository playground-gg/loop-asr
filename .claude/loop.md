# Maintenance loop for loop-asr

Each round, in this order. While the PR is open, schedule the next round before you finish
(sooner after a push, later when all is green); stop only once it is merged or closed.

1. Finish any work still open in this conversation.
2. Tend the current branch's PR (`gh pr view --json number,state,mergeable,statusCheckRollup,reviews`):
   - CI red: read the failed job's log (`gh run view <id> --log-failed`), fix the root cause,
     run the project's gate (`uv run ruff check . && uv run ruff format --check . && uv run pytest`;
     through the kit's `gate.py verify` where the kit is installed, so the run is recorded), push.
     After 3 rounds on the same failure, stop and leave one comment on the PR: what fails and why
     you did not fix it.
   - Review comments: apply the mechanical ones and reply on each thread. A product, design or
     experiment-design question goes to Giovanni Giofrè as a comment; do not decide it yourself.
   - Conflict with main: sync the way /kit:go does (rebase a branch only you push to, merge a
     shared one), resolve, run the gate, push.
   - Green with nothing open: say so in one line and schedule the next round at a longer
     interval. Never merge.
3. Clean up: nothing left behind (`git status`); worktrees of merged PRs removed.

Do not touch other PRs, open new ones, start training runs, download data, or deploy anything.
Reply in one line: what changed, or "nothing", and nothing more.
