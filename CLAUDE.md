# CLAUDE.md

Architecture, mission, and status live in [PROJECT_SPEC.md](PROJECT_SPEC.md) — that
file is the single source of truth for this project. This file only holds
operating instructions for Claude Code sessions.

## Git workflow (standing authorization)

The user does not want to manage git manually for this repo. Commit and push
are pre-authorized — do not ask for confirmation each time:

- Commit after completing and testing each module/fix, per PROJECT_SPEC.md
  Section 8's "finish one module, test it, commit it" discipline. Don't batch
  unrelated changes into one commit.
- Push to `origin master` after committing, so GitHub stays current as a
  backup. This repo has no branch-protection workflow — pushing straight to
  `master` is expected here.
- Never commit `Output/` (real private book OCR output) — it's gitignored on
  purpose per PROJECT_SPEC.md Section 2's privacy rules. Don't add it back.
- Still never force-push, never rewrite published history, and still ask
  before any destructive git operation (reset --hard, discarding uncommitted
  work, deleting branches).
