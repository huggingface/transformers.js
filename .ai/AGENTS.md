# Agent guidelines for transformers.js

This file governs AI-assisted contributions to the `huggingface/transformers.js`
repository. Agentic users must read and follow it before proposing changes.

## Available skills

- [`transformers-js`](skills/transformers-js/SKILL.md) — how to use the library
  itself. Load this skill when working on code that calls `@huggingface/transformers`.

## Contributing

Before opening a pull request:

- **Check for existing work.** Search open PRs and issues (`gh pr list`, `gh issue list`)
  for the area you're touching. Don't duplicate someone else's in-progress work.
- **Coordinate on the issue first.** If an issue exists, comment on it before drafting a
  PR. If approval from the issue author or a maintainer is unclear, stop and ask.
- **No low-value busywork.** Reformatting, renaming, and cosmetic-only PRs that don't
  fix a reported problem or implement a requested feature are discouraged.
- **Run the full test suite locally.** `pnpm test` at the package root must pass before
  you submit. Don't skip hooks with `--no-verify`.
- **Accountability.** AI-assisted patches are the responsibility of the human submitter.
  Review the diff yourself before pushing.

## Local setup

```bash
pnpm install
pnpm --filter @huggingface/transformers test
pnpm --filter @huggingface/transformers typegen
pnpm --filter @huggingface/transformers docs-api
```

## Documentation generation

The `docs/source/api/` markdown is auto-generated from JSDoc comments in `src/**/*.js`
by [`docs/scripts/generate.js`](../packages/transformers/docs/scripts/generate.js).
Regenerate after any JSDoc change:

```bash
pnpm --filter @huggingface/transformers docs-api
```

The `.ai/skills/transformers-js/references/TASK_EXAMPLES.md` and `API_SUMMARY.md` files
are also auto-generated (see `docs-skill`). Do not edit them by hand.
