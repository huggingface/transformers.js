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
pnpm --filter @huggingface/transformers docs-generate
```

## Documentation generation

Run the full documentation generator after any JSDoc, docs snippet, task metadata,
or generated skill content change:

```bash
pnpm --filter @huggingface/transformers docs-generate
```

`docs-generate` runs [`docs/scripts/generate-all.js`](../packages/transformers/docs/scripts/generate-all.js),
which generates:

- `packages/transformers/docs/source/api/**/*.md` from JSDoc comments in
  `packages/transformers/src/**/*.js`.
- Generated sections in `.ai/skills/transformers-js/SKILL.md`.
- `.ai/skills/transformers-js/references/TASKS.md`.
- The repo-root `README.md` from `packages/transformers/docs/snippets/*.snippet`.

Do not edit generated API pages, the generated skill reference files, or the
root README by hand. Update the source JSDoc, docs snippets, or generator
modules instead.

The generator also validates generated API pages against `docs/source/_toctree.yml`
and checks local Markdown links and anchors under `docs/source/`. To run only
the validator (without regenerating), use `pnpm --filter @huggingface/transformers docs-validate`.

### Workflow after touching JSDoc

1. Edit the JSDoc in `packages/transformers/src/**/*.js`.
2. Run `pnpm --filter @huggingface/transformers docs-generate` from the repo root.
3. Inspect the diff under `packages/transformers/docs/source/api/` — that's the
   page a human or agent will read. If the JSDoc didn't render the way you
   expected, tweak the source, not the generated markdown.
4. Watch for "doc-quality warnings" at the end of the generator's output —
   these flag undocumented public exports and malformed `@param` tags.
5. Stage the source files; the generated `api/**/*.md` files are gitignored.

### What is generated vs. hand-written

| Path | Generated? |
|------|------------|
| `packages/transformers/docs/source/api/**/*.md` | yes — gitignored |
| `packages/transformers/docs/source/{guides,tutorials,integrations}/*.md` | hand-written |
| `packages/transformers/docs/source/_toctree.yml` | hand-written (must list every generated API page) |
| `.ai/skills/transformers-js/SKILL.md` | mixed — prose hand-written, blocks between `<!-- @generated:start ... -->` markers regenerated |
| `.ai/skills/transformers-js/references/TASKS.md` | fully generated, do not edit |
| `.ai/skills/transformers-js/references/{CONFIGURATION,PIPELINE_OPTIONS}.md` | mixed (same marker convention as SKILL.md) |
| `README.md` | generated from `packages/transformers/docs/snippets/*.snippet` |
