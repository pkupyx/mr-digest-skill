---
name: mr-digest-skill
description: Analyze Go merge requests with Tree-sitter and generate a single Markdown diff report covering identifier words, files, functions, structs, and internal call relations. Use when asked to analyze a merge request, summarize MR code changes, or produce MR diff artifacts while skipping Go unit tests by default.
---

# MR Digest Skill

Use `scripts/analyze_mr.py` for the normal workflow. It writes a combined `diff.md`
and removes intermediate report shards after generation.

Example:

```bash
PYTHONPATH=/path/to/tree-sitter-deps python3 scripts/analyze_mr.py \
  --mr-url https://code.byted.org/group/project/merge_requests/123/commits \
  --mr-ref-name tmp-merge \
  --output-dir /private/tmp/mr-digest-reports/project-mr123
```

By default, `*_test.go` files are skipped. Pass `--include-tests` only when the
user explicitly asks to include Go unit tests.
