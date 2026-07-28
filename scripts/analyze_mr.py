#!/usr/bin/env python3
"""Generate a single combined merge-request diff report."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import tempfile
from pathlib import Path

import analyze_mr_call_graph as call_graph
import analyze_mr_structure as structure
import analyze_mr_words as words


GENERATED_MARKDOWN_FILES = {
    "before_words.md",
    "after_words.md",
    "diff_words.md",
    "before_files.md",
    "after_files.md",
    "diff_files.md",
    "before_functions.md",
    "after_functions.md",
    "diff_functions.md",
    "before_structs.md",
    "after_structs.md",
    "diff_structs.md",
    "before_call_graph.md",
    "after_call_graph.md",
    "diff_call_graph.md",
}


@dataclasses.dataclass(frozen=True)
class PreparedInputs:
    before_dir: Path
    after_dir: Path
    before_revision: str
    after_revision: str
    repo_url: str
    mr_url: str
    include_tests: bool


@dataclasses.dataclass(frozen=True)
class DiffSummary:
    area: str
    added: int
    removed: int
    changed: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch or read a Go merge request, analyze it with Tree-sitter, "
            "and leave only a combined diff.md in the output directory."
        )
    )
    parser.add_argument("--mr-url", help="Merge request URL to analyze")
    parser.add_argument(
        "--mr-ref-name",
        default=words.DEFAULT_MR_REF_NAME,
        help="MR ref suffix to fetch, default: tmp-squash",
    )
    parser.add_argument("--work-dir", type=Path, help="Temporary workspace for fetches and worktrees")
    parser.add_argument("--before")
    parser.add_argument("--after")
    parser.add_argument("--before-dir", type=Path)
    parser.add_argument("--after-dir", type=Path)
    parser.add_argument("--output-dir", default=Path("."), type=Path)
    parser.add_argument("--repo-url", help="Repository URL for report metadata when not using --mr-url")
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include Go unit test files. By default, *_test.go files are skipped.",
    )
    return parser


def prepare_inputs(args: argparse.Namespace) -> PreparedInputs:
    if args.before_dir and args.after_dir:
        if not args.before or not args.after:
            raise SystemExit("--before-dir/--after-dir mode requires --before and --after")
        return PreparedInputs(
            before_dir=args.before_dir,
            after_dir=args.after_dir,
            before_revision=args.before,
            after_revision=args.after,
            repo_url=args.repo_url or "",
            mr_url=args.mr_url or "",
            include_tests=args.include_tests,
        )

    if args.mr_url:
        mr_info = words.parse_mr_url(args.mr_url, args.mr_ref_name)
        prepared = words.prepare_mr_revisions(
            mr_info,
            args.work_dir or words.default_work_dir(mr_info),
            args.before,
            args.after,
        )
        return PreparedInputs(
            before_dir=prepared.before_dir,
            after_dir=prepared.after_dir,
            before_revision=prepared.before_revision,
            after_revision=prepared.after_revision,
            repo_url=args.repo_url or mr_info.repo_url,
            mr_url=mr_info.mr_url,
            include_tests=args.include_tests,
        )

    raise SystemExit("Provide --mr-url or both --before-dir and --after-dir.")


def strip_style_block(lines: list[str]) -> list[str]:
    stripped: list[str] = []
    in_style = False
    for line in lines:
        if line.strip() == "<style>":
            in_style = True
            continue
        if in_style:
            if line.strip() == "</style>":
                in_style = False
            continue
        stripped.append(line)
    return stripped


def strip_component_preamble(lines: list[str]) -> list[str]:
    body = strip_style_block(lines)
    if body and body[0].startswith("# "):
        body = body[1:]
    while body and not body[0].strip():
        body = body[1:]
    while body and body[0].startswith("- "):
        body = body[1:]
    while body and not body[0].strip():
        body = body[1:]
    return body


def demote_headings(lines: list[str]) -> list[str]:
    return [f"#{line}" if line.startswith("#") else line for line in lines]


def component_section(title: str, path: Path) -> list[str]:
    body = strip_component_preamble(path.read_text().splitlines())
    return [f"## {title}", "", *demote_headings(body)]


def write_combined_diff(
    path: Path,
    inputs: PreparedInputs,
    summaries: list[DiffSummary],
    component_paths: list[tuple[str, Path]],
) -> None:
    lines = [
        "# MR Diff",
        "",
        f"- Repository: `{inputs.repo_url}`",
        f"- Merge request: {inputs.mr_url}",
        f"- Before revision: `{inputs.before_revision}`",
        f"- After revision: `{inputs.after_revision}`",
        f"- Generated at: `{dt.datetime.now(dt.UTC).isoformat()}`",
        f"- Unit test files included: `{str(inputs.include_tests).lower()}`",
        "",
        "## Summary",
        "",
        "| area | added | removed | changed |",
        "| --- | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.area} | {summary.added} | {summary.removed} | {summary.changed} |"
        )

    for title, component_path in component_paths:
        lines.extend(["", *component_section(title, component_path)])

    lines.append("")
    words.write_markdown(path, lines)


def cleanup_generated_markdown(output_dir: Path, final_path: Path) -> None:
    final_resolved = final_path.resolve()
    for file_name in GENERATED_MARKDOWN_FILES:
        path = output_dir / file_name
        if path.exists() and path.resolve() != final_resolved:
            path.unlink()


def analyze_to_diff(inputs: PreparedInputs, output_dir: Path) -> list[DiffSummary]:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "diff.md"

    with tempfile.TemporaryDirectory(prefix=".mr-digest-", dir=output_dir) as temp_name:
        temp_dir = Path(temp_name)

        before_words = words.analyze_directory(
            inputs.before_dir,
            inputs.before_revision,
            "Before",
            include_tests=inputs.include_tests,
        )
        after_words = words.analyze_directory(
            inputs.after_dir,
            inputs.after_revision,
            "After",
            include_tests=inputs.include_tests,
        )
        words.write_diff_md(temp_dir / "diff_words.md", before_words, after_words, mr_url=inputs.mr_url)
        before_word_set = set(before_words.word_counts)
        after_word_set = set(after_words.word_counts)
        changed_words = {
            word
            for word in before_word_set & after_word_set
            if before_words.word_counts[word] != after_words.word_counts[word]
        }

        before_structure = structure.analyze_directory(
            inputs.before_dir,
            inputs.before_revision,
            "Before",
            include_tests=inputs.include_tests,
        )
        after_structure = structure.analyze_directory(
            inputs.after_dir,
            inputs.after_revision,
            "After",
            include_tests=inputs.include_tests,
        )
        structure.write_file_diff_md(
            temp_dir / "diff_files.md",
            before_structure,
            after_structure,
            mr_url=inputs.mr_url,
        )
        structure.write_function_diff_md(
            temp_dir / "diff_functions.md",
            before_structure,
            after_structure,
            mr_url=inputs.mr_url,
        )
        structure.write_struct_diff_md(
            temp_dir / "diff_structs.md",
            before_structure,
            after_structure,
            mr_url=inputs.mr_url,
        )

        before_calls = call_graph.analyze_directory(
            inputs.before_dir,
            inputs.before_revision,
            "Before",
            include_tests=inputs.include_tests,
        )
        after_calls = call_graph.analyze_directory(
            inputs.after_dir,
            inputs.after_revision,
            "After",
            include_tests=inputs.include_tests,
        )
        call_graph.write_call_graph_diff_md(
            temp_dir / "diff_call_graph.md",
            before_calls,
            after_calls,
            mr_url=inputs.mr_url,
        )

        summaries = [
            DiffSummary(
                "words",
                len(after_word_set - before_word_set),
                len(before_word_set - after_word_set),
                len(changed_words),
            ),
            DiffSummary(
                "files",
                len(structure.missing_entries(after_structure.files, before_structure.files)),
                len(structure.missing_entries(before_structure.files, after_structure.files)),
                len(structure.changed_entries(before_structure.files, after_structure.files)),
            ),
            DiffSummary(
                "functions",
                len(structure.missing_entries(after_structure.functions, before_structure.functions)),
                len(structure.missing_entries(before_structure.functions, after_structure.functions)),
                len(structure.changed_entries(before_structure.functions, after_structure.functions)),
            ),
            DiffSummary(
                "structs",
                len(structure.missing_entries(after_structure.structs, before_structure.structs)),
                len(structure.missing_entries(before_structure.structs, after_structure.structs)),
                len(structure.changed_entries(before_structure.structs, after_structure.structs)),
            ),
            DiffSummary(
                "internal call relations",
                len(call_graph.missing_relations(after_calls.relations, before_calls.relations)),
                len(call_graph.missing_relations(before_calls.relations, after_calls.relations)),
                len(call_graph.changed_relations(before_calls.relations, after_calls.relations)),
            ),
        ]

        write_combined_diff(
            final_path,
            inputs,
            summaries,
            [
                ("Words", temp_dir / "diff_words.md"),
                ("Files", temp_dir / "diff_files.md"),
                ("Functions", temp_dir / "diff_functions.md"),
                ("Structs", temp_dir / "diff_structs.md"),
                ("Internal Call Graph", temp_dir / "diff_call_graph.md"),
            ],
        )

    cleanup_generated_markdown(output_dir, final_path)
    return summaries


def main() -> None:
    args = build_parser().parse_args()
    inputs = prepare_inputs(args)
    summaries = analyze_to_diff(inputs, args.output_dir)
    for summary in summaries:
        print(
            f"{summary.area} added/removed/changed: "
            f"{summary.added}/{summary.removed}/{summary.changed}"
        )
    print(f"wrote: {args.output_dir / 'diff.md'}")


if __name__ == "__main__":
    main()
