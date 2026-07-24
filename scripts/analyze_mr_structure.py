#!/usr/bin/env python3
"""Analyze file, function, and struct diffs in a Go merge request."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
from pathlib import Path
from typing import Iterable

from analyze_mr_words import (
    DEFAULT_MR_REF_NAME,
    MRInfo,
    default_work_dir,
    is_test_file,
    iter_nodes,
    make_parser,
    md_escape,
    parse_mr_url,
    prepare_mr_revisions,
)


@dataclasses.dataclass(frozen=True)
class FileEntry:
    path: str
    line_count: int
    byte_count: int
    digest: str

    @property
    def key(self) -> str:
        return self.path


@dataclasses.dataclass(frozen=True)
class FunctionEntry:
    path: str
    name: str
    kind: str
    receiver: str
    start_line: int
    end_line: int
    digest: str

    @property
    def key(self) -> str:
        return f"{self.path}::{self.name}"


@dataclasses.dataclass(frozen=True)
class StructEntry:
    path: str
    name: str
    fields: tuple[str, ...]
    start_line: int
    end_line: int
    digest: str

    @property
    def key(self) -> str:
        return f"{self.path}::{self.name}"


@dataclasses.dataclass
class StructureAnalysis:
    name: str
    revision: str
    include_tests: bool
    files: list[FileEntry]
    functions: list[FunctionEntry]
    structs: list[StructEntry]
    parse_errors: list[str]


def source_digest(source: bytes) -> str:
    return hashlib.sha256(source).hexdigest()[:16]


def node_text(source: bytes, node) -> str:
    return source[node.start_byte : node.end_byte].decode(errors="replace")


def node_digest(source: bytes, node) -> str:
    return source_digest(source[node.start_byte : node.end_byte])


def list_go_files(root: Path, *, include_tests: bool) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.go")
        if ".git" not in path.relative_to(root).parts
        and (include_tests or not is_test_file(path))
    )


def receiver_name(source: bytes, method_node) -> str:
    receiver = method_node.child_by_field_name("receiver")
    if receiver is None:
        return ""
    receiver_text = node_text(source, receiver).strip()
    for node in iter_nodes(receiver):
        if node.type == "type_identifier":
            type_name = node_text(source, node)
            pointer = "*" if "*" in receiver_text[: receiver_text.find(type_name)] else ""
            return f"{pointer}{type_name}"
    return receiver_text


def function_name(source: bytes, node) -> tuple[str, str, str]:
    name_node = node.child_by_field_name("name")
    raw_name = node_text(source, name_node) if name_node is not None else "<anonymous>"
    if node.type == "method_declaration":
        receiver = receiver_name(source, node)
        display_name = f"({receiver}).{raw_name}" if receiver else raw_name
        return display_name, "method", receiver
    return raw_name, "function", ""


def extract_field_names(source: bytes, struct_node) -> tuple[str, ...]:
    fields: list[str] = []
    for node in iter_nodes(struct_node):
        if node.type == "field_declaration":
            names = [
                node_text(source, child)
                for child in node.children
                if child.type == "field_identifier"
            ]
            if names:
                fields.extend(names)
            else:
                field_text = " ".join(node_text(source, node).split())
                if field_text:
                    fields.append(field_text)
    return tuple(fields)


def extract_struct_name(source: bytes, type_spec_node) -> str:
    name_node = type_spec_node.child_by_field_name("name")
    return node_text(source, name_node) if name_node is not None else "<anonymous>"


def analyze_directory(root: Path, revision: str, name: str, *, include_tests: bool) -> StructureAnalysis:
    parser = make_parser()
    files: list[FileEntry] = []
    functions: list[FunctionEntry] = []
    structs: list[StructEntry] = []
    parse_errors: list[str] = []

    for file_path in list_go_files(root, include_tests=include_tests):
        relative_path = file_path.relative_to(root).as_posix()
        source = file_path.read_bytes()
        line_count = source.count(b"\n") + (0 if source.endswith(b"\n") else 1)
        files.append(
            FileEntry(
                path=relative_path,
                line_count=line_count,
                byte_count=len(source),
                digest=source_digest(source),
            )
        )

        tree = parser.parse(source)
        if tree.root_node.has_error:
            parse_errors.append(relative_path)

        for node in iter_nodes(tree.root_node):
            if node.type in {"function_declaration", "method_declaration"}:
                name_value, kind, receiver = function_name(source, node)
                functions.append(
                    FunctionEntry(
                        path=relative_path,
                        name=name_value,
                        kind=kind,
                        receiver=receiver,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        digest=node_digest(source, node),
                    )
                )
                continue

            if node.type == "type_spec":
                type_node = node.child_by_field_name("type")
                if type_node is None or type_node.type != "struct_type":
                    continue
                structs.append(
                    StructEntry(
                        path=relative_path,
                        name=extract_struct_name(source, node),
                        fields=extract_field_names(source, type_node),
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        digest=node_digest(source, node),
                    )
                )

    return StructureAnalysis(
        name=name,
        revision=revision,
        include_tests=include_tests,
        files=sorted(files, key=lambda entry: entry.path),
        functions=sorted(functions, key=lambda entry: (entry.path, entry.name)),
        structs=sorted(structs, key=lambda entry: (entry.path, entry.name)),
        parse_errors=parse_errors,
    )


def index_by_key(entries: Iterable) -> dict[str, object]:
    return {entry.key: entry for entry in entries}


def changed_entries(before_entries: Iterable, after_entries: Iterable) -> list[tuple[object, object]]:
    before_by_key = index_by_key(before_entries)
    after_by_key = index_by_key(after_entries)
    changed: list[tuple[object, object]] = []
    for key in sorted(before_by_key.keys() & after_by_key.keys()):
        before_entry = before_by_key[key]
        after_entry = after_by_key[key]
        if before_entry.digest != after_entry.digest:
            changed.append((before_entry, after_entry))
    return changed


def missing_entries(left_entries: Iterable, right_entries: Iterable) -> list[object]:
    left_by_key = index_by_key(left_entries)
    right_by_key = index_by_key(right_entries)
    return [left_by_key[key] for key in sorted(left_by_key.keys() - right_by_key.keys())]


def metadata_lines(
    title: str,
    analysis: StructureAnalysis,
    *,
    repo_url: str,
    mr_url: str,
) -> list[str]:
    return [
        f"# {title}",
        "",
        f"- Repository: `{repo_url}`",
        f"- Merge request: {mr_url}",
        f"- Revision: `{analysis.revision}`",
        f"- Generated at: `{dt.datetime.now(dt.UTC).isoformat()}`",
        f"- Parser: `tree-sitter-go`",
        f"- Unit test files included: `{str(analysis.include_tests).lower()}`",
        f"- Parse errors: `{len(analysis.parse_errors)}`",
        "",
    ]


def write_files_md(path: Path, analysis: StructureAnalysis, *, repo_url: str, mr_url: str) -> None:
    lines = metadata_lines(f"{analysis.name} Files", analysis, repo_url=repo_url, mr_url=mr_url)
    lines.extend(
        [
            f"- Go files: `{len(analysis.files)}`",
            "",
            "## Files",
            "",
            "| path | lines | bytes | hash |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for entry in analysis.files:
        lines.append(
            f"| `{md_escape(entry.path)}` | {entry.line_count} | {entry.byte_count} | `{entry.digest}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def write_functions_md(path: Path, analysis: StructureAnalysis, *, repo_url: str, mr_url: str) -> None:
    lines = metadata_lines(
        f"{analysis.name} Functions",
        analysis,
        repo_url=repo_url,
        mr_url=mr_url,
    )
    lines.extend(
        [
            f"- Functions and methods: `{len(analysis.functions)}`",
            "",
            "## Functions",
            "",
            "| path | name | kind | receiver | lines | hash |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for entry in analysis.functions:
        lines.append(
            f"| `{md_escape(entry.path)}` | `{md_escape(entry.name)}` | {entry.kind} | "
            f"`{md_escape(entry.receiver)}` | {entry.start_line}-{entry.end_line} | `{entry.digest}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def write_structs_md(path: Path, analysis: StructureAnalysis, *, repo_url: str, mr_url: str) -> None:
    lines = metadata_lines(f"{analysis.name} Structs", analysis, repo_url=repo_url, mr_url=mr_url)
    lines.extend(
        [
            f"- Structs: `{len(analysis.structs)}`",
            "",
            "## Structs",
            "",
            "| path | name | fields | field examples | lines | hash |",
            "| --- | --- | ---: | --- | --- | --- |",
        ]
    )
    for entry in analysis.structs:
        field_preview = ", ".join(entry.fields[:12])
        if len(entry.fields) > 12:
            field_preview += f" (+{len(entry.fields) - 12} more)"
        lines.append(
            f"| `{md_escape(entry.path)}` | `{md_escape(entry.name)}` | "
            f"{len(entry.fields)} | {md_escape(field_preview)} | "
            f"{entry.start_line}-{entry.end_line} | `{entry.digest}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def write_file_diff_md(path: Path, before: StructureAnalysis, after: StructureAnalysis, *, mr_url: str) -> None:
    added = missing_entries(after.files, before.files)
    removed = missing_entries(before.files, after.files)
    modified = changed_entries(before.files, after.files)
    lines = [
        "# Diff Files",
        "",
        f"- Merge request: {mr_url}",
        f"- Before revision: `{before.revision}`",
        f"- After revision: `{after.revision}`",
        f"- Generated at: `{dt.datetime.now(dt.UTC).isoformat()}`",
        f"- Unit test files included: `{str(before.include_tests).lower()}`",
        f"- Added files: `{len(added)}`",
        f"- Removed files: `{len(removed)}`",
        f"- Modified files: `{len(modified)}`",
        "",
        "## Added Files",
        "",
        "| path | lines | bytes | hash |",
        "| --- | ---: | ---: | --- |",
    ]
    for entry in added:
        lines.append(f"| `{md_escape(entry.path)}` | {entry.line_count} | {entry.byte_count} | `{entry.digest}` |")
    if not added:
        lines.append("| _none_ | 0 | 0 |  |")

    lines.extend(["", "## Removed Files", "", "| path | lines | bytes | hash |", "| --- | ---: | ---: | --- |"])
    for entry in removed:
        lines.append(f"| `{md_escape(entry.path)}` | {entry.line_count} | {entry.byte_count} | `{entry.digest}` |")
    if not removed:
        lines.append("| _none_ | 0 | 0 |  |")

    lines.extend(
        [
            "",
            "## Modified Files",
            "",
            "| path | before lines | after lines | before hash | after hash |",
            "| --- | ---: | ---: | --- | --- |",
        ]
    )
    for before_entry, after_entry in modified:
        lines.append(
            f"| `{md_escape(before_entry.path)}` | {before_entry.line_count} | "
            f"{after_entry.line_count} | `{before_entry.digest}` | `{after_entry.digest}` |"
        )
    if not modified:
        lines.append("| _none_ | 0 | 0 |  |  |")
    lines.append("")
    path.write_text("\n".join(lines))


def write_function_diff_md(path: Path, before: StructureAnalysis, after: StructureAnalysis, *, mr_url: str) -> None:
    added = missing_entries(after.functions, before.functions)
    removed = missing_entries(before.functions, after.functions)
    modified = changed_entries(before.functions, after.functions)
    lines = [
        "# Diff Functions",
        "",
        f"- Merge request: {mr_url}",
        f"- Before revision: `{before.revision}`",
        f"- After revision: `{after.revision}`",
        f"- Generated at: `{dt.datetime.now(dt.UTC).isoformat()}`",
        f"- Unit test files included: `{str(before.include_tests).lower()}`",
        f"- Added functions: `{len(added)}`",
        f"- Removed functions: `{len(removed)}`",
        f"- Modified functions: `{len(modified)}`",
        "",
        "## Added Functions",
        "",
        "| path | name | kind | lines | hash |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in added:
        lines.append(
            f"| `{md_escape(entry.path)}` | `{md_escape(entry.name)}` | {entry.kind} | "
            f"{entry.start_line}-{entry.end_line} | `{entry.digest}` |"
        )
    if not added:
        lines.append("| _none_ |  |  |  |  |")

    lines.extend(["", "## Removed Functions", "", "| path | name | kind | lines | hash |", "| --- | --- | --- | --- | --- |"])
    for entry in removed:
        lines.append(
            f"| `{md_escape(entry.path)}` | `{md_escape(entry.name)}` | {entry.kind} | "
            f"{entry.start_line}-{entry.end_line} | `{entry.digest}` |"
        )
    if not removed:
        lines.append("| _none_ |  |  |  |  |")

    lines.extend(
        [
            "",
            "## Modified Functions",
            "",
            "| path | name | before lines | after lines | before hash | after hash |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for before_entry, after_entry in modified:
        lines.append(
            f"| `{md_escape(before_entry.path)}` | `{md_escape(before_entry.name)}` | "
            f"{before_entry.start_line}-{before_entry.end_line} | "
            f"{after_entry.start_line}-{after_entry.end_line} | "
            f"`{before_entry.digest}` | `{after_entry.digest}` |"
        )
    if not modified:
        lines.append("| _none_ |  |  |  |  |  |")
    lines.append("")
    path.write_text("\n".join(lines))


def write_struct_diff_md(path: Path, before: StructureAnalysis, after: StructureAnalysis, *, mr_url: str) -> None:
    added = missing_entries(after.structs, before.structs)
    removed = missing_entries(before.structs, after.structs)
    modified = changed_entries(before.structs, after.structs)
    lines = [
        "# Diff Structs",
        "",
        f"- Merge request: {mr_url}",
        f"- Before revision: `{before.revision}`",
        f"- After revision: `{after.revision}`",
        f"- Generated at: `{dt.datetime.now(dt.UTC).isoformat()}`",
        f"- Unit test files included: `{str(before.include_tests).lower()}`",
        f"- Added structs: `{len(added)}`",
        f"- Removed structs: `{len(removed)}`",
        f"- Modified structs: `{len(modified)}`",
        "",
        "## Added Structs",
        "",
        "| path | name | fields | lines | hash |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for entry in added:
        lines.append(
            f"| `{md_escape(entry.path)}` | `{md_escape(entry.name)}` | "
            f"{len(entry.fields)} | {entry.start_line}-{entry.end_line} | `{entry.digest}` |"
        )
    if not added:
        lines.append("| _none_ |  | 0 |  |  |")

    lines.extend(["", "## Removed Structs", "", "| path | name | fields | lines | hash |", "| --- | --- | ---: | --- | --- |"])
    for entry in removed:
        lines.append(
            f"| `{md_escape(entry.path)}` | `{md_escape(entry.name)}` | "
            f"{len(entry.fields)} | {entry.start_line}-{entry.end_line} | `{entry.digest}` |"
        )
    if not removed:
        lines.append("| _none_ |  | 0 |  |  |")

    lines.extend(
        [
            "",
            "## Modified Structs",
            "",
            "| path | name | before fields | after fields | before lines | after lines |",
            "| --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for before_entry, after_entry in modified:
        lines.append(
            f"| `{md_escape(before_entry.path)}` | `{md_escape(before_entry.name)}` | "
            f"{len(before_entry.fields)} | {len(after_entry.fields)} | "
            f"{before_entry.start_line}-{before_entry.end_line} | "
            f"{after_entry.start_line}-{after_entry.end_line} |"
        )
    if not modified:
        lines.append("| _none_ |  | 0 | 0 |  |  |")
    lines.append("")
    path.write_text("\n".join(lines))


def write_all_outputs(
    output_dir: Path,
    before: StructureAnalysis,
    after: StructureAnalysis,
    *,
    repo_url: str,
    mr_url: str,
) -> None:
    write_files_md(output_dir / "before_files.md", before, repo_url=repo_url, mr_url=mr_url)
    write_files_md(output_dir / "after_files.md", after, repo_url=repo_url, mr_url=mr_url)
    write_file_diff_md(output_dir / "diff_files.md", before, after, mr_url=mr_url)
    write_functions_md(output_dir / "before_functions.md", before, repo_url=repo_url, mr_url=mr_url)
    write_functions_md(output_dir / "after_functions.md", after, repo_url=repo_url, mr_url=mr_url)
    write_function_diff_md(output_dir / "diff_functions.md", before, after, mr_url=mr_url)
    write_structs_md(output_dir / "before_structs.md", before, repo_url=repo_url, mr_url=mr_url)
    write_structs_md(output_dir / "after_structs.md", after, repo_url=repo_url, mr_url=mr_url)
    write_struct_diff_md(output_dir / "diff_structs.md", before, after, mr_url=mr_url)


def analyze_from_args(args: argparse.Namespace) -> tuple[StructureAnalysis, StructureAnalysis, str, str]:
    report_mr_url = args.mr_url or ""
    report_repo_url = args.repo_url or ""

    if args.before_dir and args.after_dir:
        if not args.before or not args.after:
            raise SystemExit("--before-dir/--after-dir mode requires --before and --after")
        before = analyze_directory(args.before_dir, args.before, "Before", include_tests=args.include_tests)
        after = analyze_directory(args.after_dir, args.after, "After", include_tests=args.include_tests)
        return before, after, report_repo_url, report_mr_url

    if report_mr_url:
        mr_info: MRInfo = parse_mr_url(report_mr_url, args.mr_ref_name)
        work_dir = args.work_dir or default_work_dir(mr_info)
        prepared = prepare_mr_revisions(mr_info, work_dir, args.before, args.after)
        before = analyze_directory(
            prepared.before_dir,
            prepared.before_revision,
            "Before",
            include_tests=args.include_tests,
        )
        after = analyze_directory(
            prepared.after_dir,
            prepared.after_revision,
            "After",
            include_tests=args.include_tests,
        )
        return before, after, report_repo_url or mr_info.repo_url, mr_info.mr_url

    raise SystemExit("Provide --mr-url or both --before-dir and --after-dir.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch a Go merge request, parse before/after source with Tree-sitter, "
            "and write file/function/struct before, after, and diff markdown files."
        )
    )
    parser.add_argument("--mr-url", help="Merge request URL to analyze")
    parser.add_argument(
        "--mr-ref-name",
        default=DEFAULT_MR_REF_NAME,
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
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    before, after, repo_url, mr_url = analyze_from_args(args)
    write_all_outputs(args.output_dir, before, after, repo_url=repo_url, mr_url=mr_url)

    file_added = len(missing_entries(after.files, before.files))
    file_removed = len(missing_entries(before.files, after.files))
    file_modified = len(changed_entries(before.files, after.files))
    function_added = len(missing_entries(after.functions, before.functions))
    function_removed = len(missing_entries(before.functions, after.functions))
    function_modified = len(changed_entries(before.functions, after.functions))
    struct_added = len(missing_entries(after.structs, before.structs))
    struct_removed = len(missing_entries(before.structs, after.structs))
    struct_modified = len(changed_entries(before.structs, after.structs))

    print(f"before files: {len(before.files)}")
    print(f"after files: {len(after.files)}")
    print(f"file added/removed/modified: {file_added}/{file_removed}/{file_modified}")
    print(f"before functions: {len(before.functions)}")
    print(f"after functions: {len(after.functions)}")
    print(f"function added/removed/modified: {function_added}/{function_removed}/{function_modified}")
    print(f"before structs: {len(before.structs)}")
    print(f"after structs: {len(after.structs)}")
    print(f"struct added/removed/modified: {struct_added}/{struct_removed}/{struct_modified}")


if __name__ == "__main__":
    main()
