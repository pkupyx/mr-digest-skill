#!/usr/bin/env python3
"""Analyze direct function call graph diffs in a Go merge request."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import re
from pathlib import Path
from typing import Iterable

from analyze_mr_structure import function_name, list_go_files, node_text
from analyze_mr_words import (
    DEFAULT_MR_REF_NAME,
    MRInfo,
    default_work_dir,
    iter_nodes,
    make_parser,
    md_escape,
    parse_mr_url,
    prepare_mr_revisions,
)


CALLER_NODE_TYPES = {"function_declaration", "method_declaration"}


@dataclasses.dataclass(frozen=True)
class CallRelation:
    caller_path: str
    caller: str
    callee: str
    resolved_callee: str
    resolution: str
    count: int
    lines: tuple[int, ...]

    @property
    def key(self) -> str:
        return f"{self.caller_path}::{self.caller}->{self.resolved_callee}"


@dataclasses.dataclass
class CallGraphAnalysis:
    name: str
    revision: str
    include_tests: bool
    files_parsed: int
    functions_seen: int
    relations: list[CallRelation]
    parse_errors: list[str]
    internal_functions: int
    ambiguous_method_names: int


@dataclasses.dataclass(frozen=True)
class FunctionDef:
    path: str
    name: str
    kind: str
    receiver: str

    @property
    def method_name(self) -> str:
        if self.kind != "method":
            return self.name
        return self.name.rsplit(".", 1)[-1]

    @property
    def display_name(self) -> str:
        return f"{self.path}::{self.name}"


@dataclasses.dataclass
class FunctionIndex:
    by_top_level: dict[str, list[FunctionDef]]
    by_method_name: dict[str, list[FunctionDef]]
    by_receiver_method: dict[tuple[str, str], list[FunctionDef]]
    total: int

    @property
    def ambiguous_method_names(self) -> int:
        return sum(1 for values in self.by_method_name.values() if len(values) > 1)


def normalize_call_text(value: str) -> str:
    return " ".join(value.split())


def callee_text(source: bytes, call_node) -> str:
    function_node = call_node.child_by_field_name("function")
    if function_node is None and call_node.children:
        function_node = call_node.children[0]
    if function_node is None:
        return "<unknown>"
    value = normalize_call_text(node_text(source, function_node))
    if value.startswith("func(") or value.startswith("func ("):
        return "<func literal>"
    return value


def selector_parts(source: bytes, node) -> list[str]:
    if node.type in {"identifier", "field_identifier", "package_identifier", "type_identifier"}:
        return [node_text(source, node)]
    if node.type == "selector_expression":
        parts: list[str] = []
        for child in node.children:
            if child.type == ".":
                continue
            parts.extend(selector_parts(source, child))
        return parts
    return []


def normalize_receiver_type(value: str) -> str:
    return value.lstrip("*")


def function_receiver_info(source: bytes, method_node) -> tuple[str, str]:
    if method_node.type != "method_declaration":
        return "", ""
    receiver = method_node.child_by_field_name("receiver")
    if receiver is None:
        return "", ""
    receiver_var = ""
    for node in iter_nodes(receiver):
        if node.type == "identifier":
            receiver_var = node_text(source, node)
            break
    _, _, receiver_type = function_name(source, method_node)
    return receiver_var, normalize_receiver_type(receiver_type)


def callee_parts(source: bytes, call_node) -> list[str]:
    function_node = call_node.child_by_field_name("function")
    if function_node is None and call_node.children:
        function_node = call_node.children[0]
    if function_node is None:
        return []
    return selector_parts(source, function_node)


def read_module_path(root: Path) -> str:
    go_mod = root / "go.mod"
    if not go_mod.exists():
        return ""
    for line in go_mod.read_text(errors="replace").splitlines():
        match = re.match(r"^\s*module\s+(\S+)\s*$", line)
        if match:
            return match.group(1)
    return ""


def import_alias_from_path(import_path: str) -> str:
    return import_path.rstrip("/").rsplit("/", 1)[-1].replace("-", "_")


def import_aliases(source: bytes) -> dict[str, str]:
    aliases: dict[str, str] = {}
    text = source.decode(errors="replace")
    for match in re.finditer(
        r'(?m)^\s*(?:(?P<alias>[A-Za-z_][A-Za-z0-9_]*|\.)\s+)?'
        r'"(?P<path>[^"]+)"',
        text,
    ):
        alias = match.group("alias")
        import_path = match.group("path")
        if alias in {".", "_"}:
            continue
        aliases[alias or import_alias_from_path(import_path)] = import_path
    return aliases


def collect_function_index(root: Path, *, include_tests: bool) -> FunctionIndex:
    parser = make_parser()
    by_top_level: dict[str, list[FunctionDef]] = collections.defaultdict(list)
    by_method_name: dict[str, list[FunctionDef]] = collections.defaultdict(list)
    by_receiver_method: dict[tuple[str, str], list[FunctionDef]] = collections.defaultdict(list)

    for file_path in list_go_files(root, include_tests=include_tests):
        relative_path = file_path.relative_to(root).as_posix()
        source = file_path.read_bytes()
        tree = parser.parse(source)
        for node in iter_nodes(tree.root_node):
            if node.type not in CALLER_NODE_TYPES:
                continue
            name_value, kind, receiver = function_name(source, node)
            function_def = FunctionDef(
                path=relative_path,
                name=name_value,
                kind=kind,
                receiver=receiver,
            )
            if kind == "function":
                by_top_level[name_value].append(function_def)
            else:
                method_name = function_def.method_name
                by_method_name[method_name].append(function_def)
                by_receiver_method[
                    (normalize_receiver_type(receiver), method_name)
                ].append(function_def)

    total = sum(len(values) for values in by_top_level.values()) + sum(
        len(values) for values in by_method_name.values()
    )
    return FunctionIndex(
        by_top_level=dict(by_top_level),
        by_method_name=dict(by_method_name),
        by_receiver_method=dict(by_receiver_method),
        total=total,
    )


def package_dir_for_import(import_path: str, module_path: str) -> str:
    if not module_path or not import_path.startswith(module_path):
        return ""
    return import_path[len(module_path) :].lstrip("/")


def same_package_candidates(
    candidates: list[FunctionDef],
    caller_path: str,
) -> list[FunctionDef]:
    caller_dir = str(Path(caller_path).parent)
    return [
        candidate
        for candidate in candidates
        if str(Path(candidate.path).parent) == caller_dir
    ]


def imported_package_candidates(
    candidates: list[FunctionDef],
    package_dir: str,
) -> list[FunctionDef]:
    if not package_dir:
        return []
    prefix = f"{package_dir.rstrip('/')}/"
    return [
        candidate
        for candidate in candidates
        if candidate.path.startswith(prefix)
    ]


def one_resolved_candidate(
    candidates: list[FunctionDef],
    resolution: str,
    ambiguous_resolution: str,
) -> tuple[str, str] | None:
    if len(candidates) == 1:
        return candidates[0].display_name, resolution
    if len(candidates) > 1:
        return (
            f"{candidates[0].name} ({len(candidates)} internal candidates)",
            ambiguous_resolution,
        )
    return None


def resolved_internal_callee(
    callee: str,
    parts: list[str],
    imports: dict[str, str],
    module_path: str,
    function_index: FunctionIndex,
    *,
    caller_path: str,
    receiver_var: str,
    receiver_type: str,
) -> tuple[str, str] | None:
    if not parts:
        return None

    if len(parts) == 1:
        defs = same_package_candidates(
            function_index.by_top_level.get(parts[0], []),
            caller_path,
        )
        return one_resolved_candidate(
            defs,
            "same-package-function",
            "ambiguous-same-package-function",
        )

    root_name = parts[0]
    final_name = parts[-1]
    import_path = imports.get(root_name)
    if import_path is not None:
        if len(parts) != 2:
            return None
        package_dir = package_dir_for_import(import_path, module_path)
        if package_dir:
            defs = imported_package_candidates(
                function_index.by_top_level.get(final_name, []),
                package_dir,
            )
            return one_resolved_candidate(
                defs,
                "internal-package-function",
                "ambiguous-internal-package-function",
            )
        return None

    if len(parts) == 2 and receiver_var and receiver_type and root_name == receiver_var:
        method_defs = function_index.by_receiver_method.get(
            (receiver_type, final_name),
            [],
        )
        return one_resolved_candidate(
            method_defs,
            "receiver-method",
            "ambiguous-receiver-method",
        )

    return None


def is_nested_function(node, root_function) -> bool:
    parent = node.parent
    while parent is not None and parent != root_function:
        if parent.type in CALLER_NODE_TYPES or parent.type == "func_literal":
            return True
        parent = parent.parent
    return False


def call_lines(values: Iterable[int], limit: int = 24) -> str:
    ordered = sorted(values)
    shown = ordered[:limit]
    suffix = f" (+{len(ordered) - limit} more)" if len(ordered) > limit else ""
    return ", ".join(str(value) for value in shown) + suffix


def analyze_directory(root: Path, revision: str, name: str, *, include_tests: bool) -> CallGraphAnalysis:
    parser = make_parser()
    parse_errors: list[str] = []
    relation_lines: dict[tuple[str, str, str, str, str], list[int]] = collections.defaultdict(list)
    files = list_go_files(root, include_tests=include_tests)
    function_index = collect_function_index(root, include_tests=include_tests)
    module_path = read_module_path(root)
    functions_seen = 0

    for file_path in files:
        relative_path = file_path.relative_to(root).as_posix()
        source = file_path.read_bytes()
        imports = import_aliases(source)
        tree = parser.parse(source)
        if tree.root_node.has_error:
            parse_errors.append(relative_path)

        for function_node in iter_nodes(tree.root_node):
            if function_node.type not in CALLER_NODE_TYPES:
                continue
            functions_seen += 1
            caller, _, _ = function_name(source, function_node)
            receiver_var, receiver_type = function_receiver_info(source, function_node)
            for node in iter_nodes(function_node):
                if node.type != "call_expression":
                    continue
                if is_nested_function(node, function_node):
                    continue
                callee = callee_text(source, node)
                resolved = resolved_internal_callee(
                    callee,
                    callee_parts(source, node),
                    imports,
                    module_path,
                    function_index,
                    caller_path=relative_path,
                    receiver_var=receiver_var,
                    receiver_type=receiver_type,
                )
                if resolved is None:
                    continue
                resolved_callee, resolution = resolved
                line = node.start_point[0] + 1
                relation_lines[
                    (relative_path, caller, callee, resolved_callee, resolution)
                ].append(line)

    relations = [
        CallRelation(
            caller_path=caller_path,
            caller=caller,
            callee=callee,
            resolved_callee=resolved_callee,
            resolution=resolution,
            count=len(lines),
            lines=tuple(sorted(lines)),
        )
        for (caller_path, caller, callee, resolved_callee, resolution), lines in relation_lines.items()
    ]
    return CallGraphAnalysis(
        name=name,
        revision=revision,
        include_tests=include_tests,
        files_parsed=len(files),
        functions_seen=functions_seen,
        relations=sorted(relations, key=lambda item: (item.caller_path, item.caller, item.callee)),
        parse_errors=parse_errors,
        internal_functions=function_index.total,
        ambiguous_method_names=function_index.ambiguous_method_names,
    )


def index_by_key(relations: Iterable[CallRelation]) -> dict[str, CallRelation]:
    return {relation.key: relation for relation in relations}


def missing_relations(left: Iterable[CallRelation], right: Iterable[CallRelation]) -> list[CallRelation]:
    left_by_key = index_by_key(left)
    right_by_key = index_by_key(right)
    return [left_by_key[key] for key in sorted(left_by_key.keys() - right_by_key.keys())]


def changed_relations(before: Iterable[CallRelation], after: Iterable[CallRelation]) -> list[tuple[CallRelation, CallRelation]]:
    before_by_key = index_by_key(before)
    after_by_key = index_by_key(after)
    changed: list[tuple[CallRelation, CallRelation]] = []
    for key in sorted(before_by_key.keys() & after_by_key.keys()):
        before_relation = before_by_key[key]
        after_relation = after_by_key[key]
        if before_relation.count != after_relation.count:
            changed.append((before_relation, after_relation))
    return changed


def group_by_caller(relations: Iterable[CallRelation]) -> dict[tuple[str, str], list[CallRelation]]:
    grouped: dict[tuple[str, str], list[CallRelation]] = collections.defaultdict(list)
    for relation in relations:
        grouped[(relation.caller_path, relation.caller)].append(relation)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def metadata_lines(title: str, analysis: CallGraphAnalysis, *, repo_url: str, mr_url: str) -> list[str]:
    return [
        f"# {title}",
        "",
        f"- Repository: `{repo_url}`",
        f"- Merge request: {mr_url}",
        f"- Revision: `{analysis.revision}`",
        f"- Generated at: `{dt.datetime.now(dt.UTC).isoformat()}`",
        f"- Parser: `tree-sitter-go`",
        f"- Unit test files included: `{str(analysis.include_tests).lower()}`",
        f"- Go files parsed: `{analysis.files_parsed}`",
        f"- Functions seen: `{analysis.functions_seen}`",
        f"- Internal functions indexed: `{analysis.internal_functions}`",
        f"- Ambiguous method names skipped: `{analysis.ambiguous_method_names}`",
        f"- Internal direct call relations: `{len(analysis.relations)}`",
        f"- Parse errors: `{len(analysis.parse_errors)}`",
        "",
    ]


def write_call_graph_md(path: Path, analysis: CallGraphAnalysis, *, repo_url: str, mr_url: str) -> None:
    lines = metadata_lines(f"{analysis.name} Call Graph", analysis, repo_url=repo_url, mr_url=mr_url)
    lines.extend(
        [
            "## Direct Call Relations",
            "",
            "| caller path | caller | source callee | resolved internal callee | resolution | count | lines |",
            "| --- | --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for relation in analysis.relations:
        lines.append(
            f"| `{md_escape(relation.caller_path)}` | `{md_escape(relation.caller)}` | "
            f"`{md_escape(relation.callee)}` | `{md_escape(relation.resolved_callee)}` | "
            f"{relation.resolution} | {relation.count} | {call_lines(relation.lines)} |"
        )
    if not analysis.relations:
        lines.append("| _none_ |  |  |  |  | 0 |  |")

    lines.extend(["", "## Call Chains By Caller", ""])
    for (caller_path, caller), relations in group_by_caller(analysis.relations).items():
        callees = ", ".join(
            f"`{md_escape(relation.callee)}`"
            f" -> `{md_escape(relation.resolved_callee)}`"
            + (f" x{relation.count}" if relation.count > 1 else "")
            for relation in sorted(relations, key=lambda item: item.callee)
        )
        lines.append(f"- `{md_escape(caller_path)}::{md_escape(caller)}` -> {callees}")
    lines.append("")
    path.write_text("\n".join(lines))


def write_call_graph_diff_md(path: Path, before: CallGraphAnalysis, after: CallGraphAnalysis, *, mr_url: str) -> None:
    added = missing_relations(after.relations, before.relations)
    removed = missing_relations(before.relations, after.relations)
    changed = changed_relations(before.relations, after.relations)
    lines = [
        "# Diff Call Graph",
        "",
        f"- Merge request: {mr_url}",
        f"- Before revision: `{before.revision}`",
        f"- After revision: `{after.revision}`",
        f"- Generated at: `{dt.datetime.now(dt.UTC).isoformat()}`",
        f"- Unit test files included: `{str(before.include_tests).lower()}`",
        f"- Added internal call relations: `{len(added)}`",
        f"- Removed internal call relations: `{len(removed)}`",
        f"- Count-changed internal call relations: `{len(changed)}`",
        "",
        "## Added Call Relations",
        "",
        "| caller path | caller | source callee | resolved internal callee | resolution | count | lines |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for relation in added:
        lines.append(
            f"| `{md_escape(relation.caller_path)}` | `{md_escape(relation.caller)}` | "
            f"`{md_escape(relation.callee)}` | `{md_escape(relation.resolved_callee)}` | "
            f"{relation.resolution} | {relation.count} | {call_lines(relation.lines)} |"
        )
    if not added:
        lines.append("| _none_ |  |  |  |  | 0 |  |")

    lines.extend(
        [
            "",
            "## Removed Call Relations",
            "",
            "| caller path | caller | source callee | resolved internal callee | resolution | count | lines |",
            "| --- | --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for relation in removed:
        lines.append(
            f"| `{md_escape(relation.caller_path)}` | `{md_escape(relation.caller)}` | "
            f"`{md_escape(relation.callee)}` | `{md_escape(relation.resolved_callee)}` | "
            f"{relation.resolution} | {relation.count} | {call_lines(relation.lines)} |"
        )
    if not removed:
        lines.append("| _none_ |  |  |  |  | 0 |  |")

    lines.extend(
        [
            "",
            "## Count-Changed Call Relations",
            "",
            "| caller path | caller | source callee | resolved internal callee | before | after | delta | before lines | after lines |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for before_relation, after_relation in changed:
        delta = after_relation.count - before_relation.count
        lines.append(
            f"| `{md_escape(before_relation.caller_path)}` | `{md_escape(before_relation.caller)}` | "
            f"`{md_escape(before_relation.callee)}` | `{md_escape(before_relation.resolved_callee)}` | "
            f"{before_relation.count} | {after_relation.count} | {delta:+d} | "
            f"{call_lines(before_relation.lines)} | "
            f"{call_lines(after_relation.lines)} |"
        )
    if not changed:
        lines.append("| _none_ |  |  |  | 0 | 0 | 0 |  |  |")

    lines.extend(["", "## Added Call Chains By Caller", ""])
    for (caller_path, caller), relations in group_by_caller(added).items():
        callees = ", ".join(
            f"`{md_escape(relation.callee)}`"
            f" -> `{md_escape(relation.resolved_callee)}`"
            + (f" x{relation.count}" if relation.count > 1 else "")
            for relation in sorted(relations, key=lambda item: item.callee)
        )
        lines.append(f"- `{md_escape(caller_path)}::{md_escape(caller)}` -> {callees}")
    if not added:
        lines.append("- _none_")

    lines.extend(["", "## Removed Call Chains By Caller", ""])
    for (caller_path, caller), relations in group_by_caller(removed).items():
        callees = ", ".join(
            f"`{md_escape(relation.callee)}`"
            f" -> `{md_escape(relation.resolved_callee)}`"
            + (f" x{relation.count}" if relation.count > 1 else "")
            for relation in sorted(relations, key=lambda item: item.callee)
        )
        lines.append(f"- `{md_escape(caller_path)}::{md_escape(caller)}` -> {callees}")
    if not removed:
        lines.append("- _none_")

    lines.append("")
    path.write_text("\n".join(lines))


def analyze_from_args(args: argparse.Namespace) -> tuple[CallGraphAnalysis, CallGraphAnalysis, str, str]:
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
            "and write direct function call graph before, after, and diff markdown files."
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
    write_call_graph_md(args.output_dir / "before_call_graph.md", before, repo_url=repo_url, mr_url=mr_url)
    write_call_graph_md(args.output_dir / "after_call_graph.md", after, repo_url=repo_url, mr_url=mr_url)
    write_call_graph_diff_md(args.output_dir / "diff_call_graph.md", before, after, mr_url=mr_url)

    added = len(missing_relations(after.relations, before.relations))
    removed = len(missing_relations(before.relations, after.relations))
    changed = len(changed_relations(before.relations, after.relations))
    print(f"before files: {before.files_parsed}")
    print(f"after files: {after.files_parsed}")
    print(f"before functions: {before.functions_seen}")
    print(f"after functions: {after.functions_seen}")
    print(f"before internal call relations: {len(before.relations)}")
    print(f"after internal call relations: {len(after.relations)}")
    print(f"internal call relations added/removed/changed: {added}/{removed}/{changed}")


if __name__ == "__main__":
    main()
