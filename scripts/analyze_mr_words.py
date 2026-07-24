#!/usr/bin/env python3
"""Analyze identifier words in a Go merge request with Tree-sitter.

This script is intended to be called by a Codex skill. Given a merge request URL,
it fetches the MR ref, checks out before/after worktrees, parses Go source files
with Tree-sitter, writes before_words.md and after_words.md, then writes
diff_words.md.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import re
import subprocess
from pathlib import Path
from typing import Iterable

try:
    from tree_sitter import Language, Parser
    import tree_sitter_go
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing Tree-sitter Python packages. Install them with: "
        "python3 -m pip install tree_sitter tree_sitter_go"
    ) from exc


IDENTIFIER_NODE_TYPES = {
    "identifier",
    "type_identifier",
    "field_identifier",
    "package_identifier",
    "label_name",
}

DEFAULT_MR_REF_NAME = "tmp-squash"
DEFAULT_WORK_ROOT = Path("/private/tmp/mr-digest-skill")


@dataclasses.dataclass
class Analysis:
    name: str
    revision: str
    files_seen: int
    files_parsed: int
    files_with_errors: list[str]
    symbol_occurrences: int
    unique_symbols: set[str]
    word_counts: collections.Counter[str]
    word_symbols: dict[str, set[str]]
    include_tests: bool


@dataclasses.dataclass
class MRInfo:
    mr_url: str
    repo_url: str
    repo_git_url: str
    mr_iid: str
    remote_ref: str
    local_ref: str


@dataclasses.dataclass
class PreparedRevisions:
    repo: Path
    before_dir: Path
    after_dir: Path
    before_revision: str
    after_revision: str


def run_git(repo: Path, args: list[str], *, input_text: str | None = None) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        input=input_text.encode() if input_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {stderr}")
    return result.stdout


def run_git_global(args: list[str]) -> bytes:
    result = subprocess.run(
        ["git", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {stderr}")
    return result.stdout


def short_revision(revision: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "-", revision)[:12]


def parse_mr_url(mr_url: str, ref_name: str) -> MRInfo:
    match = re.match(r"(?P<repo>https?://.+?)/merge_requests/(?P<iid>\d+)(?:/.*)?$", mr_url)
    if not match:
        raise ValueError(
            "MR URL must look like https://host/group/project/merge_requests/123"
        )
    repo_url = match.group("repo").removesuffix(".git")
    mr_iid = match.group("iid")
    remote_ref = f"refs/merge-requests/{mr_iid}/{ref_name}"
    local_ref = f"refs/remotes/origin/mr-{mr_iid}-{ref_name}"
    return MRInfo(
        mr_url=mr_url,
        repo_url=repo_url,
        repo_git_url=f"{repo_url}.git",
        mr_iid=mr_iid,
        remote_ref=remote_ref,
        local_ref=local_ref,
    )


def default_work_dir(mr_info: MRInfo) -> Path:
    repo_slug = re.sub(r"[^0-9A-Za-z._-]+", "-", mr_info.repo_url).strip("-")
    return DEFAULT_WORK_ROOT / f"{repo_slug}-mr-{mr_info.mr_iid}"


def ensure_repo(repo_dir: Path, repo_url: str) -> None:
    if not (repo_dir / ".git").exists():
        repo_dir.mkdir(parents=True, exist_ok=True)
        run_git_global(["init", str(repo_dir)])
    existing_remote = subprocess.run(
        ["git", "-C", str(repo_dir), "remote", "get-url", "origin"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if existing_remote.returncode == 0:
        current = existing_remote.stdout.decode(errors="replace").strip()
        if current != repo_url:
            run_git(repo_dir, ["remote", "set-url", "origin", repo_url])
    else:
        run_git(repo_dir, ["remote", "add", "origin", repo_url])


def first_parent(repo: Path, revision: str) -> str:
    parents = (
        run_git(repo, ["rev-list", "--parents", "-n", "1", revision])
        .decode(errors="replace")
        .strip()
        .split()
    )
    if len(parents) < 2:
        raise RuntimeError(f"Revision {revision} does not have a parent commit")
    return parents[1]


def ensure_worktree(repo: Path, path: Path, revision: str) -> None:
    if path.exists():
        if not (path / ".git").exists():
            raise RuntimeError(f"Worktree path exists but is not a git worktree: {path}")
        current = (
            run_git(path, ["rev-parse", "HEAD"]).decode(errors="replace").strip()
        )
        if current != revision:
            run_git(path, ["checkout", "--detach", revision])
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    run_git(repo, ["worktree", "add", "--detach", str(path), revision])


def prepare_mr_revisions(
    mr_info: MRInfo,
    work_dir: Path,
    before_revision: str | None,
    after_revision: str | None,
) -> PreparedRevisions:
    repo_dir = work_dir / "repo"
    ensure_repo(repo_dir, mr_info.repo_git_url)
    run_git(
        repo_dir,
        [
            "fetch",
            "--filter=blob:none",
            "origin",
            f"{mr_info.remote_ref}:{mr_info.local_ref}",
        ],
    )

    after = after_revision or (
        run_git(repo_dir, ["rev-parse", mr_info.local_ref])
        .decode(errors="replace")
        .strip()
    )
    before = before_revision or first_parent(repo_dir, after)
    before_dir = work_dir / f"before-{short_revision(before)}"
    after_dir = work_dir / f"after-{short_revision(after)}"
    ensure_worktree(repo_dir, before_dir, before)
    ensure_worktree(repo_dir, after_dir, after)
    return PreparedRevisions(
        repo=repo_dir,
        before_dir=before_dir,
        after_dir=after_dir,
        before_revision=before,
        after_revision=after,
    )


def is_test_file(path: str | Path) -> bool:
    return Path(path).name.endswith("_test.go")


def list_go_files(repo: Path, revision: str, *, include_tests: bool) -> list[str]:
    out = run_git(repo, ["ls-tree", "-r", "--name-only", revision])
    return sorted(
        line.decode(errors="replace")
        for line in out.splitlines()
        if line.endswith(b".go")
        and (include_tests or not is_test_file(line.decode(errors="replace")))
    )


def read_file_at_revision(repo: Path, revision: str, path: str) -> bytes:
    return run_git(repo, ["show", f"{revision}:{path}"])


def split_symbol(symbol: str) -> list[str]:
    words: list[str] = []
    for chunk in re.split(r"[^0-9A-Za-z]+", symbol):
        if not chunk:
            continue
        spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", chunk)
        spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", spaced)
        for word in spaced.split():
            if re.search(r"[A-Za-z]", word):
                words.append(word.lower())
    return words


def iter_nodes(root) -> Iterable:
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


def make_parser() -> Parser:
    parser = Parser()
    parser.language = Language(tree_sitter_go.language())
    return parser


def analyze_revision(
    repo: Path,
    revision: str,
    name: str,
    *,
    include_tests: bool,
) -> Analysis:
    parser = make_parser()
    files = list_go_files(repo, revision, include_tests=include_tests)
    word_counts: collections.Counter[str] = collections.Counter()
    word_symbols: dict[str, set[str]] = collections.defaultdict(set)
    unique_symbols: set[str] = set()
    files_with_errors: list[str] = []
    symbol_occurrences = 0

    for path in files:
        source = read_file_at_revision(repo, revision, path)
        tree = parser.parse(source)
        if tree.root_node.has_error:
            files_with_errors.append(path)
        for node in iter_nodes(tree.root_node):
            if node.type not in IDENTIFIER_NODE_TYPES:
                continue
            symbol = source[node.start_byte : node.end_byte].decode(errors="replace")
            if not symbol or symbol == "_":
                continue
            words = split_symbol(symbol)
            if not words:
                continue
            symbol_occurrences += 1
            unique_symbols.add(symbol)
            for word in words:
                word_counts[word] += 1
                word_symbols[word].add(symbol)

    return Analysis(
        name=name,
        revision=revision,
        files_seen=len(files),
        files_parsed=len(files),
        files_with_errors=files_with_errors,
        symbol_occurrences=symbol_occurrences,
        unique_symbols=unique_symbols,
        word_counts=word_counts,
        word_symbols=word_symbols,
        include_tests=include_tests,
    )


def list_go_files_in_dir(root: Path, *, include_tests: bool) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.go")
        if ".git" not in path.relative_to(root).parts
        and (include_tests or not is_test_file(path))
    )


def analyze_directory(
    root: Path,
    revision: str,
    name: str,
    *,
    include_tests: bool,
) -> Analysis:
    parser = make_parser()
    files = list_go_files_in_dir(root, include_tests=include_tests)
    word_counts: collections.Counter[str] = collections.Counter()
    word_symbols: dict[str, set[str]] = collections.defaultdict(set)
    unique_symbols: set[str] = set()
    files_with_errors: list[str] = []
    symbol_occurrences = 0

    for file_path in files:
        relative_path = file_path.relative_to(root).as_posix()
        source = file_path.read_bytes()
        tree = parser.parse(source)
        if tree.root_node.has_error:
            files_with_errors.append(relative_path)
        for node in iter_nodes(tree.root_node):
            if node.type not in IDENTIFIER_NODE_TYPES:
                continue
            symbol = source[node.start_byte : node.end_byte].decode(errors="replace")
            if not symbol or symbol == "_":
                continue
            words = split_symbol(symbol)
            if not words:
                continue
            symbol_occurrences += 1
            unique_symbols.add(symbol)
            for word in words:
                word_counts[word] += 1
                word_symbols[word].add(symbol)

    return Analysis(
        name=name,
        revision=revision,
        files_seen=len(files),
        files_parsed=len(files),
        files_with_errors=files_with_errors,
        symbol_occurrences=symbol_occurrences,
        unique_symbols=unique_symbols,
        word_counts=word_counts,
        word_symbols=word_symbols,
        include_tests=include_tests,
    )


def md_escape(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", " ")


def symbol_examples(symbols: Iterable[str], limit: int = 12) -> str:
    ordered = sorted(symbols, key=lambda item: (item.lower(), item))
    shown = ordered[:limit]
    suffix = f" (+{len(ordered) - limit} more)" if len(ordered) > limit else ""
    return ", ".join(md_escape(item) for item in shown) + suffix


def write_words_md(path: Path, analysis: Analysis, *, repo_url: str, mr_url: str) -> None:
    lines = [
        f"# {analysis.name} Words",
        "",
        f"- Repository: `{repo_url}`",
        f"- Merge request: {mr_url}",
        f"- Revision: `{analysis.revision}`",
        f"- Generated at: `{dt.datetime.now(dt.UTC).isoformat()}`",
        f"- Parser: `tree-sitter-go`",
        f"- Unit test files included: `{str(analysis.include_tests).lower()}`",
        f"- Go files parsed: `{analysis.files_parsed}`",
        f"- Files with parse errors: `{len(analysis.files_with_errors)}`",
        f"- Identifier node occurrences: `{analysis.symbol_occurrences}`",
        f"- Unique symbols: `{len(analysis.unique_symbols)}`",
        f"- Unique words: `{len(analysis.word_counts)}`",
        "",
    ]
    if analysis.files_with_errors:
        lines.extend(["## Files With Parse Errors", ""])
        for file_path in analysis.files_with_errors:
            lines.append(f"- `{file_path}`")
        lines.append("")

    lines.extend(
        [
            "## Words",
            "",
            "| word | occurrences | distinct symbols | symbol examples |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for word in sorted(analysis.word_counts):
        symbols = analysis.word_symbols[word]
        lines.append(
            f"| `{md_escape(word)}` | {analysis.word_counts[word]} | "
            f"{len(symbols)} | {symbol_examples(symbols)} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def write_diff_md(path: Path, before: Analysis, after: Analysis, *, mr_url: str) -> None:
    before_words = set(before.word_counts)
    after_words = set(after.word_counts)
    added = sorted(after_words - before_words)
    removed = sorted(before_words - after_words)
    changed = sorted(
        word
        for word in before_words & after_words
        if before.word_counts[word] != after.word_counts[word]
    )

    lines = [
        "# Diff Words",
        "",
        f"- Merge request: {mr_url}",
        f"- Before revision: `{before.revision}`",
        f"- After revision: `{after.revision}`",
        f"- Generated at: `{dt.datetime.now(dt.UTC).isoformat()}`",
        f"- Added words: `{len(added)}`",
        f"- Removed words: `{len(removed)}`",
        f"- Count-changed words: `{len(changed)}`",
        f"- Unit test files included: `{str(before.include_tests).lower()}`",
        "",
        "## Added Words",
        "",
        "| word | after occurrences | after distinct symbols | after symbol examples |",
        "| --- | ---: | ---: | --- |",
    ]
    for word in added:
        lines.append(
            f"| `{md_escape(word)}` | {after.word_counts[word]} | "
            f"{len(after.word_symbols[word])} | {symbol_examples(after.word_symbols[word])} |"
        )
    if not added:
        lines.append("| _none_ | 0 | 0 |  |")

    lines.extend(
        [
            "",
            "## Removed Words",
            "",
            "| word | before occurrences | before distinct symbols | before symbol examples |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for word in removed:
        lines.append(
            f"| `{md_escape(word)}` | {before.word_counts[word]} | "
            f"{len(before.word_symbols[word])} | {symbol_examples(before.word_symbols[word])} |"
        )
    if not removed:
        lines.append("| _none_ | 0 | 0 |  |")

    lines.extend(
        [
            "",
            "## Changed Word Counts",
            "",
            "| word | before | after | delta | before symbols | after symbols |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for word in changed:
        before_count = before.word_counts[word]
        after_count = after.word_counts[word]
        delta = after_count - before_count
        lines.append(
            f"| `{md_escape(word)}` | {before_count} | {after_count} | {delta:+d} | "
            f"{symbol_examples(before.word_symbols[word], limit=8)} | "
            f"{symbol_examples(after.word_symbols[word], limit=8)} |"
        )
    if not changed:
        lines.append("| _none_ | 0 | 0 | 0 |  |  |")

    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch a Go merge request, parse before/after source with Tree-sitter, "
            "and write before_words.md, after_words.md, and diff_words.md."
        )
    )
    parser.add_argument("--mr-url", help="Merge request URL to analyze")
    parser.add_argument(
        "--mr-ref-name",
        default=DEFAULT_MR_REF_NAME,
        help="MR ref suffix to fetch, default: tmp-squash",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Temporary workspace for fetches and worktrees",
    )
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--before")
    parser.add_argument("--after")
    parser.add_argument("--before-dir", type=Path)
    parser.add_argument("--after-dir", type=Path)
    parser.add_argument("--output-dir", default=Path("."), type=Path)
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include Go unit test files. By default, *_test.go files are skipped.",
    )
    parser.add_argument(
        "--repo-url",
        help="Repository URL for report metadata when not using --mr-url",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_mr_url = args.mr_url or ""
    report_repo_url = args.repo_url or ""

    if args.before_dir and args.after_dir:
        if not args.before or not args.after:
            raise SystemExit("--before-dir/--after-dir mode requires --before and --after")
        before = analyze_directory(
            args.before_dir,
            args.before,
            "Before",
            include_tests=args.include_tests,
        )
        after = analyze_directory(
            args.after_dir,
            args.after,
            "After",
            include_tests=args.include_tests,
        )
    elif args.repo:
        if not args.before or not args.after:
            raise SystemExit("--repo mode requires --before and --after")
        before = analyze_revision(
            args.repo,
            args.before,
            "Before",
            include_tests=args.include_tests,
        )
        after = analyze_revision(
            args.repo,
            args.after,
            "After",
            include_tests=args.include_tests,
        )
    elif report_mr_url:
        mr_info = parse_mr_url(report_mr_url, args.mr_ref_name)
        work_dir = args.work_dir or default_work_dir(mr_info)
        prepared = prepare_mr_revisions(mr_info, work_dir, args.before, args.after)
        report_repo_url = report_repo_url or mr_info.repo_url
        report_mr_url = mr_info.mr_url
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
    else:
        raise SystemExit(
            "Provide --mr-url, --repo, or both --before-dir and --after-dir."
        )

    write_words_md(
        args.output_dir / "before_words.md",
        before,
        repo_url=report_repo_url,
        mr_url=report_mr_url,
    )
    write_words_md(
        args.output_dir / "after_words.md",
        after,
        repo_url=report_repo_url,
        mr_url=report_mr_url,
    )
    write_diff_md(
        args.output_dir / "diff_words.md",
        before,
        after,
        mr_url=report_mr_url,
    )

    print(f"before files: {before.files_parsed}")
    print(f"after files: {after.files_parsed}")
    print(f"before unique words: {len(before.word_counts)}")
    print(f"after unique words: {len(after.word_counts)}")
    print(f"added words: {len(set(after.word_counts) - set(before.word_counts))}")
    print(f"removed words: {len(set(before.word_counts) - set(after.word_counts))}")
    print(
        "changed words: "
        f"{sum(1 for word in set(before.word_counts) & set(after.word_counts) if before.word_counts[word] != after.word_counts[word])}"
    )


if __name__ == "__main__":
    main()
