#!/usr/bin/env python3
"""Remove build-host paths from Maturin CycloneDX SBOMs.

The sanitizer accepts individual JSON files, directories containing SBOM JSON,
or wheels. Wheel updates include a regenerated ``.dist-info/RECORD``.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote
import zipfile


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR.parent / "rust" / "Cargo.toml"
SBOM_MEMBER = re.compile(r"(?:^|/)[^/]+\.dist-info/sboms/.*\.json$")
SENSITIVE_PATHS = (
    re.compile(r"(?i)(?:path\+)?file:///(?:[a-z]:/)?(?:users|home|homes|root)/"),
    re.compile(r"(?i)(?:^|[\s\"'])(?:[a-z]:[\\/])users[\\/]"),
    re.compile(r"^/(?:Users|home|homes|root)/"),
    re.compile(r"(?i)(?:path\+)?file:///(?:private/(?:tmp|var)|tmp|var/folders)/"),
    re.compile(r"^/(?:private/(?:tmp|var)|tmp|var/folders)/"),
)


class SanitizationError(RuntimeError):
    """Raised when an SBOM cannot be sanitized safely."""


def _path_variants(source: str) -> List[str]:
    raw = source.rstrip("/\\")
    variants = {
        raw,
        os.path.abspath(raw).rstrip("/\\"),
        os.path.realpath(raw).rstrip("/\\"),
    }
    variants.update(value.replace("\\", "/") for value in list(variants))
    variants.update(quote(value, safe="/:") for value in list(variants))
    return sorted((value for value in variants if value), key=len, reverse=True)


def build_replacements(mappings: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Expand native and URI forms, preferring the longest source prefixes."""
    replacements: Dict[str, str] = {}
    for source, destination in mappings:
        if not source:
            raise SanitizationError("mapping source must not be empty")
        source_path = Path(source)
        if source_path.anchor and source_path == Path(source_path.anchor):
            raise SanitizationError("refusing to map a filesystem root")
        destination = destination.rstrip("/\\")
        for variant in _path_variants(source):
            replacements[variant] = destination.replace("\\", "/")
    return sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True)


def default_mappings(manifest_path: Path) -> List[Tuple[str, str]]:
    """Derive deterministic virtual roots from locked Cargo metadata."""
    command = [
        "cargo",
        "metadata",
        "--locked",
        "--format-version",
        "1",
        "--manifest-path",
        str(manifest_path.resolve()),
    ]
    result = subprocess.run(
        command,
        cwd=manifest_path.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    local_roots = [
        str(Path(package["manifest_path"]).resolve().parent)
        for package in metadata["packages"]
        if package.get("source") is None
    ]
    if not local_roots:
        raise SanitizationError("Cargo metadata did not report any local packages")

    workspace_root = os.path.commonpath(local_roots)
    workspace_path = Path(workspace_root)
    if workspace_path == Path(workspace_path.anchor):
        raise SanitizationError("local Cargo packages do not share a safe workspace root")
    mappings = [(workspace_root, "/workspace")]
    home = str(Path.home())
    mappings.extend(
        [
            (home, "/build/user"),
            (os.environ.get("CARGO_HOME", str(Path(home) / ".cargo")), "/build/cargo"),
            (os.environ.get("RUSTUP_HOME", str(Path(home) / ".rustup")), "/build/rustup"),
        ]
    )
    return mappings


def sanitize_value(value: Any, replacements: Sequence[Tuple[str, str]]) -> Tuple[Any, bool]:
    if isinstance(value, str):
        sanitized = value
        for source, destination in replacements:
            sanitized = sanitized.replace(source, destination)
        return sanitized, sanitized != value
    if isinstance(value, list):
        changed = False
        output = []
        for item in value:
            sanitized, item_changed = sanitize_value(item, replacements)
            output.append(sanitized)
            changed = changed or item_changed
        return output, changed
    if isinstance(value, dict):
        changed = False
        output = {}
        for key, item in value.items():
            sanitized, item_changed = sanitize_value(item, replacements)
            output[key] = sanitized
            changed = changed or item_changed
        return output, changed
    return value, False


def iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from iter_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_strings(item)


def validate_sanitized(value: Any, replacements: Sequence[Tuple[str, str]]) -> None:
    sources = [source for source, _ in replacements]
    for item in iter_strings(value):
        if any(source in item for source in sources):
            raise SanitizationError(f"mapped build path remains in SBOM value: {item!r}")
        if any(pattern.search(item) for pattern in SENSITIVE_PATHS):
            raise SanitizationError(f"host path remains in SBOM value: {item!r}")


def sanitize_json_bytes(
    raw: bytes, replacements: Sequence[Tuple[str, str]]
) -> Tuple[bytes, bool]:
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SanitizationError(f"invalid UTF-8 JSON: {error}") from error
    sanitized, changed = sanitize_value(document, replacements)
    validate_sanitized(sanitized, replacements)
    if not changed:
        return raw, False
    rendered = (json.dumps(sanitized, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    return rendered, True


def _record_digest(content: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest())
    return "sha256=" + digest.rstrip(b"=").decode("ascii")


def _rewrite_record(
    record_name: str, original: bytes, contents: Dict[str, bytes]
) -> bytes:
    rows = list(csv.reader(io.StringIO(original.decode("utf-8"))))
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    seen = set()
    for row in rows:
        if len(row) != 3:
            raise SanitizationError(f"malformed wheel RECORD row: {row!r}")
        name = row[0]
        seen.add(name)
        if name == record_name:
            writer.writerow([name, "", ""])
        elif name in contents:
            content = contents[name]
            writer.writerow([name, _record_digest(content), str(len(content))])
        else:
            writer.writerow(row)
    for name in sorted(set(contents) - seen - {record_name}):
        content = contents[name]
        writer.writerow([name, _record_digest(content), str(len(content))])
    return output.getvalue().encode("utf-8")


def sanitize_json_file(
    path: Path, replacements: Sequence[Tuple[str, str]], check_only: bool
) -> int:
    rendered, changed = sanitize_json_bytes(path.read_bytes(), replacements)
    if check_only and changed:
        raise SanitizationError(f"{path} still contains mapped build paths")
    if changed:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            handle.write(rendered)
            temporary = Path(handle.name)
        os.replace(temporary, path)
    return 1


def _installed_record_context(path: Path) -> Optional[Tuple[Path, str, str]]:
    for parent in path.parents:
        if not parent.name.endswith(".dist-info"):
            continue
        record = parent / "RECORD"
        if not record.is_file():
            raise SanitizationError(f"installed metadata has no RECORD: {parent}")
        site_packages = parent.parent
        return (
            record,
            path.relative_to(site_packages).as_posix(),
            record.relative_to(site_packages).as_posix(),
        )
    return None


def _validate_record(
    record_name: str, original: bytes, contents: Dict[str, bytes]
) -> None:
    rows = {
        row[0]: row
        for row in csv.reader(io.StringIO(original.decode("utf-8")))
        if len(row) == 3
    }
    for name, content in contents.items():
        row = rows.get(name)
        if row is None:
            raise SanitizationError(f"installed RECORD has no entry for {name}")
        expected = [_record_digest(content), str(len(content))]
        if row[1:] != expected:
            raise SanitizationError(f"installed RECORD is stale for {name}")
    record_row = rows.get(record_name)
    if record_row is None or record_row[1:] != ["", ""]:
        raise SanitizationError("installed RECORD must contain an unhashed self-entry")


def sanitize_json_target(
    path: Path, replacements: Sequence[Tuple[str, str]], check_only: bool
) -> int:
    json_files = target_json_files(path)
    record_contents: Dict[Path, Tuple[str, Dict[str, bytes]]] = {}
    count = 0
    for json_path in json_files:
        count += sanitize_json_file(json_path, replacements, check_only)
        context = _installed_record_context(json_path)
        if context is None:
            continue
        record, member_name, record_name = context
        stored_record_name, contents = record_contents.setdefault(
            record, (record_name, {})
        )
        if stored_record_name != record_name:
            raise SanitizationError(f"inconsistent RECORD context for {json_path}")
        contents[member_name] = json_path.read_bytes()

    for record, (record_name, contents) in record_contents.items():
        original = record.read_bytes()
        if check_only:
            _validate_record(record_name, original, contents)
            continue
        rendered = _rewrite_record(record_name, original, contents)
        with tempfile.NamedTemporaryFile(dir=record.parent, delete=False) as handle:
            handle.write(rendered)
            temporary = Path(handle.name)
        os.replace(temporary, record)
    return count


def sanitize_wheel(
    path: Path, replacements: Sequence[Tuple[str, str]], check_only: bool
) -> int:
    with zipfile.ZipFile(path, "r") as archive:
        infos = archive.infolist()
        contents = {info.filename: archive.read(info.filename) for info in infos}

    sbom_names = [name for name in contents if SBOM_MEMBER.search(name)]
    if not sbom_names:
        raise SanitizationError(f"{path} contains no .dist-info/sboms/*.json files")

    changed = False
    for name in sbom_names:
        rendered, member_changed = sanitize_json_bytes(contents[name], replacements)
        contents[name] = rendered
        changed = changed or member_changed

    record_names = [name for name in contents if name.endswith(".dist-info/RECORD")]
    if len(record_names) != 1:
        raise SanitizationError(
            f"{path} must contain exactly one .dist-info/RECORD, found {len(record_names)}"
        )
    record_name = record_names[0]
    if check_only:
        if changed:
            raise SanitizationError(f"{path} still contains mapped build paths")
        _validate_record(
            record_name,
            contents[record_name],
            {name: contents[name] for name in sbom_names},
        )
        return len(sbom_names)

    rewritten_record = _rewrite_record(
        record_name, contents[record_name], contents
    )
    if not changed and rewritten_record == contents[record_name]:
        return len(sbom_names)
    contents[record_name] = rewritten_record

    with tempfile.NamedTemporaryFile(
        suffix=".whl", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            for info in infos:
                archive.writestr(info, contents[info.filename])
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return len(sbom_names)


def target_json_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise SanitizationError(f"SBOM target does not exist: {path}")
    files = [
        candidate
        for candidate in path.rglob("*.json")
        if "sboms" in candidate.parts
    ]
    if not files:
        raise SanitizationError(f"{path} contains no SBOM JSON files")
    return sorted(files)


def parse_mapping(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("mapping must be SOURCE=DESTINATION")
    source, destination = value.split("=", 1)
    if not source or not destination:
        raise argparse.ArgumentTypeError("mapping must be SOURCE=DESTINATION")
    return source, destination


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Cargo manifest used to derive locked local-package roots",
    )
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        type=parse_mapping,
        metavar="SOURCE=DESTINATION",
        help="additional deterministic prefix replacement",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify without modifying SBOMs",
    )
    args = parser.parse_args(argv)

    try:
        mappings = default_mappings(args.manifest_path)
        mappings.extend(args.map)
        replacements = build_replacements(mappings)
        count = 0
        for path in args.paths:
            if path.suffix == ".whl":
                count += sanitize_wheel(path, replacements, args.check)
            else:
                count += sanitize_json_target(path, replacements, args.check)
    except (OSError, subprocess.CalledProcessError, SanitizationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    action = "verified" if args.check else "sanitized"
    print(f"{action} {count} SBOM document(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
