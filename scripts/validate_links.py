#!/usr/bin/env python3
"""Validates all internal Markdown links, anchors, and image references across the repository."""

from __future__ import annotations

import glob
import os
import re
import sys
import xml.etree.ElementTree as ET


def slugify(heading: str) -> str:
    """Converts a markdown heading into a GitHub/MkDocs anchor slug."""
    h = heading.strip().lstrip("#").strip().lower()
    h = re.sub(r"[^\w\s-]", "", h)
    return h.replace(" ", "-")


def main() -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(repo_root)

    md_files: list[str] = []
    headings: dict[str, set[str]] = {}

    ignore_dirs = {".git", ".pytest_cache", ".venv", "venv", "site", "__pycache__"}

    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for f in files:
            if f.endswith(".md") and not os.path.islink(os.path.join(root, f)):
                rel = os.path.relpath(os.path.join(root, f), ".").replace("\\", "/")
                md_files.append(rel)
                with open(rel, encoding="utf-8") as fp:
                    headings[rel] = {slugify(line) for line in fp if line.startswith("#")}

    errors: list[str] = []

    for rel in sorted(md_files):
        with open(rel, encoding="utf-8") as fp:
            content = fp.read()

        # Remove fenced code blocks so code syntax isn't misidentified as markdown links
        stripped = re.sub(r"```.*?```", "", content, flags=re.DOTALL)

        for m in re.finditer(r"\[([^\]]*)\]\(([^)]+)\)", stripped):
            target = m.group(2).strip()
            if target.startswith(("http://", "https://", "mailto:", "tel:")):
                continue

            path_part, _, anchor = target.partition("#")
            if path_part:
                resolved = os.path.normpath(os.path.join(os.path.dirname(rel), path_part))
            else:
                resolved = rel

            resolved_rel = os.path.relpath(resolved, ".").replace("\\", "/")

            if not os.path.exists(resolved):
                errors.append(f"{rel}: missing target file '{target}' (resolved: '{resolved_rel}')")
            elif anchor:
                if resolved_rel in headings and anchor not in headings[resolved_rel]:
                    # Check partial matches for complex headings
                    matched = any(anchor in h for h in headings[resolved_rel])
                    if not matched:
                        errors.append(f"{rel}: broken anchor '#{anchor}' in '{resolved_rel}'")

        for im in re.finditer(r'<img[^>]+src=[\'"]([^\'"]+)[\'"]', stripped):
            src_target = im.group(1).strip()
            if src_target.startswith(("http://", "https://", "data:")):
                continue
            resolved_img = os.path.normpath(os.path.join(os.path.dirname(rel), src_target))
            if not os.path.exists(resolved_img):
                errors.append(f"{rel}: missing image <img src='{src_target}'>")

    # Validate SVG integrity in assets
    for svg_path in sorted(glob.glob("assets/**/*.svg", recursive=True)):
        try:
            ET.parse(svg_path)
        except ET.ParseError as exc:
            errors.append(f"Invalid XML syntax in {svg_path}: {exc}")

    if errors:
        print(f"Validation FAILED with {len(errors)} error(s):")
        for err in errors:
            print(f"  ❌ {err}")
        return 1

    print(f"Validation PASSED! Verified {len(md_files)} Markdown files with 0 broken links or anchors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
