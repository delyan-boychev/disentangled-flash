import re
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python bump_version.py [major|minor|patch]")
        sys.exit(1)

    bump_type = sys.argv[1].lower()
    if bump_type not in ("major", "minor", "patch"):
        print("Invalid bump type. Choose from: major, minor, patch")
        sys.exit(1)

    root = Path(__file__).parent.parent

    # 1. Read current version from pyproject.toml
    pyproject_path = root / "pyproject.toml"
    pyproject_content = pyproject_path.read_text(encoding="utf-8")

    version_match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject_content, re.MULTILINE)
    if not version_match:
        print("Could not find version in pyproject.toml")
        sys.exit(1)

    current_version = version_match.group(1)
    major, minor, patch = map(int, current_version.split("."))

    if bump_type == "major":
        new_version = f"{major + 1}.0.0"
    elif bump_type == "minor":
        new_version = f"{major}.{minor + 1}.0"
    else:
        new_version = f"{major}.{minor}.{patch + 1}"

    print(f"Bumping version from {current_version} to {new_version}")

    # 2. Update pyproject.toml
    pyproject_content = re.sub(
        r'^version\s*=\s*"[^"]+"',
        f'version = "{new_version}"',
        pyproject_content,
        flags=re.MULTILINE,
    )
    pyproject_path.write_text(pyproject_content, encoding="utf-8")

    # 3. Update __init__.py
    init_path = root / "src" / "disentangled_flash" / "__init__.py"
    if init_path.exists():
        init_content = init_path.read_text(encoding="utf-8")
        init_content = re.sub(
            r'^__version__\s*=\s*"[^"]+"',
            f'__version__ = "{new_version}"',
            init_content,
            flags=re.MULTILINE,
        )
        init_path.write_text(init_content, encoding="utf-8")

    # 4. Update CITATION.cff
    citation_path = root / "CITATION.cff"
    if citation_path.exists():
        citation_content = citation_path.read_text(encoding="utf-8")
        citation_content = re.sub(
            r"^version:\s*\S+", f"version: {new_version}", citation_content, flags=re.MULTILINE
        )
        citation_path.write_text(citation_content, encoding="utf-8")

    # 5. Update README.md (bibtex version)
    readme_path = root / "README.md"
    if readme_path.exists():
        readme_content = readme_path.read_text(encoding="utf-8")
        readme_content = re.sub(
            r"version\s*=\s*\{[^}]+\}", f"version = {{{new_version}}}", readme_content
        )
        readme_path.write_text(readme_content, encoding="utf-8")

    print("Version bump successful!")


if __name__ == "__main__":
    main()
