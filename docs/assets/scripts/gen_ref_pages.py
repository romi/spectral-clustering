"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files
import logging

# Create an object to manage the navigation structure of the documentation
nav = mkdocs_gen_files.Nav()

# Set up the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API Builder")

# Define the source directory where the Python code resides
src = Path(__file__).parent.parent.parent.parent / "src"
logger.info(f"Browsing source code in {src}...")

# Recursively scan all Python files in the source directory
for path in sorted(src.rglob("*.py")):
    logger.info(f"Processing {Path(path).relative_to(src)}...")
    # Get the module path (relative path without file extension)
    module_path = path.relative_to(src).with_suffix("")
    # Create the corresponding documentation path ending in .md
    doc_path = path.relative_to(src).with_suffix(".md")
    # Full path in the "reference" directory for generated Markdown files
    full_doc_path = Path("reference", doc_path)

    # Extract the individual path parts (used for navigation hierarchy)
    parts = tuple(module_path.parts)

    # Special handling for __init__.py: treat as index.md in navigation
    if parts[-1] == "__init__":
        parts = parts[:-1]  # Exclude the "__init__" part
        doc_path = doc_path.with_name("index.md")  # Rename to index.md
        full_doc_path = full_doc_path.with_name("index.md")

    # Skip any __main__.py files as they shouldn't be included in the docs
    elif parts[-1] == "__main__":
        continue

    # Add the module to the navigation structure, converting to a POSIX path
    nav[parts] = doc_path.as_posix()

    # Write the documentation stub for the module
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)  # Module identifier (e.g., package.module)
        # Write the `:::` directive for mkdocs-material plugin to include the module docs
        fd.write(f"::: {ident}")

    # Assign an edit path for the generated Markdown file
    mkdocs_gen_files.set_edit_path(full_doc_path, path)

# Create a summary file (reference index) with the full navigation structure
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    # Write the navigation structure in a human-readable format
    nav_file.writelines(nav.build_literate_nav())
