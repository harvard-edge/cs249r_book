#!/usr/bin/env python3
"""
TeX Live Package Extractor for Quarto Projects

This script analyzes Quarto project files to extract LaTeX package dependencies
and generate a list of required TeX Live packages and collections. It searches
through both TEX header files and Quarto YAML configuration to find all
\\usepackage declarations and TikZ library usage.

The script uses tlmgr (TeX Live package manager) to map LaTeX package names
to their corresponding TeX Live packages and collections, making it easier
to install the correct dependencies for building the project.

Usage:
    python mk_tl_packages.py

Output:
    Creates a 'tl_packages' file containing:
    - TeX Live collections that need to be installed
    - Individual packages not part of collections
    - Packages that couldn't be found (for manual review)

Dependencies:
    - tlmgr (TeX Live package manager)
    - PyYAML for parsing Quarto YAML files
"""

import re
import subprocess
import yaml
from pathlib import Path

# Configuration constants
TEX_FILE = "tex/header-includes.tex"  # Path to TEX header file
QUARTO_YML = "_quarto.yml"            # Path to Quarto configuration
OUTPUT_FILE = "tl_packages"           # Output file name

def extract_from_tex_file(path):
    """
    Extract LaTeX package dependencies from a TEX header file.
    
    Parses the file line by line, looking for:
    - \\usepackage declarations
    - \\usetikzlibrary commands (adds 'pgf' package)
    - \\usepgfplotslibrary commands (adds 'pgfplots' package)
    
    Args:
        path (Path): Path to the TEX file to analyze
        
    Returns:
        tuple: (components_set, include_pgf_bool, include_pgfplots_bool)
            - components_set: Set of package names found
            - include_pgf_bool: True if TikZ libraries are used
            - include_pgfplots_bool: True if PGFPlots libraries are used
    """
    components = set()
    include_pgf = include_pgfplots = False

    print(f"🔍 Extracting from TEX file: {path}")
    
    if not path.exists():
        print(f"   ⚠️  File does not exist: {path}")
        return components, False, False

    print(f"   📖 Reading file: {path}")
    content = path.read_text(encoding="utf-8")
    print(f"   📄 File size: {len(content)} characters")
    
    # Process each line, ignoring comments (everything after %)
    for line_num, line in enumerate(content.splitlines(), 1):
        original_line = line
        # Remove comments (everything after %)
        line = line.split('%')[0].strip()
        
        # Check for TikZ library usage
        if "\\usetikzlibrary" in line:
            include_pgf = True
            print(f"   📍 Line {line_num}: Found \\usetikzlibrary")
        if "\\usepgfplotslibrary" in line:
            include_pgfplots = True
            print(f"   📍 Line {line_num}: Found \\usepgfplotslibrary")
            
        # Extract package names from \\usepackage declarations
        # Regex matches: \usepackage[options]{package1,package2}
        match = re.findall(r'\\usepackage(?:\[[^\]]*\])?{([^}]+)}', line)
        for entry in match:
            # Split comma-separated packages and clean whitespace
            packages = [pkg.strip() for pkg in entry.split(',') if pkg.strip()]
            components.update(packages)
            if packages:
                print(f"   📦 Line {line_num}: Found packages: {packages}")
    
    print(f"   ✅ Extracted {len(components)} packages from TEX file")
    print(f"   📊 PGF: {include_pgf}, PGFPlots: {include_pgfplots}")
    return components, include_pgf, include_pgfplots

def extract_from_quarto_yml(path):
    """
    Extract LaTeX package dependencies from Quarto YAML configuration.
    
    Searches for tikz configuration in the YAML file and extracts packages
    from the header-includes section. Handles the same package patterns as
    the TEX file extractor.
    
    Args:
        path (Path): Path to the Quarto YAML configuration file
        
    Returns:
        tuple: (components_set, include_pgf_bool, include_pgfplots_bool)
            - components_set: Set of package names found
            - include_pgf_bool: True if TikZ libraries are used
            - include_pgfplots_bool: True if PGFPlots libraries are used
    """
    components = set()
    include_pgf = include_pgfplots = False

    print(f"🔍 Extracting from Quarto YAML: {path}")
    
    if not path.exists():
        print(f"   ⚠️  File does not exist: {path}")
        return components, False, False

    print(f"   📖 Reading file: {path}")
    content = path.read_text(encoding="utf-8")
    print(f"   📄 File size: {len(content)} characters")
    
    # Parse YAML content
    yml = yaml.safe_load(content)
    tikz_config = None

    def find_tikz_config(node):
        """
        Recursively search for tikz configuration in YAML structure.
        
        Looks for a 'tikz' key with a 'header-includes' subkey containing
        LaTeX package declarations.
        """
        nonlocal tikz_config
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "tikz" and isinstance(v, dict) and "header-includes" in v:
                    tikz_config = v
                    print(f"   🎯 Found tikz configuration")
                else:
                    find_tikz_config(v)
        elif isinstance(node, list):
            for item in node:
                find_tikz_config(item)

    # Search for tikz configuration in the YAML structure
    find_tikz_config(yml)

    if not tikz_config:
        print(f"   ⚠️  No tikz configuration found in YAML")
        return components, False, False

    # Process header-includes from tikz configuration
    header_includes = tikz_config.get("header-includes", [])
    print(f"   📋 Processing {len(header_includes)} header-includes")
    
    for line_num, line in enumerate(header_includes, 1):
        # Check for TikZ library usage
        if "\\usetikzlibrary" in line:
            include_pgf = True
            print(f"   📍 Header {line_num}: Found \\usetikzlibrary")
        if "\\usepgfplotslibrary" in line:
            include_pgfplots = True
            print(f"   📍 Header {line_num}: Found \\usepgfplotslibrary")
            
        # Extract package names from \\usepackage declarations
        match = re.findall(r'\\usepackage(?:\[[^\]]*\])?{([^}]+)}', line)
        for entry in match:
            packages = [pkg.strip() for pkg in entry.split(',') if pkg.strip()]
            components.update(packages)
            if packages:
                print(f"   📦 Header {line_num}: Found packages: {packages}")
    
    print(f"   ✅ Extracted {len(components)} packages from YAML")
    print(f"   📊 PGF: {include_pgf}, PGFPlots: {include_pgfplots}")
    return components, include_pgf, include_pgfplots

def find_tlmgr_package(component):
    """
    Find the TeX Live package that provides a given LaTeX component.
    
    Uses tlmgr to search for the .sty file corresponding to the component
    and then gets information about which TeX Live collection it belongs to.
    
    Args:
        component (str): LaTeX package name (e.g., 'geometry', 'graphicx')
        
    Returns:
        tuple: (package_name, collection_name)
            - package_name: TeX Live package name, or None if not found
            - collection_name: TeX Live collection name, or None if not in a collection
    """
    print(f"   🔎 Looking up package for component: {component}")
    try:
        # Search for the .sty file in TeX Live packages
        cmd = ["tlmgr", "search", "--file", f"/{component}.sty"]
        print(f"   💻 Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )
        
        if not result.stdout.strip():
            print(f"   ❌ No tlmgr package found for {component}")
            return (None, None)
            
        # Extract package name from tlmgr output
        pkg = result.stdout.split(":")[0].strip()
        print(f"   ✅ Found tlmgr package: {pkg}")
        
        # Get detailed information about the package
        info_cmd = ["tlmgr", "info", pkg]
        print(f"   💻 Running: {' '.join(info_cmd)}")
        
        info = subprocess.run(info_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        
        # Extract collection information from package details
        coll = re.search(r"collection:\s*(\S+)", info.stdout)
        
        if coll:
            collection = coll.group(1)
            print(f"   📚 Package {pkg} belongs to collection: {collection}")
            return pkg, collection
        else:
            print(f"   📦 Package {pkg} is not part of a collection")
            return pkg, None
            
    except Exception as e:
        print(f"   💥 Error looking up {component}: {e}")
        return None, None

def main():
    """
    Main function that orchestrates the package extraction process.
    
    The process consists of three phases:
    1. Extract packages from TEX and YAML files
    2. Look up TeX Live packages and collections
    3. Generate the output file with organized results
    """
    print("🚀 Starting TeX Live package extraction...")
    print("=" * 50)
    
    # Initialize data structures for tracking results
    components = set()      # All unique package names found
    explicit = set()        # Packages not part of collections
    collections = set()     # TeX Live collections needed
    missing = set()         # Packages that couldn't be found

    # Phase 1: Extract packages from source files
    print("\n📂 PHASE 1: Extracting packages from source files")
    print("-" * 40)
    tex_components, tex_pgf, tex_pgfplots = extract_from_tex_file(Path(TEX_FILE))
    yml_components, yml_pgf, yml_pgfplots = extract_from_quarto_yml(Path(QUARTO_YML))

    # Combine results from both sources
    components.update(tex_components)
    components.update(yml_components)
    
    # Add special packages based on TikZ usage
    if tex_pgf or yml_pgf:
        components.add("pgf")
        print("   ➕ Added 'pgf' due to tikzlibrary usage")
    if tex_pgfplots or yml_pgfplots:
        components.add("pgfplots")
        print("   ➕ Added 'pgfplots' due to pgfplotslibrary usage")

    # Display summary of found components
    print(f"\n📋 SUMMARY: Found {len(components)} unique components:")
    for comp in sorted(components):
        print(f"   • {comp}")

    # Phase 2: Look up TeX Live packages and collections
    print(f"\n🔍 PHASE 2: Looking up TeX Live packages")
    print("-" * 40)
    for comp in sorted(components):
        print(f"\n🔎 Processing component: {comp}")
        pkg, coll = find_tlmgr_package(comp)
        
        # Categorize the result
        if coll:
            collections.add(coll)
            print(f"   ✅ Added to collections: {coll}")
        elif pkg:
            explicit.add(pkg)
            print(f"   ✅ Added to explicit packages: {pkg}")
        else:
            missing.add(comp)
            print(f"   ❌ Added to missing: {comp}")

    # Phase 3: Generate output file
    print(f"\n📝 PHASE 3: Writing output file")
    print("-" * 40)
    print(f"   📄 Writing to: {OUTPUT_FILE}")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        # Write header comment
        f.write("# Auto-generated TeX Live package list\n\n")
        
        # Write collections section
        f.write("# Collections:\n")
        for c in sorted(collections):
            f.write(f"{c}\n")
        
        # Write explicit packages section (if any)
        if explicit:
            f.write("\n# Explicit packages (not in collections):\n")
            for p in sorted(explicit):
                f.write(f"{p}\n")
        
        # Write missing packages section (if any)
        if missing:
            f.write("\n# Not found via tlmgr (check manually):\n")
            for m in sorted(missing):
                f.write(f"# {m}\n")

    # Display final summary
    print(f"\n✅ FINAL SUMMARY:")
    print(f"   📚 Collections: {len(collections)}")
    for c in sorted(collections):
        print(f"      • {c}")
    print(f"   📦 Explicit packages: {len(explicit)}")
    for p in sorted(explicit):
        print(f"      • {p}")
    print(f"   ❓ Missing/unknown: {len(missing)}")
    for m in sorted(missing):
        print(f"      • {m}")
    
    print(f"\n🎉 Successfully wrote {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
