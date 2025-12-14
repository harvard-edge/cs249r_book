# Section ID Management System

## Overview

The section ID management system provides automated tools for managing unique, consistent section IDs in Quarto/Markdown book projects. The system uses a **hierarchy-based approach** to generate stable, meaningful section IDs that reflect the actual document structure and ensures **global uniqueness** across the entire book project.

## Key Features

### Hierarchy-Based ID Generation
- **Stable IDs**: Section IDs remain consistent even when sections are reordered (as long as the hierarchy doesn't change)
- **Meaningful Structure**: IDs reflect the actual document organization and parent-child relationships
- **Natural Duplicate Handling**: Sections with the same name but different parents automatically get different IDs
- **No Counter Dependency**: No need to worry about section reordering affecting IDs
- **Global Uniqueness**: File path inclusion ensures unique IDs across the entire book project

### ID Format
```sec-{chapter-title}-{section-title}-{hash}
```

### Hash Generation
The hash is generated from:
```
{file_path}|{chapter_title}|{section_title}|{parent_hierarchy}
```

Where:
- `file_path`: The file path (ensures global uniqueness across different files)
- `chapter_title`: The chapter title
- `section_title`: The section title
- `parent_hierarchy`: A pipe-separated list of all parent sections (e.g., `parent1|parent2|parent3`)

## Global Uniqueness Guarantee

The inclusion of the file path in the hash generation ensures that sections with identical names and hierarchies in different files will have different IDs. This prevents conflicts when:

- Multiple chapters have sections with the same name (e.g., "Introduction" in different files)
- Different files have identical section hierarchies (e.g., "Techniques > Advanced > Optimization")
- The same section name appears in multiple contexts across the book

### Example: Same Section Name in Different Files

```markdown
# File: contents/chapter1.qmd
# Getting Started

## Introduction {#sec-getting-started-introduction-d212}

# File: contents/chapter2.qmd
# Getting Started

## Introduction {#sec-getting-started-introduction-8435}
```

Hash inputs:
- File 1: `"contents/chapter1.qmd|Getting Started|Introduction|"` → hash: `d212`
- File 2: `"contents/chapter2.qmd|Getting Started|Introduction|"` → hash: `8435`

Result: Different 4-character hashes ensure unique IDs across the entire book.

## How It Works

### 1. Hierarchy Tracking
The system maintains a stack of parent sections as it processes the document:

```python
section_hierarchy = []  # Stack of parent sections

# For each header level, update the hierarchy
while len(section_hierarchy) >= header_level - 1:
    section_hierarchy.pop()
section_hierarchy.append(title.strip())

# Get parent sections for current section
parent_sections = section_hierarchy[:-1] if len(section_hierarchy) > 1 else []
```

### 2. Hash Generation
```python
# Build hierarchy string from parent sections
hierarchy = ""
if parent_sections:
    hierarchy_parts = []
    for parent in parent_sections:
        hierarchy_parts.append(simple_slugify(parent))
    hierarchy = "|".join(hierarchy_parts)

# Generate hash with file path for global uniqueness
hash_input = f"{file_path}|{chapter_title}|{title}|{hierarchy}".encode('utf-8')
hash_suffix = hashlib.sha1(hash_input).hexdigest()[:4]
```

## Example

Consider a document with this structure:

```markdown
# Introduction

## AI Evolution

### Symbolic AI Era

#### Data Considerations

### Expert Systems Era

#### Data Considerations

### Deep Learning Era

#### Data Considerations
```

The three "Data Considerations" sections will get different IDs:

- `sec-introduction-data-considerations-d32a` (under Symbolic AI Era)
- `sec-introduction-data-considerations-8ae1` (under Expert Systems Era)
- `sec-introduction-data-considerations-fdab` (under Deep Learning Era)

## Benefits Over Counter-Based Approach

| Aspect | Counter-Based | Hierarchy-Based |
|--------|---------------|-----------------|
| **Stability** | Changes when sections reordered | Stable unless hierarchy changes |
| **Meaning** | Arbitrary position-based | Reflects document structure |
| **Duplicates** | Requires manual counter management | Handled naturally by context |
| **Maintenance** | Fragile to document changes | Robust and self-maintaining |
| **Global Uniqueness** | May conflict across files | Guaranteed by file path inclusion |

## Usage

### Basic Commands

```bash
# Add missing IDs
python section_id_manager.py -d contents/

# Repair existing IDs to new format
python section_id_manager.py -d contents/ --repair --backup

# Verify all IDs
python section_id_manager.py -d contents/ --verify

# List all IDs
python section_id_manager.py -d contents/ --list
```

### Safety Features

- **Backup Creation**: `--backup` creates timestamped backups
- **Dry Run**: `--dry-run` previews changes without modifying files
- **Interactive Prompts**: Confirms changes before applying
- **Force Mode**: `--force` automatically accepts all changes

## Migration from Counter-Based System

If you have existing counter-based IDs, the system will automatically migrate them:

1. Run repair mode: `python section_id_manager.py -d contents/ --repair --backup`
2. The system will update all IDs to the new hierarchy-based format
3. Cross-references will be automatically updated
4. Old IDs are preserved in the backup files

## Best Practices

1. **Use backups**: Always use `--backup` when making bulk changes
2. **Verify before commits**: Use `--verify` to ensure ID integrity
3. **Preview changes**: Use `--dry-run` to see what will change
4. **Consider automation**: Use in pre-commit hooks or CI pipelines

## Technical Details

### Function Signature
```python
def generate_section_id(title, file_path, chapter_title, section_counter, parent_sections=None):
```

### Parameters
- `title`: The section title
- `file_path`: The file path (included in hash for global uniqueness)
- `chapter_title`: The chapter title
- `section_counter`: Counter for this section (not used in hash)
- `parent_sections`: List of parent section titles (included in hash)

### Parent Sections Format
- `parent_sections` is a list of strings representing the full hierarchy
- Each parent is processed through `simple_slugify()` to remove stopwords
- Parents are joined with `|` separator in the hash input

### Hash Algorithm
- Uses SHA-1 for hash generation
- Takes first 4 hex characters for the suffix
- Ensures uniqueness while keeping IDs readable
- Includes file path to guarantee global uniqueness across the book project

## Troubleshooting

### Common Issues

1. **Duplicate IDs**: Should not occur with hierarchy-based system and file path inclusion
2. **Changing IDs**: IDs may change when document structure changes (this is expected)
3. **Cross-reference breaks**: Use `--repair` to update all references

### Debugging

- Use `--list` to see all current IDs
- Use `--verify` to check for missing or malformed IDs
- Check backup files if you need to revert changes
