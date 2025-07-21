#!/usr/bin/env python3
"""
ML Systems Cross-Reference Generator
====================================

Complete toolkit for domain adaptation and cross-reference generation.

MODES:
    Training Mode:
        python3 cross_referencing.py -t -d ../../contents/core/ -o ./my_model
        python3 cross_referencing.py --train --dirs ../../contents/core/ --output ./my_model --base-model sentence-t5-base --epochs 5
    
    Generation Mode (Domain-adapted):
        python3 cross_referencing.py -g -m ./t5-mlsys-domain-adapted -o cross_refs.json
        python3 cross_referencing.py --generate --model ./t5-mlsys-domain-adapted --output cross_refs.json
    
    Generation Mode (Base model):
        python3 cross_referencing.py -g -m sentence-t5-base -o cross_refs.json
        python3 cross_referencing.py --generate --model all-MiniLM-L6-v2 --output cross_refs.json

TRAINING:
    • Extracts content from specified directories
    • Excludes introduction/conclusion chapters (champion approach)
    • Creates sophisticated training examples with nuanced similarity labels
    • Domain-adapts base model using contrastive learning
    • Saves trained model for later use

GENERATION:
    • Works with domain-adapted models OR base sentence-transformer models
    • Extracts sections and generates embeddings
    • Finds cross-references with 65%+ similarity threshold
    • Outputs Lua-compatible JSON for inject_xrefs.lua

REQUIREMENTS:
    pip install sentence-transformers scikit-learn numpy torch pyyaml
"""

import json
import numpy as np
import argparse
import sys
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Any, cast

def get_quarto_file_order() -> List[str]:
    """Extract file order from _quarto.yml chapters section, including commented lines."""
    quarto_yml_path = Path.cwd() / "_quarto.yml"
    
    if not quarto_yml_path.exists():
        print(f"❌ Error: _quarto.yml not found at {quarto_yml_path}")
        print("   Please run this script from the project root directory where _quarto.yml is located.")
        sys.exit(1)
    
    try:
        with open(quarto_yml_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        ordered_files = []
        in_chapters_section = False
        
        for line in lines:
            stripped = line.strip()
            
            # Detect chapters section
            if stripped.startswith('chapters:'):
                in_chapters_section = True
                continue
            
            # End of chapters section (next major section)
            if in_chapters_section and stripped and not stripped.startswith(('-', '#')):
                if ':' in stripped and not stripped.startswith('  '):
                    break
            
            if in_chapters_section:
                # Extract file paths from both commented and uncommented lines
                if stripped.startswith('- ') or stripped.startswith('#     - '):
                    # Remove comment markers and list indicators
                    file_part = stripped.replace('# ', '').replace('- ', '').strip()
                    
                    # Handle 'part:' entries
                    if file_part.startswith('part: '):
                        file_part = file_part.replace('part: ', '').strip()
                    
                    # Skip text entries and empty lines
                    if (file_part.endswith('.qmd') and 
                        not file_part.startswith('text:') and
                        'contents/core/' in file_part):
                        ordered_files.append(file_part)
        
        print(f"📋 Found {len(ordered_files)} ordered files in _quarto.yml (including commented)")
        if ordered_files:
            print("🔍 _quarto.yml file order preview:")
            for i, file_path in enumerate(ordered_files[:5], 1):
                print(f"    {i}. {file_path}")
            if len(ordered_files) > 5:
                print(f"    ... and {len(ordered_files) - 5} more")
        return ordered_files
        
    except Exception as e:
        print(f"⚠️  Warning: Could not parse _quarto.yml: {e}")
        return []

def find_qmd_files(directories: List[str]) -> List[str]:
    """Find all .qmd files in directories, ordered by _quarto.yml if available."""
    # Get all qmd files
    all_qmd_files = []
    for directory in directories:
        for path in Path(directory).rglob("*.qmd"):
            all_qmd_files.append(str(path))
    
    # Get order from _quarto.yml
    quarto_order = get_quarto_file_order()
    
    if not quarto_order:
        # Fallback to alphabetical sorting
        print("📂 Using alphabetical file ordering (no _quarto.yml order found)")
        return sorted(list(set(all_qmd_files)))
    
    # Create mapping for efficient lookup
    all_files_set = set(all_qmd_files)
    ordered_files = []
    
    # First, add files in _quarto.yml order
    for ordered_file in quarto_order:
        # Try different path combinations to match
        possible_paths = [
            ordered_file,
            str(Path.cwd() / "../../" / ordered_file),
            str(Path(ordered_file).resolve()) if Path(ordered_file).exists() else None
        ]
        
        # Also try matching by filename pattern
        filename = Path(ordered_file).name
        for discovered_file in all_files_set.copy():
            if Path(discovered_file).name == filename:
                # Extra check: ensure it's the same chapter directory
                ordered_chapter = Path(ordered_file).parent.name
                discovered_chapter = Path(discovered_file).parent.name
                if ordered_chapter == discovered_chapter:
                    ordered_files.append(discovered_file)
                    all_files_set.remove(discovered_file)
                    break
        else:
            # If no filename match, try the path-based matching
            for possible_path in possible_paths:
                if possible_path and possible_path in all_files_set:
                    ordered_files.append(possible_path)
                    all_files_set.remove(possible_path)
                    break
    
    # Add any remaining files alphabetically  
    remaining_files = sorted(list(all_files_set))
    ordered_files.extend(remaining_files)
    
    print(f"📊 File ordering: {len(ordered_files)} total ({len(ordered_files) - len(remaining_files)} from _quarto.yml, {len(remaining_files)} alphabetical)")
    
    return ordered_files

def extract_sections(file_path: str) -> List[Dict]:
    """Extract sections from a Quarto markdown file."""
    sections = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        current_section = None
        section_content = []
        
        for line in lines:
            if line.startswith('## ') or line.startswith('### '):
                # Save previous section
                if current_section and section_content:
                    clean_content = '\n'.join(section_content).strip()
                    if len(clean_content) > 100:  # Only substantial sections
                        sections.append({
                            'file_path': file_path,
                            'title': current_section,
                            'content': clean_content[:1500]  # Limit for embeddings
                        })
                
                # Start new section
                current_section = line.replace('#', '').strip()
                section_content = []
            else:
                section_content.append(line)
        
        # Save final section
        if current_section and section_content:
            clean_content = '\n'.join(section_content).strip()
            if len(clean_content) > 100:
                sections.append({
                    'file_path': file_path,
                    'title': current_section,
                    'content': clean_content[:1500]
                })
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return sections

def load_content(directories: List[str], exclude_chapters: Optional[List[str]] = None) -> List[Dict]:
    """Load and filter content from directories."""
    if exclude_chapters is None:
        exclude_chapters = []  # Default to including all chapters
    
    print(f"📚 Loading content from: {', '.join(directories)}")
    qmd_files = find_qmd_files(directories)
    
    print(f"📋 Found {len(qmd_files)} .qmd files:")
    for i, file_path in enumerate(qmd_files, 1):
        try:
            relative_path = str(Path(file_path).relative_to(Path.cwd()))
        except ValueError:
            relative_path = str(Path(file_path))
        print(f"    {i:2d}. {relative_path}")
    
    all_sections = []
    processed_files = []
    excluded_files = []
    
    for file_path in qmd_files:
        sections = extract_sections(file_path)
        if sections:
            # Apply filtering
            filtered_sections = []
            chapter_name = Path(file_path).parent.name
            is_excluded = any(excluded in chapter_name.lower() for excluded in exclude_chapters)
            
            if is_excluded:
                excluded_files.append(file_path)
            else:
                filtered_sections = sections
                processed_files.append(file_path)
                all_sections.extend(filtered_sections)
    
    print(f"\n✅ PROCESSING SUMMARY:")
    print(f"📖 Processing {len(processed_files)} files:")
    for i, file_path in enumerate(processed_files, 1):
        try:
            relative_path = str(Path(file_path).relative_to(Path.cwd()))
        except ValueError:
            relative_path = str(Path(file_path))
        chapter_name = Path(file_path).parent.name
        sections_count = len([s for s in all_sections if s['file_path'] == file_path])
        print(f"    {i:2d}. {relative_path} [{chapter_name}] ({sections_count} sections)")
    
    if excluded_files:
        print(f"\n🚫 Excluded {len(excluded_files)} files ({', '.join(exclude_chapters)} chapters):")
        for i, file_path in enumerate(excluded_files, 1):
            try:
                relative_path = str(Path(file_path).relative_to(Path.cwd()))
            except ValueError:
                relative_path = str(Path(file_path))
            chapter_name = Path(file_path).parent.name
            print(f"    {i:2d}. {relative_path} [{chapter_name}]")
    
    print(f"\n📊 FINAL COUNTS:")
    print(f"    • Total files found: {len(qmd_files)}")
    print(f"    • Files processed: {len(processed_files)}")
    print(f"    • Files excluded: {len(excluded_files)}")
    print(f"    • Sections extracted: {len(all_sections)}")
    
    return all_sections

def create_training_examples(sections: List[Dict]) -> List:
    """Create sophisticated training examples for domain adaptation."""
    try:
        from sentence_transformers import InputExample
    except ImportError:
        print("❌ sentence-transformers not installed. Run: pip install sentence-transformers")
        return []
    
    train_examples = []
    
    # Group by chapter
    chapters = {}
    for section in sections:
        chapter = Path(section['file_path']).parent.name
        if chapter not in chapters:
            chapters[chapter] = []
        chapters[chapter].append(section)
    
    chapter_list = list(chapters.items())
    
    print("🎯 Creating sophisticated training examples...")
    
    # Same chapter = high similarity (75-85%)
    for chapter_sections in chapters.values():
        if len(chapter_sections) > 1:
            for i, s1 in enumerate(chapter_sections):
                for s2 in chapter_sections[i+1:]:
                    similarity = np.random.uniform(0.75, 0.85)
                    train_examples.append(InputExample(
                        texts=[s1['content'], s2['content']], 
                        label=similarity
                    ))
    
    # Adjacent chapters = medium similarity (40-60%)
    for i, (_, sections1) in enumerate(chapter_list[:-1]):
        _, sections2 = chapter_list[i+1]
        for s1 in sections1[:2]:
            for s2 in sections2[:2]:
                similarity = np.random.uniform(0.4, 0.6)
                train_examples.append(InputExample(
                    texts=[s1['content'], s2['content']], 
                    label=similarity
                ))
    
    # Distant chapters = low similarity (10-30%)
    for i, (_, sections1) in enumerate(chapter_list):
        for j, (_, sections2) in enumerate(chapter_list):
            if abs(i - j) >= 3:
                for s1 in sections1[:1]:
                    for s2 in sections2[:1]:
                        similarity = np.random.uniform(0.1, 0.3)
                        train_examples.append(InputExample(
                            texts=[s1['content'], s2['content']], 
                            label=similarity
                        ))
    
    # Random negative examples
    import random
    random.seed(42)
    for _ in range(50):
        s1, s2 = random.sample(sections, 2)
        similarity = np.random.uniform(0.05, 0.25)
        train_examples.append(InputExample(
            texts=[s1['content'], s2['content']], 
            label=similarity
        ))
    
    print(f"✅ Created {len(train_examples)} training examples")
    return train_examples

def train_model(directories: List[str], output_path: str, base_model: str = "sentence-t5-base", 
                epochs: int = 5, exclude_chapters: Optional[List[str]] = None) -> bool:
    """Train a domain-adapted model."""
    print("🔥 TRAINING MODE: Domain Adaptation")
    print("=" * 50)
    
    try:
        from sentence_transformers import SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except ImportError:
        print("❌ Required packages not installed. Run:")
        print("   pip install sentence-transformers torch")
        return False
    
    # Load content
    all_sections = load_content(directories, exclude_chapters)
    if len(all_sections) < 50:
        print(f"❌ Need at least 50 sections for training, got {len(all_sections)}")
        return False
    
    # Create training examples
    train_examples = create_training_examples(all_sections)
    if not train_examples:
        return False
    
    # Load base model
    print(f"🧠 Loading base model: {base_model}")
    try:
        model = SentenceTransformer(base_model)
        print(f"✅ Model loaded: {model.get_sentence_embedding_dimension()} dimensions")
    except Exception as e:
        print(f"❌ Failed to load base model: {e}")
        return False
    
    # Setup training
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    def collate_fn(batch):
        return batch
    
    train_dataloader = DataLoader(
        cast(Any, train_examples), 
        batch_size=8, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    warmup_steps = int(len(train_examples) // 8 * 0.2)
    
    # Train model
    print(f"🚀 Training for {epochs} epochs...")
    print(f"📊 Training examples: {len(train_examples)}")
    print(f"📊 Batch size: 8")
    print(f"📊 Warmup steps: {warmup_steps}")
    
    import time
    start_time = time.time()
    
    try:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)], 
            epochs=epochs,
            warmup_steps=warmup_steps,
            show_progress_bar=True,
            output_path=output_path
        )
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.1f} seconds")
        print(f"💾 Model saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def extract_section_id(title: str) -> str:
    """Extract section ID from title with pattern {sec-something}."""
    import re
    match = re.search(r'\{(sec-[^}]+)\}', title)
    return match.group(1) if match else ""

def clean_title(title: str) -> str:
    """Remove section ID from title for display."""
    import re
    return re.sub(r'\s*\{[^}]+\}', '', title).strip()

def generate_cross_references(model_path: str, directories: List[str], output_file: str, 
                            exclude_chapters: Optional[List[str]] = None,
                            max_suggestions: int = 5,
                            similarity_threshold: float = 0.65,
                            verbose: bool = False) -> Dict:
    """Generate cross-references using any sentence-transformer model."""
    
    print("🚀 GENERATION MODE: Cross-Reference Generation")
    print("=" * 50)
    
    # Determine if model_path is a local path or HuggingFace model name
    is_local_model = Path(model_path).exists()
    model_type = "Domain-adapted" if is_local_model else "Base HuggingFace"
    
    print(f"📂 Model: {model_path} ({model_type})")
    
    # Load model
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_path)
        print(f"✅ Model loaded: {model.get_sentence_embedding_dimension()} dimensions")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return {}
    
    # Load content
    all_sections = load_content(directories, exclude_chapters)
    if not all_sections:
        print("❌ No content loaded")
        return {}
    
    # Generate embeddings
    print("🧮 Generating embeddings...")
    contents = [section['content'] for section in all_sections]
    embeddings = model.encode(contents, show_progress_bar=True)
    
    # Find similar sections
    print("🔍 Finding cross-references...")
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        print("❌ scikit-learn not installed. Run: pip install scikit-learn")
        return {}
    
    # Get file order for determining connection type
    file_order = find_qmd_files(directories)
    file_index_map = {file_path: i for i, file_path in enumerate(file_order)}

    nn_model = NearestNeighbors(n_neighbors=min(10, len(all_sections)), metric='cosine')
    nn_model.fit(embeddings)
    distances, indices = nn_model.kneighbors(embeddings)
    
    # Use a dictionary to collect cross-references by file and section
    file_section_refs = {}
    
    for i, section in enumerate(all_sections):
        source_file = section['file_path']
        source_filename = Path(source_file).name
        source_chapter = Path(section['file_path']).parent.name
        source_section_id = extract_section_id(section['title'])
        
        if not source_section_id:
            continue
            
        if source_filename not in file_section_refs:
            file_section_refs[source_filename] = {}
            
        if source_section_id not in file_section_refs[source_filename]:
            file_section_refs[source_filename][source_section_id] = {
                'section_title': clean_title(section['title']),
                'targets': []
            }
        
        suggestions_count = 0
        
        for j in range(1, min(10, len(indices[i]))):
            if suggestions_count >= max_suggestions:
                break

            target_idx = indices[i][j]
            target_section = all_sections[target_idx]
            target_chapter = Path(target_section['file_path']).parent.name
            
            similarity = 1 - distances[i][j]
            
            # Filter: different chapter, good similarity
            if (similarity > similarity_threshold and 
                source_chapter != target_chapter):
                
                target_id = extract_section_id(target_section['title'])

                if target_id:
                    # Determine connection type
                    source_idx = file_index_map.get(source_file, -1)
                    target_idx_map = file_index_map.get(target_section['file_path'], -1)
                    connection_type = "related" # Default
                    if source_idx != -1 and target_idx_map != -1:
                        if target_idx_map > source_idx:
                            connection_type = "Preview"
                        else:
                            connection_type = "Foundation"

                    file_section_refs[source_filename][source_section_id]['targets'].append({
                        'target_section_id': target_id,
                        'target_section_title': clean_title(target_section['title']),
                        'connection_type': connection_type,
                        'similarity': float(similarity)
                    })
                    suggestions_count += 1
    
    # Convert to final array structure
    cross_references = []
    for filename, sections in file_section_refs.items():
        if sections:  # Only include files that have sections with targets
            file_entry = {
                'file': filename,
                'sections': []
            }
            
            for section_id, section_data in sections.items():
                if section_data['targets']:  # Only include sections that have targets
                    file_entry['sections'].append({
                        'section_id': section_id,
                        'section_title': section_data['section_title'],
                        'targets': section_data['targets']
                    })
            
            if file_entry['sections']:  # Only add file if it has sections with targets
                cross_references.append(file_entry)
    
    total_refs = sum(len(section['targets']) for file_entry in cross_references for section in file_entry['sections'])

    # Save results
    result = {
        'metadata': {
            'generated_at': str(np.datetime64('now')),
            'model_used': model_path,
            'model_type': model_type,
            'total_sections': len(all_sections),
            'total_cross_references': total_refs,
            'approach': 'domain_adapted_t5' if is_local_model else 'base_model'
        },
        'cross_references': cross_references
    }
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ Generated {total_refs} cross-references across {len(cross_references)} files.")
    all_sims = [target['similarity'] for file_entry in cross_references for section in file_entry['sections'] for target in section['targets']]
    print(f"📊 Average similarity: {np.mean(all_sims):.3f}" if all_sims else "📊 No valid cross-references")
    print(f"📄 Results saved to: {output_file}")
    print(f"🎯 Model type: {model_type}")
    
    return result

def main():
    """Main function with full CLI support."""
    parser = argparse.ArgumentParser(
        description="ML Systems Cross-Reference Generator with Domain Adaptation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train a domain-adapted model
    python3 cross_referencing.py -t -d ../../contents/core/ -o ./t5-mlsys-domain-adapted
    
    # Generate with domain-adapted model  
    python3 cross_referencing.py -g -m ./t5-mlsys-domain-adapted -o cross_refs.json -d ../../contents/core/
    
    # Generate with base model (no training needed)
    python3 cross_referencing.py -g -m sentence-t5-base -o cross_refs.json -d ../../contents/core/
    
    # Train with custom parameters
    python3 cross_referencing.py -t -d ../../contents/core/ -o ./t5-mini-custom --base-model all-MiniLM-L6-v2 --epochs 3
        """)
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('-t', '--train', action='store_true', help='Training mode: Domain-adapt a base model')
    mode_group.add_argument('-g', '--generate', action='store_true', help='Generation mode: Generate cross-references')
    
    # Common arguments
    parser.add_argument('-d', '--dirs', nargs='+', required=True, 
                       help='Directories containing .qmd files')
    parser.add_argument('-o', '--output', required=True,
                       help='Output path (model directory for training, JSON file for generation)')
    
    # Training-specific arguments
    parser.add_argument('--base-model', default='sentence-t5-base',
                       help='Base model for training (default: sentence-t5-base)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs (default: 5)')
    
    # Generation-specific arguments
    parser.add_argument('-m', '--model', 
                       help='Model path (for generation): local path or HuggingFace name')
    
    # Optional arguments
    parser.add_argument('--exclude-chapters', nargs='*', default=[],
                        help='Space-separated list of chapter folder names to exclude (e.g., introduction conclusion)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--max-suggestions', type=int, default=5, 
                        help='Max cross-references per section (default: 5)')
    parser.add_argument('--similarity-threshold', type=float, default=0.65,
                        help='Minimum similarity for cross-references (default: 0.65)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.generate and not args.model:
        print("❌ Generation mode requires --model argument")
        return 1
    
    # Exclusions are now handled by argparse `exclude_chapters`
    
    # Validate directories
    for directory in args.dirs:
        if not Path(directory).exists():
            print(f"❌ Directory not found: {directory}")
            return 1
    
    try:
        if args.train:
            # Training mode
            success = train_model(
                directories=args.dirs,
                output_path=args.output,
                base_model=args.base_model,
                epochs=args.epochs,
                exclude_chapters=args.exclude_chapters
            )
            return 0 if success else 1
            
        elif args.generate:
            # Generation mode
            result = generate_cross_references(
                model_path=args.model,
                directories=args.dirs,
                output_file=args.output,
                exclude_chapters=args.exclude_chapters,
                max_suggestions=args.max_suggestions,
                similarity_threshold=args.similarity_threshold,
                verbose=args.verbose
            )
            
            if result and 'cross_references' in result and result['cross_references']:
                print(f"🎉 Success! Ready for Quarto injection. NOTE: lua/inject_xrefs.lua may need updates for the new JSON structure.")
                return 0
            else:
                print("⚠️  No cross-references generated")
                return 1
                
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 