import json
import re
import sys
from pathlib import Path

# Import taxonomy for competency_area mapping
sys.path.insert(0, str(Path(__file__).parent / "engine"))
from taxonomy import normalize_tag, get_area_for_tag, ALL_TAGS, _NORMALIZE_MAP_LOWER


def _resolve_competency_area(raw_topic: str) -> str:
    """Resolve competency area for a topic tag, including compound tags.

    First tries exact match via normalize_tag + get_area_for_tag.
    If that fails, splits the compound tag on hyphens and tries
    progressively shorter suffixes and prefixes to find a match.
    """
    canonical = normalize_tag(raw_topic)
    area = get_area_for_tag(canonical)
    if area:
        return area

    # Try sub-segments of compound tags like "federated-learning-economics"
    parts = raw_topic.lower().strip().split('-')
    # Try progressively shorter prefixes and suffixes
    for length in range(len(parts) - 1, 0, -1):
        for start in range(len(parts) - length + 1):
            candidate = '-'.join(parts[start:start + length])
            normalized = normalize_tag(candidate)
            area = get_area_for_tag(normalized)
            if area:
                return area

    return ""


def parse_md_to_questions(file_path, track, scope):
    content = Path(file_path).read_text(encoding='utf-8')
    blocks = content.split('<details>\n<summary><b><img')

    questions = []
    for i, block in enumerate(blocks):
        if i == 0: continue

        try:
            full_block = '<summary><b><img' + block

            # --- Level extraction ---
            # Use regex to pull the level from the badge URL, handling:
            #   Level-L6+_Principal  (literal +)
            #   Level-L6%2B_Staff    (URL-encoded +)
            #   Level-L3_Senior      (normal levels)
            level = "Unknown"
            level_match = re.search(r'badge/Level-(L\d)%2B', full_block)
            if level_match:
                level = level_match.group(1) + '+'
            else:
                level_match = re.search(r'badge/Level-(L\d\+?)', full_block)
                if level_match:
                    level = level_match.group(1)

            # Normalize: L6 (without +) is always L6+ in our taxonomy
            if level == 'L6':
                level = 'L6+'

            # Safety: strip any residual HTML that might have leaked in
            if '<' in level or '>' in level:
                level = re.sub(r'<[^>]+>', '', level).strip()

            title = "Unknown"
            if '</b>' in full_block:
                title_part = full_block.split('</b>')[0].split('>')[-1].strip()
                title = title_part

            topic = "Unknown"
            if '<code>' in full_block:
                topic = full_block.split('<code>')[1].split('</code>')[0]

            # --- Scenario extraction ---
            # Markdown format: **Interviewer:** "question text"
            # Strip surrounding quotes and markdown prefix characters.
            scenario = ""
            if '**Interviewer:**' in full_block:
                scenario_part = full_block.split('**Interviewer:**')[1]
                if '<details>' in scenario_part:
                    scenario = scenario_part.split('<details>')[0].split('\n\n')[0]
                else:
                    scenario = scenario_part.split('\n\n')[0]
                # Strip leading/trailing whitespace, then quotes and markdown chars
                scenario = scenario.strip()
                scenario = scenario.strip('"').strip()

            # --- competency_area from taxonomy ---
            competency_area = _resolve_competency_area(topic) if topic != "Unknown" else ""

            details = {}
            if '**Common Mistake:**' in full_block:
                details['common_mistake'] = full_block.split('**Common Mistake:**')[1].split('\n')[0].strip()

            if '**Realistic Solution:**' in full_block:
                sol_part = full_block.split('**Realistic Solution:**')[1].strip()
                if '\n\n' in sol_part:
                    details['realistic_solution'] = sol_part.split('\n\n')[0].strip()
                elif '> **' in sol_part:
                    details['realistic_solution'] = sol_part.split('> **')[0].strip()
                else:
                    details['realistic_solution'] = sol_part

            if '**Napkin Math:**' in full_block:
                nm_raw = full_block.split('**Napkin Math:**')[1]
                # Handle both single-line and multi-line formats
                first_line = nm_raw.split('\n')[0].strip()
                if first_line:
                    details['napkin_math'] = first_line
                else:
                    # Multi-line: collect blockquote lines until empty line or non-blockquote
                    nm_lines = []
                    for line in nm_raw.split('\n')[1:]:
                        stripped = line.strip()
                        if stripped.startswith('> ') or stripped.startswith('>-'):
                            nm_lines.append(stripped.lstrip('> ').strip())
                        elif stripped == '>' or stripped == '':
                            if nm_lines:
                                break
                        elif stripped.startswith('-'):
                            nm_lines.append(stripped.lstrip('- ').strip())
                        else:
                            break
                    details['napkin_math'] = ' | '.join(nm_lines) if nm_lines else ''

            # Legacy markdown import: the old '📖 Deep Dive:' single-link format
            # maps to a one-element resources list under the new schema.
            if '📖 **Deep Dive:**' in full_block:
                link_part = full_block.split('📖 **Deep Dive:**')[1].split('\n')[0]
                if '[' in link_part and '](' in link_part:
                    name = link_part.split('[')[1].split(']')[0]
                    url = link_part.split('(')[1].split(')')[0]
                    if name and url and url.startswith('https://'):
                        details.setdefault('resources', []).append({'name': name, 'url': url})

            if '> **Options:**' in full_block:
                options_block = full_block.split('> **Options:**')[1].split('</details>')[0]
                options_lines = [line.strip() for line in options_block.split('\n') if line.strip().startswith('> [')]

                parsed_options = []
                correct_index = -1

                for idx, line in enumerate(options_lines):
                    if '[x]' in line.lower():
                        correct_index = idx
                    clean_text = re.sub(r'^>\s*\[[xX\s]\]\s*', '', line).strip()
                    if clean_text:
                        parsed_options.append(clean_text)

                if parsed_options:
                    details['options'] = parsed_options
                    details['correct_index'] = correct_index

            questions.append({
                "id": f"{track}-{topic}-{title.lower().replace(' ', '-')[:40]}-{len(questions)}",
                "track": track,
                "scope": scope,
                "level": level,
                "title": title,
                "topic": topic,
                "competency_area": competency_area,
                "scenario": scenario,
                "details": details
            })
        except Exception as e:
            continue

    return questions

def run():
    base_path = Path(__file__).parent
    tracks = ["cloud", "edge", "mobile", "tinyml"]
    corpus = []

    print("🚀 Building MLSys Interview Corpus...")

    foundations_file = base_path / "foundations.md"
    if foundations_file.exists():
        qs = parse_md_to_questions(foundations_file, "global", "Foundations")
        print(f"  - foundations.md: {len(qs)} questions")
        corpus.extend(qs)

    for track in tracks:
        track_dir = base_path / track
        if not track_dir.exists(): continue

        for md_file in track_dir.glob("*.md"):
            if md_file.name == "README.md": continue

            scope_name = md_file.stem.replace('_', ' ').title()
            scope_name = re.sub(r'^\d+\s+', '', scope_name)

            qs = parse_md_to_questions(md_file, track, scope_name)
            print(f"  - {track}/{md_file.name}: {len(qs)} questions")
            corpus.extend(qs)

    output_file = base_path / "corpus.json"
    output_file.write_text(json.dumps(corpus, indent=2), encoding='utf-8')
    print(f"\n✅ Done! Found {len(corpus)} questions. Saved to {output_file}")

if __name__ == "__main__":
    run()
