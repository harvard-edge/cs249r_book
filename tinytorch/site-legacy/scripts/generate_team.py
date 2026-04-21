#!/usr/bin/env python3
"""
Generate team.md from .all-contributorsrc for TinyTorch site.

This script reads the .all-contributorsrc file and generates the team.md page
with all contributors automatically. Run this before building the site.

Usage:
    python3 scripts/generate_team.py
"""

import json
from pathlib import Path

# Contribution type to emoji mapping (matches all-contributors spec)
CONTRIBUTION_EMOJIS = {
    "bug": "🪲",
    "code": "🧑‍💻",
    "doc": "✍️",
    "design": "🎨",
    "ideas": "🧠",
    "review": "🔎",
    "test": "🧪",
    "tool": "🛠️",
    "content": "✍️",
    "maintenance": "🛠️",
}

# Special roles for staff members (not auto-generated)
STAFF_MEMBERS = {
    "profvjreddi": {
        "role": "🤓 Nerdy Professor · Harvard University",
        "bio": "Gordon McKay Professor of Electrical Engineering at Harvard. Passionate about creating the next generation of AI engineers.",
        "is_lead": True,
    },
    "AndreaMattiaGaravagno": {
        "role": "🧭 Tech Lead",
        "is_staff": True,
    },
    "kai4avaya": {
        "role": "🌐 Web Wizard",
        "is_staff": True,
    },
}

# Non-GitHub staff (manually added)
MANUAL_STAFF = [
    {
        "name": "Kari Janapareddi",
        "role": "👑 Chief of Staff",
        "avatar_url": "https://ui-avatars.com/api/?name=Kari+Janapareddi&background=f97316&color=fff&size=120",
        "profile": "#",
    },
]


def load_contributors(rc_path: Path) -> list[dict]:
    """Load contributors from .all-contributorsrc file."""
    if not rc_path.exists():
        print(f"Warning: {rc_path} not found")
        return []

    with open(rc_path) as f:
        data = json.load(f)

    return data.get("contributors", [])


def get_contribution_emojis(contributions: list[str]) -> str:
    """Convert contribution types to emoji string."""
    emojis = []
    for contrib in contributions:
        if contrib in CONTRIBUTION_EMOJIS:
            emojis.append(CONTRIBUTION_EMOJIS[contrib])
    return " ".join(emojis)


def generate_team_md(contributors: list[dict]) -> str:
    """Generate the team.md content from contributors list."""

    # Separate lead, staff, and regular contributors
    lead = None
    staff = []
    regular = []

    for c in contributors:
        login = c.get("login", "")
        if login in STAFF_MEMBERS:
            info = STAFF_MEMBERS[login]
            c["_role"] = info.get("role", "")
            c["_bio"] = info.get("bio", "")
            if info.get("is_lead"):
                lead = c
            elif info.get("is_staff"):
                staff.append(c)
            else:
                regular.append(c)
        else:
            regular.append(c)

    # Sort regular contributors by number of contributions (descending)
    regular.sort(key=lambda x: len(x.get("contributions", [])), reverse=True)

    # Build the markdown
    lines = []

    # Header
    lines.append("# Team")
    lines.append("")
    lines.append("**Meet the people building TinyTorch.**")
    lines.append("")
    lines.append("TinyTorch is built by a passionate community dedicated to making ML systems education accessible to everyone.")
    lines.append("")

    # CSS styles
    lines.append("```{raw} html")
    lines.append(CSS_STYLES)
    lines.append("")

    # Role legend
    lines.append('<div class="role-legend">')
    lines.append('  <span>🪲 Bug Hunter</span>')
    lines.append('  <span>🧑‍💻 Code Warrior</span>')
    lines.append('  <span>✍️ Documentation</span>')
    lines.append('  <span>🎨 Design</span>')
    lines.append('  <span>🧠 Ideas</span>')
    lines.append('  <span>🔎 Reviewer</span>')
    lines.append('  <span>🧪 Testing</span>')
    lines.append('  <span>🛠️ Tooling</span>')
    lines.append('</div>')
    lines.append("")

    # Team grid
    lines.append('<div class="team-grid">')

    # Lead
    if lead:
        avatar = lead.get("avatar_url", "")
        if "?" not in avatar:
            avatar += "?v=4&s=200"
        name = lead.get("name", lead.get("login", ""))
        profile = lead.get("profile", f"https://github.com/{lead.get('login', '')}")
        emojis = get_contribution_emojis(lead.get("contributions", []))

        lines.append('  <div class="team-lead">')
        lines.append(f'    <a href="{profile}">')
        lines.append(f'      <img src="{avatar}" alt="{name}" />')
        lines.append('    </a>')
        lines.append('    <div class="info">')
        lines.append(f'      <div class="name">{name}</div>')
        lines.append(f'      <div class="role">{lead.get("_role", "")}</div>')
        lines.append(f'      <div class="bio">{lead.get("_bio", "")}</div>')
        lines.append(f'      <div class="roles">{emojis}</div>')
        lines.append('    </div>')
        lines.append('  </div>')
        lines.append("")

    # Staff section
    if staff or MANUAL_STAFF:
        lines.append('  <div class="section-label">Community Staff</div>')
        lines.append("")
        lines.append('  <div class="staff-row">')

        for s in staff:
            avatar = s.get("avatar_url", "")
            if "?" not in avatar:
                avatar += "?v=4&s=120"
            name = s.get("name", s.get("login", ""))
            profile = s.get("profile", f"https://github.com/{s.get('login', '')}")
            role = s.get("_role", "")

            lines.append(f'    <a href="{profile}" class="team-staff">')
            lines.append(f'      <img src="{avatar}" alt="{name}" />')
            lines.append(f'      <div class="name">{name}</div>')
            lines.append(f'      <div class="role">{role}</div>')
            lines.append('    </a>')
            lines.append("")

        # Add manual staff
        for s in MANUAL_STAFF:
            lines.append(f'    <a href="{s["profile"]}" class="team-staff">')
            lines.append(f'      <img src="{s["avatar_url"]}" alt="{s["name"]}" />')
            lines.append(f'      <div class="name">{s["name"]}</div>')
            lines.append(f'      <div class="role">{s["role"]}</div>')
            lines.append('    </a>')
            lines.append("")

        lines.append('  </div>')
        lines.append("")

    # Contributors section
    if regular:
        lines.append('  <div class="section-label">Contributors</div>')

        for c in regular:
            avatar = c.get("avatar_url", "")
            if "?" not in avatar:
                avatar += "?v=4&s=160"
            name = c.get("name", c.get("login", ""))
            profile = c.get("profile", f"https://github.com/{c.get('login', '')}")
            emojis = get_contribution_emojis(c.get("contributions", []))

            lines.append(f'  <a href="{profile}" class="team-member">')
            lines.append(f'    <img src="{avatar}" alt="{name}" />')
            lines.append(f'    <span class="name">{name}</span>')
            lines.append(f'    <span class="roles">{emojis}</span>')
            lines.append('  </a>')

    lines.append('</div>')
    lines.append('```')
    lines.append("")

    # Footer
    lines.append("## Join the Team")
    lines.append("")
    lines.append("TinyTorch is open source and we welcome contributors of all experience levels!")
    lines.append("")
    lines.append("**Ways to get involved:**")
    lines.append("")
    lines.append("- 🪲 **Found a bug?** [Report it on GitHub](https://github.com/harvard-edge/cs249r_book/issues)")
    lines.append("- 💡 **Have an idea?** [Start a discussion](https://github.com/harvard-edge/cs249r_book/discussions)")
    lines.append("- 🧑‍💻 **Want to contribute code?** [See our contributing guide](https://github.com/harvard-edge/cs249r_book/blob/main/CONTRIBUTING.md)")
    lines.append("- ✍️ **Improve documentation?** Submit a pull request")
    lines.append("")
    lines.append('**Get recognized:** Comment on any issue or PR with `@all-contributors please add @username for bug, code, doc, or ideas`')
    lines.append("")
    lines.append("")
    lines.append("## Contact")
    lines.append("")
    lines.append("- **GitHub Issues**: [Report bugs or request features](https://github.com/harvard-edge/cs249r_book/issues)")
    lines.append("- **GitHub Discussions**: [Ask questions or share ideas](https://github.com/harvard-edge/cs249r_book/discussions)")
    lines.append("- **Project Lead**: [Prof. Vijay Janapa Reddi](https://scholar.harvard.edu/vijay-janapa-reddi) · Harvard University")
    lines.append("")

    return "\n".join(lines)


# CSS styles (kept separate for readability)
CSS_STYLES = """<style>
.team-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0;
}
.team-member {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-decoration: none;
  color: inherit;
  transition: transform 0.15s;
}
.team-member:hover {
  transform: translateY(-4px);
}
.team-member img {
  width: 80px;
  height: 80px;
  border-radius: 50%;
  border: 3px solid #e5e7eb;
  transition: border-color 0.2s;
}
.team-member:hover img {
  border-color: #f97316;
}
.team-member .name {
  font-size: 0.85rem;
  font-weight: 600;
  margin-top: 0.75rem;
  text-align: center;
}
.team-member .roles {
  font-size: 0.75rem;
  margin-top: 0.25rem;
  opacity: 0.8;
}
.team-lead {
  grid-column: 1 / -1;
  display: flex;
  align-items: center;
  gap: 1.5rem;
  padding: 1.5rem 2rem;
  background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
  border: 2px solid #fed7aa;
  border-radius: 1rem;
  margin-bottom: 1rem;
}
.team-lead img {
  width: 100px;
  height: 100px;
  border-radius: 50%;
  border: 4px solid #f97316;
  flex-shrink: 0;
}
.team-lead .info {
  flex: 1;
}
.team-lead .name {
  font-size: 1.35rem;
  font-weight: 700;
  color: #1f2937;
}
.team-lead .role {
  font-size: 1rem;
  color: #c2410c;
  margin-top: 0.25rem;
  font-weight: 500;
}
.team-lead .bio {
  font-size: 0.9rem;
  color: #57534e;
  margin-top: 0.5rem;
  line-height: 1.5;
}
.team-lead .roles {
  font-size: 0.85rem;
  margin-top: 0.5rem;
}
.role-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem 1.5rem;
  margin: 1.5rem 0;
  padding: 1rem 1.25rem;
  background: #f8fafc;
  border-radius: 0.75rem;
  font-size: 0.8rem;
}
.role-legend span {
  white-space: nowrap;
}
.staff-row {
  grid-column: 1 / -1;
  display: flex;
  justify-content: center;
  gap: 2.5rem;
  padding: 1rem 0;
}
.team-staff {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-decoration: none;
  color: inherit;
  transition: transform 0.15s;
}
.team-staff:hover {
  transform: translateY(-3px);
}
.team-staff img {
  width: 64px;
  height: 64px;
  border-radius: 50%;
  border: 2px solid #94a3b8;
  transition: border-color 0.15s;
}
.team-staff:hover img {
  border-color: #f97316;
}
.team-staff .name {
  font-size: 0.85rem;
  font-weight: 600;
  margin-top: 0.5rem;
  text-align: center;
}
.team-staff .role {
  font-size: 0.7rem;
  color: #64748b;
  margin-top: 0.125rem;
  text-align: center;
}
.section-label {
  grid-column: 1 / -1;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #94a3b8;
  margin-top: 1rem;
  margin-bottom: -0.5rem;
}
@media (max-width: 600px) {
  .team-lead {
    flex-direction: column;
    text-align: center;
    padding: 1.25rem;
  }
  .team-lead img {
    width: 90px;
    height: 90px;
  }
  .team-staff {
    padding: 0.75rem 1rem;
  }
}
</style>"""


def main():
    # Find the .all-contributorsrc file
    script_dir = Path(__file__).parent
    site_dir = script_dir.parent
    tinytorch_dir = site_dir.parent
    rc_path = tinytorch_dir / ".all-contributorsrc"

    # Load contributors
    contributors = load_contributors(rc_path)
    if not contributors:
        print("No contributors found!")
        return 1

    print(f"Found {len(contributors)} contributors")

    # Generate team.md
    content = generate_team_md(contributors)

    # Write to team.md
    team_path = site_dir / "team.md"
    with open(team_path, "w") as f:
        f.write(content)

    print(f"Generated {team_path}")
    return 0


if __name__ == "__main__":
    exit(main())
