def generate_feedback(persona):
    if persona == "UX Expert (Logo & Branding)":
        return [
            "The current logo `> StaffML _` is strong, but the spacing feels slightly disjointed. The `>` chevron and the `_` cursor need to feel like they belong to the word, not just floating next to it.",
            "In modern dev tool branding (like Vercel, Linear, or Raycast), the 'mark' (the icon) usually stands alone or is very tightly integrated. Right now, the `>` and the `_` are competing for attention.",
            "If the goal is to expand to `StaffX` (StaffSys, StaffSec), the word 'Staff' should be the anchor (heaviest weight), and 'ML' should be the variable suffix (lighter weight or different color).",
            "Try dropping the `_` cursor. The `>` chevron alone is a stronger, cleaner, more scalable favicon for a tab or an app icon."
        ]
    return []

if __name__ == "__main__":
    print("--- UX Expert Logo Critique ---")
    for comment in generate_feedback("UX Expert (Logo & Branding)"):
        print(f" - {comment}")