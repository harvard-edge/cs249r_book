import json

def generate_feedback(persona):
    feedback_profiles = {
        "Staff Engineer (End User)": [
            "The Cmd+Enter shortcut is great, but the text area is too small. When I write a full architectural diagnosis, I need more vertical space to see my math.",
            "The 'Knowledge Check' (L1/L2) feels too disconnected from the 'Terminal' UI. It feels like two different apps. Unify the visual language.",
            "I want to see the 'Key Equation' explicitly highlighted if I miss it."
        ],
        "UX Designer (Product)": [
            "The layout is a bit boxy. The 3-panel layout (Sidebar | Scenario | Terminal) is good, but the middle panel (Scenario) needs more breathing room (max-width).",
            "The Star Trap modal is a bit abrupt. It should feel like a 'Level Up' celebration before asking for the star.",
            "The typography hierarchy in the 'Ground Truth' section is slightly muddy. The 'Anti-Pattern' and 'Ground Truth' headers need to pop more."
        ],
        "Open Source Maintainer (Growth)": [
            "Where is the 'Edit on GitHub' button? If someone finds a typo in the math, they should be able to click one button to open a PR on the specific markdown file.",
            "The 'StaffML' logo is great, but we need a clear link back to 'MLSysBook' in the header so people know this is part of the textbook ecosystem."
        ]
    }
    return feedback_profiles.get(persona, ["Looking good."])

if __name__ == "__main__":
    print("--- Simulating Cross-Functional Feedback Loop ---")
    for persona in ["Staff Engineer (End User)", "UX Designer (Product)", "Open Source Maintainer (Growth)"]:
        print(f"\n[{persona}]")
        for comment in generate_feedback(persona):
            print(f" - {comment}")