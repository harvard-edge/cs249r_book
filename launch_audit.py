import json

chapters = [
    ("01", "01_boundary/01_boundary.qmd"),
    ("02", "02_body/02_body.qmd"),
    ("03", "03_brain/03_brain.qmd"),
    ("04", "04_nervous/04_nervous.qmd"),
    ("05", "05_data/05_data.qmd"),
    ("06", "06_training/06_training.qmd"),
    ("07", "07_evaluation/07_evaluation.qmd"),
    ("08", "08_perception/08_perception.qmd"),
    ("09", "09_memory/09_memory.qmd"),
    ("10", "10_intent/10_intent.qmd"),
    ("11", "11_planning/11_planning.qmd"),
    ("12", "12_enforcement/12_enforcement.qmd"),
    ("13", "13_placement/13_placement.qmd"),
    ("14", "14_intervention/14_intervention.qmd"),
    ("15", "15_verification/15_verification.qmd"),
    ("16", "16_release/16_release.qmd"),
    ("17", "17_frontier/17_frontier.qmd")
]

subagents = []
for ch_num, ch_path in chapters:
    subagents.append({
        "TypeName": "research",
        "Role": f"Ch {ch_num} Auditor",
        "Model": "flash",
        "Prompt": f"You are a pedagogical auditor. Your task is to READ THE FULL TEXT of Chapter {ch_num} in `/private/tmp/MLSysBook-dev-notation/books/vol4/{ch_path}`.\n1. Read the prose of every section fully.\n2. Check for Progressive Disclosure Violations: Are any concepts or acronyms used before being defined in the text?\n3. Check for Forward Connectors: Find any forward references to later chapters (e.g., 'as we will see in Section Y', 'in the next chapter', or forward `@sec-` references) and list them. (References to earlier chapters or appendices are fine).\nReport back to me via `send_message` with:\n- A list of any progressive disclosure violations or forward connectors you found.\n- If the chapter is perfectly clean, explicitly state 'CLEAN SIGN-OFF'."
    })

with open("audit_payload.json", "w") as f:
    json.dump({"Subagents": subagents}, f, indent=2)
