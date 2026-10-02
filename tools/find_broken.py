import yaml
with open("books/shared/_partials/notation_registry_clean.yml") as f:
    data = yaml.safe_load(f)

for var in data["variables"]:
    latex = var["latex"]
    if latex.count('{') != latex.count('}'):
        print(f"BROKEN: {var['macro']} -> {latex}")
