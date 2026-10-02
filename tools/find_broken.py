import yaml
with open("books/shared/_includes/notation_registry.yml") as f:
    data = yaml.safe_load(f)

for var in data["variables"]:
    latex = var["latex"]
    if latex.count('{') != latex.count('}'):
        print(f"BROKEN: {var['macro']} -> {latex}")
