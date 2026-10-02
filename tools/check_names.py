import yaml
with open("books/shared/_includes/notation_registry.yml") as f:
    data = yaml.safe_load(f)
for var in data["variables"]:
    if len(var["definition"].split()) > 4 or ";" in var["definition"]:
        print(f"{var['latex']:<20} | {var['definition']}")
