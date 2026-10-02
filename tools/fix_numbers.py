import yaml
import re

with open("books/shared/_partials/notation_registry.yml") as f:
    data = yaml.safe_load(f)

num_map = {'0': 'Zero', '1': 'One', '2': 'Two', '3': 'Three', '4': 'Four', '5': 'Five', '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine'}

for var in data["variables"]:
    macro = var["macro"]
    new_macro = ""
    for char in macro:
        if char in num_map:
            new_macro += num_map[char]
        else:
            new_macro += char
    var["macro"] = new_macro
    var["id"] = new_macro.replace("\\", "")

with open("books/shared/_partials/notation_registry.yml", "w") as f:
    yaml.dump(data, f, sort_keys=False, allow_unicode=True)
