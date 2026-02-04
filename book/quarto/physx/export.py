# export.py
# Exports Python constants to JSON for use in Quarto OJS (Web) blocks.
# Usage: python3 -m book.quarto.physx.export

import json
import os
import inspect
from . import constants

OUTPUT_PATH = "book/quarto/assets/constants.json"

def main():
    data = {}
    
    # Iterate over all variables in constants.py
    for name, value in inspect.getmembers(constants):
        if name.startswith("_"): continue
        if inspect.ismodule(value): continue
        if inspect.isfunction(value): continue
        if inspect.isclass(value): continue
        
        # If it's a Pint Quantity, export magnitude and unit string
        if hasattr(value, "magnitude") and hasattr(value, "units"):
            data[name] = {
                "value": value.magnitude,
                "unit": str(value.units)
            }
        # If it's a simple number (float/int), export as-is
        elif isinstance(value, (int, float)):
            data[name] = {
                "value": value,
                "unit": "dimensionless"
            }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Exported {len(data)} constants to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
