# generate_appendix.py
# Generates a Quarto Markdown appendix of all physical and economic assumptions
# used in the book.

import inspect
import pandas as pd
try:
    from . import constants
except ImportError:
    import constants

OUTPUT_FILE = "../contents/vol1/backmatter/appendix_assumptions.qmd"

def main():
    records = []
    
    for name, value in inspect.getmembers(constants):
        if name.startswith("_"): continue
        if inspect.ismodule(value): continue
        if inspect.isfunction(value): continue
        if inspect.isclass(value): continue
        if name in ["ureg", "Q_", "pint"]: continue
        
        # Determine Type and Value
        if hasattr(value, "units"):
            val_str = f"{value.magnitude:g}"
            unit_str = f"{value.units:~P}" # Pretty print units
        else:
            val_str = str(value)
            unit_str = "-"
            
        records.append({
            "Constant": name,
            "Value": val_str,
            "Unit": unit_str
        })
    
    df = pd.DataFrame(records)
    
    # Generate Markdown
    md = f"""---
title: "Appendix: System Assumptions"
---

This appendix lists the physical constants, hardware specifications, and economic assumptions used to calculate the "Napkin Math" throughout this book. These values are defined in the book's source code (`mlsys/constants.py`) and ensure that all derivations are consistent.

::: {{.column-page}}
{df.to_markdown(index=False)}
:::
"""
    
    print(md)
    # In a real workflow, we would write this to OUTPUT_FILE
    with open(OUTPUT_FILE, "w") as f:
        f.write(md)

if __name__ == "__main__":
    main()
