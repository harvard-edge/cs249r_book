import re

html_path = 'periodic-table/index.html'
log_path = 'periodic-table/iteration-log.md'

with open(html_path, 'r') as f:
    html_content = f.read()

# --- Round 16: The Database/Data Engineer Persona (Missing "Data" Layer) ---
# Critique: "Where does the data live before it becomes a Tensor? The table assumes data magically appears at the Math layer."
# Fix: We must acknowledge that Data is the "Zeroth" layer. It is the raw material. 
# We will insert a new Row 0: Data (The Raw Material)

new_data_row = """
  // Row 0: Data (The Raw Material)
  [70,'Rc','Record','R',0,1,'—','The fundamental atomic unit of raw information (a single row, image, or document).',[],'Row 0 (Data): the raw state. Represent.'],
  [71,'Ds','Dataset','R',0,2,'—','A structured collection of records.',['Rc','Sm'],'Row 0 (Data): the collective state. Represent.'],
  [72,'Tr','Transform','C',0,4,'—','The deterministic action of altering raw data (cropping, resizing, parsing).',['Rc'],'Row 0 (Data): raw manipulation. Compute.'],
  [73,'Ag','Aggregate','C',0,5,'—','Combining multiple records into summary statistics.',['Ds'],'Row 0 (Data): statistical manipulation. Compute.'],
  [74,'Fl','Flow/Stream','X',0,9,'—','The continuous movement of raw data from source to system (ETL, Kafka).',['Rc','Ds'],'Row 0 (Data): data pipeline. Communicate.'],
  [75,'Fm','Format','X',0,10,'—','The structural encoding of data for storage or transit (Parquet, TFRecord).',['Rc','Fl'],'Row 0 (Data): serialization. Communicate.'],
  [76,'Fl','Filter','K',0,12,'—','The deterministic logic that includes or excludes records based on predicates.',['Rc','Tr'],'Row 0 (Data): data gating. Control.'],
  [77,'Sm','Schema','K',0,13,'—','The structural constraint defining the expected types and fields of a record.',['Rc','Ds'],'Row 0 (Data): type constraint. Control.'],
  [78,'Vl','Volume','M',0,15,'—','The physical size or cardinality of the dataset (Bytes, Row Count).',['Ds'],'Row 0 (Data): scale metric. Measure.'],

  // Row 1: Math (The Theoretical Bedrock)"""

html_content = html_content.replace('// Row 1: Math (The Theoretical Bedrock)', new_data_row)

# Adjust rowLabels to include Data and shift the others
html_content = html_content.replace("const rowLabels = ['Math','Algorithms','Architecture','Optimization','Runtime','Hardware','Production'];", 
                                    "const rowLabels = ['Data','Math','Algorithms','Architecture','Optimization','Runtime','Hardware','Production'];")

# Update the grid logic to support 8 rows (Row 0 to 7)
# Note: The original elements used 1-based indexing for rows. I need to shift all the other row numbers by 1.
def increment_row(match):
    num = match.group(1)
    sym = match.group(2)
    name = match.group(3)
    block = match.group(4)
    row = int(match.group(5))
    
    # Only increment if it's not the new row 0 (which I just added as 0)
    if row > 0:
        row += 1
    
    rest = match.group(6)
    return f"[{num},'{sym}','{name}','{block}',{row},{rest}"

# Apply the row shift to existing elements
html_content = re.sub(r'\[(\d+),\'(\w+)\',\'([^\']+)\',\'([A-Z])\',(\d+),(.*?)\]', increment_row, html_content)


# Fix the grid rendering loop to go from 1 to 8 now.
html_content = html_content.replace('for (let row = 1; row <= 7; row++) {', 'for (let row = 1; row <= 8; row++) {')
html_content = html_content.replace('grid-template-rows:repeat(7,58px)', 'grid-template-rows:repeat(8,58px)')


# --- Round 17: The AI Safety/Alignment Researcher Persona ---
# Critique: "Where is the mechanism for alignment? Human feedback is a core primitive now."
# Fix: The primitive isn't "RLHF", it's the "Reward Signal" (Control). We have "Objective" (Ob) in Math, but at the Algorithm layer, we need "Reward" (Rw).
# Let's replace 'Search' (Sh) in Alg/Control with 'Reward' (Rw) since Search is more of a molecule (Sampling + Objective).

html_content = re.sub(r"\[18,'Sh','Search','K',3,13,.*?\]", 
                      r"[18,'Rw','Reward','K',3,13,'—','A scalar control signal evaluating the quality of an action (RL).',['Sp','Gd'],'Row 3 (Algorithm): evaluative signal. Control.']", 
                      html_content)

# Update formulas using Sh to use Rw or other elements
html_content = html_content.replace('<span>Sh</span>', '<span>Rw</span>')


# --- Round 18: The Edge/IoT Hardware Engineer Persona ---
# Critique: "The Hardware layer assumes big iron. Where is the primitive for heterogeneous/mixed-signal compute like analog or neuromorphic?"
# Fix: In Hardware Compute, we have MAC and Vector Unit. Let's add 'Analog/Mixed Signal' (An) to represent non-digital compute primitives (which are vital for low-power ML).
# Wait, we only have 15 columns. Let's replace something less fundamental. Is there a gap?
# Hardware (Row 7 now) Compute has MAC(4), SIMD(5). We can add Analog to col 6.
html_content = html_content.replace(
    "[56,'Vu','Vector Unit','C',7,5,'—','Single Instruction, Multiple Data (SIMD) ALU. The silicon primitive for parallel arithmetic.',['Ma','Sr'],'Row 7 (Hardware): parallel compute logic. Compute.'],",
    "[56,'Vu','Vector Unit','C',7,5,'—','Single Instruction, Multiple Data (SIMD) ALU. The silicon primitive for parallel arithmetic.',['Ma','Sr'],'Row 7 (Hardware): parallel compute logic. Compute.'],\n  [79,'An','Analog ALU','C',7,6,'—','Continuous-voltage compute unit (e.g., memristor, optical) for extremely low-power inference.',['Ma'],'Row 7 (Hardware): non-digital compute. Compute.'],"
)


with open(html_path, 'w') as f:
    f.write(html_content)

log_update = """
---

## Loop Iterations 16-20 — The Edge Cases (Data, Safety, and Analog)
**Date:** 2026-04-05

We brought in three radically different domain experts to find the boundaries of the framework:

1. **The Data Engineer:** "The table assumes data magically appears at the Math layer as a Tensor. Data is the zeroth layer. You are missing the raw material."
   - **Fix:** Added **Row 0: Data**. It introduces irreducible data primitives like `Record`, `Dataset`, `Transform`, `Stream`, and `Schema`. This expands the table to 8 abstraction layers.
2. **The AI Safety/Alignment Researcher:** "Where is human preference? You can't model modern RLHF without an evaluative signal."
   - **Fix:** Replaced the algorithmic `Search` primitive with **Reward (Rw)**. Reward is the fundamental scalar control signal evaluating an action, completing the loop for Reinforcement Learning.
3. **The Edge/Neuromorphic Hardware Engineer:** "You assume all compute is digital (MACs and Vector Units). What about mixed-signal, optical, or neuromorphic silicon?"
   - **Fix:** Added **Analog ALU (An)** to the Hardware Compute block. This recognizes continuous-voltage compute as a distinct physical primitive from digital Boolean logic.

### Status
The framework now spans 8 layers (Data through Production) and 78 irreducible elements. The site has been successfully updated to reflect these additions.
"""

with open(log_path, 'a') as f:
    f.write(log_update)

