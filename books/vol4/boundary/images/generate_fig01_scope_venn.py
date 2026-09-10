import os
import math

svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 800" width="800" height="800">
  <style>
    .title-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 24px; font-weight: bold; fill: #1F407A; text-anchor: middle; }
    .label-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 18px; font-weight: bold; fill: #1F407A; text-anchor: middle; }
    .desc-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 14px; fill: #2D3748; text-anchor: middle; }
    .center-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 28px; font-weight: bold; fill: #ffffff; text-anchor: middle; }
  </style>

  <!-- Background -->
  <rect width="800" height="800" fill="#ffffff" />

  <!-- Circles with textbook flat colors -->
  <!-- Top: Learned Statistical Model -->
  <!-- Left: Consequential Physical Feedback -->
  <!-- Right: Delegated Physical Authority -->
  <g style="mix-blend-mode: multiply;">
    <!-- Top Circle -->
    <circle cx="400" cy="300" r="220" fill="#E2E8F0" stroke="#1F407A" stroke-width="2" />
    <!-- Bottom Left Circle -->
    <circle cx="270" cy="525" r="220" fill="#FFF0F0" stroke="#A51C30" stroke-width="2" />
    <!-- Bottom Right Circle -->
    <circle cx="530" cy="525" r="220" fill="#F0FDF4" stroke="#15803D" stroke-width="2" />
  </g>

  <!-- Labels -->
  <g>
    <!-- Top Circle Label -->
    <text x="400" y="160" class="title-text">Learned Statistical Model</text>
    <text x="400" y="180" class="desc-text">(Unspecifiable Policy)</text>

    <!-- Bottom Left Label -->
    <text x="160" y="660" class="title-text" fill="#A51C30">Consequential</text>
    <text x="160" y="685" class="title-text" fill="#A51C30">Physical Feedback</text>

    <!-- Bottom Right Label -->
    <text x="640" y="660" class="title-text" fill="#15803D">Delegated</text>
    <text x="640" y="685" class="title-text" fill="#15803D">Physical Authority</text>

    <!-- Intersections -->
    <!-- Top + Bottom Left -->
    <text x="250" y="380" class="label-text">Advisory</text>
    <text x="250" y="400" class="label-text">Decision Support</text>
    <text x="250" y="420" class="desc-text">(No Authority)</text>

    <!-- Top + Bottom Right -->
    <text x="550" y="380" class="label-text">Digital</text>
    <text x="550" y="400" class="label-text">Autonomous Systems</text>
    <text x="550" y="420" class="desc-text">(No Phys. Feedback)</text>

    <!-- Bottom Left + Bottom Right -->
    <text x="400" y="650" class="label-text">Classical</text>
    <text x="400" y="670" class="label-text">Control Theory</text>
    <text x="400" y="690" class="desc-text">(Analytical Transfer Fn)</text>

    <!-- Center -->
    <text x="400" y="450" class="center-text" fill="#1F407A">Physical AI</text>
  </g>
</svg>
"""

out_dir = "/Users/VJ/GitHub/MLSysBook-vol4-physical/books/vol4/boundary/images/svg"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "fig01_scope_venn.svg")
with open(out_path, "w") as f:
    f.write(svg_content)
print(f"Generated {out_path}")
