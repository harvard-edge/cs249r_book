import os

def create_svg():
    svg_content = """<?xml version="1.0" encoding="UTF-8" ?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 400" width="1200" height="400">
    <rect width="1200" height="400" fill="#ffffff" />
    
    <!-- Title -->
    <text x="600" y="40" font-family="Arial, sans-serif" font-size="24" font-weight="bold" text-anchor="middle" fill="#2D3748">The End-to-End Cognitive Flow of Physical AI</text>
    
    <!-- Flow Boxes -->
    <g font-family="Arial, sans-serif" font-size="16" font-weight="bold" text-anchor="middle">
        
        <!-- Physical Sensors -->
        <rect x="30" y="150" width="130" height="80" fill="#E2E8F0" stroke="#2D3748" stroke-width="2"/>
        <text x="95" y="195" fill="#2D3748">Sensors</text>

        <!-- Arrow -->
        <path fill="none" d="M 160 190 L 190 190" stroke="#2D3748" stroke-width="3" marker-end="url(#arrow)" />

        <!-- Ingestion -->
        <rect x="200" y="150" width="130" height="80" fill="#1F407A" stroke="#2D3748" stroke-width="2"/>
        <text x="265" y="195" fill="#ffffff">Ingestion</text>
        <text x="265" y="215" font-size="12" fill="#E2E8F0">(Tokenization)</text>

        <!-- Arrow -->
        <path fill="none" d="M 330 190 L 360 190" stroke="#2D3748" stroke-width="3" marker-end="url(#arrow)" />

        <!-- Perception -->
        <rect x="370" y="150" width="130" height="80" fill="#1F407A" stroke="#2D3748" stroke-width="2"/>
        <text x="435" y="195" fill="#ffffff">Perception</text>
        <text x="435" y="215" font-size="12" fill="#E2E8F0">(Spatial/Afford.)</text>

        <!-- Arrow -->
        <path fill="none" d="M 500 190 L 530 190" stroke="#2D3748" stroke-width="3" marker-end="url(#arrow)" />

        <!-- Memory -->
        <rect x="540" y="150" width="130" height="80" fill="#1F407A" stroke="#2D3748" stroke-width="2"/>
        <text x="605" y="195" fill="#ffffff">Memory</text>
        <text x="605" y="215" font-size="12" fill="#E2E8F0">(World Model)</text>

        <!-- Arrow -->
        <path fill="none" d="M 670 190 L 700 190" stroke="#2D3748" stroke-width="3" marker-end="url(#arrow)" />

        <!-- Intent -->
        <rect x="710" y="150" width="130" height="80" fill="#1F407A" stroke="#2D3748" stroke-width="2"/>
        <text x="775" y="195" fill="#ffffff">Intent</text>
        <text x="775" y="215" font-size="12" fill="#E2E8F0">(Goal Leases)</text>

        <!-- Arrow -->
        <path fill="none" d="M 840 190 L 870 190" stroke="#2D3748" stroke-width="3" marker-end="url(#arrow)" />

        <!-- Planning -->
        <rect x="880" y="150" width="130" height="80" fill="#1F407A" stroke="#2D3748" stroke-width="2"/>
        <text x="945" y="195" fill="#ffffff">Planning</text>
        <text x="945" y="215" font-size="12" fill="#E2E8F0">(Action Chunk.)</text>

        <!-- Arrow to Boundary -->
        <path fill="none" d="M 1010 190 L 1040 190" stroke="#2D3748" stroke-width="3" marker-end="url(#arrow)" />

        <!-- Proposal-Permission Boundary Line -->
        <line x1="1060" y1="100" x2="1060" y2="280" stroke="#A51C30" stroke-width="4" stroke-dasharray="8,8" />
        <text x="1060" y="90" fill="#A51C30" font-size="14">Proposal / Permission</text>
        <text x="1060" y="300" fill="#A51C30" font-size="14">Boundary</text>

        <!-- Reflexes (Downstream) -->
        <rect x="1080" y="150" width="90" height="80" fill="#E2E8F0" stroke="#A51C30" stroke-width="2"/>
        <text x="1125" y="195" fill="#2D3748" font-size="14">Reflexes</text>
        <text x="1125" y="215" font-size="12" fill="#2D3748">(Real-time)</text>
    </g>

    <!-- Defs for arrowheads -->
    <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#2D3748" />
        </marker>
    </defs>
</svg>
"""
    
    out_dir = "svg"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'fig03_cognitive_pipeline_horizontal.svg')
    with open(out_path, 'w') as f:
        f.write(svg_content)
    print(f"Generated {out_path}")

if __name__ == "__main__":
    create_svg()
