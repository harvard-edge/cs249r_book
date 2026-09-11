import os

svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 300">
    <!-- Clean flat 2D textbook engineering style -->
    <!-- Grid/Background -->
    <rect width="900" height="300" fill="#ffffff" />
    
    <!-- Title -->
    <text x="450" y="30" font-family="sans-serif" font-size="18" font-weight="bold" fill="#1F407A" text-anchor="middle">Policy Performance Ranking Inversion</text>
    
    <!-- Group 1: Flow Regulation -->
    <g transform="translate(50, 60)">
        <rect width="240" height="200" fill="#E2E8F0" rx="4" />
        <text x="120" y="25" font-family="sans-serif" font-size="14" font-weight="bold" fill="#2D3748" text-anchor="middle">1. Flow Regulation</text>
        <text x="120" y="50" font-family="sans-serif" font-size="12" fill="#2D3748" text-anchor="middle">Sanitize Recovery Runs</text>
        
        <rect x="20" y="70" width="90" height="40" fill="#1F407A" rx="2" />
        <text x="65" y="90" font-family="sans-serif" font-size="10" fill="white" text-anchor="middle">Train Metric:</text>
        <text x="65" y="105" font-family="sans-serif" font-size="10" fill="white" text-anchor="middle">Val Loss: 0.011 (4x drop)</text>
        
        <rect x="130" y="70" width="90" height="40" fill="#A51C30" rx="2" />
        <text x="175" y="90" font-family="sans-serif" font-size="10" fill="white" text-anchor="middle">Hardware:</text>
        <text x="175" y="105" font-family="sans-serif" font-size="10" fill="white" text-anchor="middle">Recovery: 0%</text>
        
        <path fill="none" d="M 65 110 L 65 160" stroke="#1F407A" stroke-width="2" marker-end="url(#arrow-blue)"/>
        <path fill="none" d="M 175 110 L 175 160" stroke="#A51C30" stroke-width="2" marker-end="url(#arrow-red)"/>
        <text x="65" y="175" font-family="sans-serif" font-size="12" fill="#1F407A" font-weight="bold" text-anchor="middle">Success</text>
        <text x="175" y="175" font-family="sans-serif" font-size="12" fill="#A51C30" font-weight="bold" text-anchor="middle">Failure</text>
    </g>

    <!-- Group 2: Surface Touchdown -->
    <g transform="translate(330, 60)">
        <rect width="240" height="200" fill="#E2E8F0" rx="4" />
        <text x="120" y="25" font-family="sans-serif" font-size="14" font-weight="bold" fill="#2D3748" text-anchor="middle">2. Surface Touchdown</text>
        <text x="120" y="50" font-family="sans-serif" font-size="12" fill="#2D3748" text-anchor="middle">Unbraked Plunge</text>
        
        <rect x="20" y="70" width="90" height="40" fill="#1F407A" rx="2" />
        <text x="65" y="90" font-family="sans-serif" font-size="10" fill="white" text-anchor="middle">Train Metric:</text>
        <text x="65" y="105" font-family="sans-serif" font-size="10" fill="white" text-anchor="middle">Sim Return: +28%</text>
        
        <rect x="130" y="70" width="90" height="40" fill="#A51C30" rx="2" />
        <text x="175" y="90" font-family="sans-serif" font-size="10" fill="white" text-anchor="middle">Hardware:</text>
        <text x="175" y="105" font-family="sans-serif" font-size="10" fill="white" text-anchor="middle">Peak Force: 68N (Break)</text>
        
        <path fill="none" d="M 65 110 L 65 160" stroke="#1F407A" stroke-width="2" marker-end="url(#arrow-blue)"/>
        <path fill="none" d="M 175 110 L 175 160" stroke="#A51C30" stroke-width="2" marker-end="url(#arrow-red)"/>
        <text x="65" y="175" font-family="sans-serif" font-size="12" fill="#1F407A" font-weight="bold" text-anchor="middle">Success</text>
        <text x="175" y="175" font-family="sans-serif" font-size="12" fill="#A51C30" font-weight="bold" text-anchor="middle">Failure</text>
    </g>
    
    <!-- Group 3: Regime Transition -->
    <g transform="translate(610, 60)">
        <rect width="240" height="200" fill="#E2E8F0" rx="4" />
        <text x="120" y="25" font-family="sans-serif" font-size="14" font-weight="bold" fill="#2D3748" text-anchor="middle">3. Regime Transition</text>
        <text x="120" y="50" font-family="sans-serif" font-size="12" fill="#2D3748" text-anchor="middle">Simulation RL</text>
        
        <rect x="20" y="70" width="90" height="40" fill="#1F407A" rx="2" />
        <text x="65" y="90" font-family="sans-serif" font-size="10" fill="white" text-anchor="middle">Train Metric:</text>
        <text x="65" y="105" font-family="sans-serif" font-size="10" fill="white" text-anchor="middle">Sim Task: 100%</text>
        
        <rect x="130" y="70" width="90" height="40" fill="#A51C30" rx="2" />
        <text x="175" y="90" font-family="sans-serif" font-size="10" fill="white" text-anchor="middle">Hardware:</text>
        <text x="175" y="105" font-family="sans-serif" font-size="10" fill="white" text-anchor="middle">Chatter/Trip: 12%</text>
        
        <path fill="none" d="M 65 110 L 65 160" stroke="#1F407A" stroke-width="2" marker-end="url(#arrow-blue)"/>
        <path fill="none" d="M 175 110 L 175 160" stroke="#A51C30" stroke-width="2" marker-end="url(#arrow-red)"/>
        <text x="65" y="175" font-family="sans-serif" font-size="12" fill="#1F407A" font-weight="bold" text-anchor="middle">Success</text>
        <text x="175" y="175" font-family="sans-serif" font-size="12" fill="#A51C30" font-weight="bold" text-anchor="middle">Failure</text>
    </g>

    <!-- Definitions -->
    <defs>
        <marker id="arrow-blue" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#1F407A" />
        </marker>
        <marker id="arrow-red" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#A51C30" />
        </marker>
    </defs>
</svg>
"""

with open("svg/fig06_policy_ranking_inversion.svg", "w") as f:
    f.write(svg_content)
