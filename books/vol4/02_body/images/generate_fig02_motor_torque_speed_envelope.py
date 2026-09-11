import os

svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 600" width="100%" height="100%">
  <defs>
    <style>
      .title { font-family: sans-serif; font-size: 20px; font-weight: bold; fill: #1F407A; }
      .label { font-family: sans-serif; font-size: 14px; fill: #2D3748; }
      .axis { stroke: #2D3748; stroke-width: 2; }
      .grid { stroke: #E2E8F0; stroke-width: 1; }
      .region-s1 { fill: #E2E8F0; fill-opacity: 0.6; stroke: #2D3748; stroke-width: 1; }
      .region-s3 { fill: #A51C30; fill-opacity: 0.2; stroke: #A51C30; stroke-width: 1; stroke-dasharray: 4,4; }
      .line-temp { fill: none; stroke: #A51C30; stroke-width: 3; }
      .limit-line { stroke: #1F407A; stroke-width: 2; stroke-dasharray: 5,5; }
      .annotation { font-family: sans-serif; font-size: 14px; font-weight: bold; fill: #1F407A; }
      .annotation-red { font-family: sans-serif; font-size: 14px; font-weight: bold; fill: #A51C30; }
    </style>
  </defs>

  <!-- Panel A: Torque-Speed Envelope -->
  <text x="50" y="40" class="title">(A) Torque-Speed Operating Envelope</text>
  
  <line x1="100" y1="100" x2="100" y2="500" class="axis" />
  <line x1="100" y1="500" x2="500" y2="500" class="axis" />
  <text x="50" y="300" class="label" text-anchor="middle" transform="rotate(-90 50,300)">Torque (N·m)</text>
  <text x="300" y="540" class="label" text-anchor="middle">Speed (rad/s)</text>

  <!-- S1 Continuous Region -->
  <path fill="none" d="M 100 500 L 100 350 L 350 350 L 450 500 Z" class="region-s1" />
  <text x="225" y="450" class="annotation" text-anchor="middle">S1 Continuous</text>
  <text x="225" y="470" class="label" text-anchor="middle">Thermal Equilibrium</text>

  <!-- S3 Intermittent Region -->
  <path fill="none" d="M 100 350 L 100 150 L 250 150 C 300 150 320 250 350 350 Z" class="region-s3" />
  <text x="200" y="250" class="annotation-red" text-anchor="middle">S3 Intermittent</text>
  <text x="200" y="270" class="label" text-anchor="middle">Peak Acceleration</text>

  <text x="110" y="140" class="label">Peak Torque (Inverter Limit)</text>
  <text x="110" y="340" class="label">Continuous Torque (Thermal Limit)</text>
  <line x1="100" y1="150" x2="480" y2="150" class="limit-line" />
  
  <line x1="250" y1="150" x2="250" y2="500" class="limit-line" stroke="#2D3748" />
  <text x="250" y="520" class="label" text-anchor="middle">Base Speed</text>


  <!-- Panel B: Cyclic Thermal Duty Dynamics -->
  <text x="600" y="40" class="title">(B) Cyclic Thermal Duty Dynamics (Temperature vs Time)</text>
  
  <line x1="650" y1="100" x2="650" y2="500" class="axis" />
  <line x1="650" y1="500" x2="1150" y2="500" class="axis" />
  <text x="600" y="300" class="label" text-anchor="middle" transform="rotate(-90 600,300)">Winding Temperature (°C)</text>
  <text x="900" y="540" class="label" text-anchor="middle">Time (s)</text>
  
  <!-- Ambient temp -->
  <line x1="650" y1="450" x2="1150" y2="450" class="grid" />
  <text x="610" y="455" class="label">Ambient</text>
  
  <!-- Max Insulation Limit -->
  <line x1="650" y1="150" x2="1150" y2="150" stroke="#A51C30" stroke-width="2" stroke-dasharray="5,5" />
  <text x="1050" y="140" class="annotation-red">Insulation Limit</text>

  <!-- Curve: Heat up then cool down -->
  <!-- Start at ambient (650, 450) -->
  <!-- Heat up exponentially to 150 y over 100 x (750, 150) -->
  <path fill="none" d="M 650 450 Q 700 450 720 250 T 750 160" class="line-temp" />
  <!-- Cool down exponentially from 750 to 950 -->
  <path fill="none" d="M 750 160 Q 800 400 950 430" class="line-temp" stroke="#1F407A" />
  
  <!-- Second cycle -->
  <path fill="none" d="M 950 430 Q 1000 430 1020 230 T 1050 160" class="line-temp" />
  <path fill="none" d="M 1050 160 Q 1100 400 1150 430" class="line-temp" stroke="#1F407A" />

  <!-- Time markers -->
  <line x1="650" y1="500" x2="650" y2="510" class="axis" />
  <line x1="750" y1="500" x2="750" y2="510" class="axis" />
  <line x1="950" y1="500" x2="950" y2="510" class="axis" />
  
  <text x="700" y="525" class="label" text-anchor="middle">Burst (t_on)</text>
  <text x="850" y="525" class="label" text-anchor="middle">Recovery (t_off)</text>

  <!-- Brackets -->
  <path d="M 650 490 L 650 495 L 750 495 L 750 490" fill="none" stroke="#2D3748" stroke-width="2" />
  <path d="M 750 490 L 750 495 L 950 495 L 950 490" fill="none" stroke="#2D3748" stroke-width="2" />

</svg>
"""
os.makedirs('books/vol4/02_body/images/svg', exist_ok=True)
with open('books/vol4/02_body/images/svg/fig02_motor_torque_speed_envelope.svg', 'w') as f:
    f.write(svg)
