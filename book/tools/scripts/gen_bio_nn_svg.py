
import math

def generate_svg():
    width = 900
    height = 380
    
    svg_header = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" font-family="Helvetica Neue, Helvetica, Arial, sans-serif">
  <defs>
    <style>
      .panel-bg     {{ fill: rgba(46, 139, 87, 0.03); stroke: #2e8b57; stroke-width: 1; rx: 12; }}
      .label-title  {{ font-size: 16px; font-weight: 700; fill: #2e8b57; }}
      .label-part   {{ font-size: 12px; fill: #333; font-weight: 600; }}
      .label-small  {{ font-size: 10px; fill: #555; }}
      .label-var    {{ font-size: 14px; fill: #004c97; font-weight: 600; font-family: Georgia, serif; font-style: italic; }}
      
      .bio-body     {{ fill: #fce4e6; stroke: #a31f34; stroke-width: 1.5; }}
      .bio-nucleus  {{ fill: #eebdc2; stroke: #a31f34; stroke-width: 1; }}
      .bio-dendrite {{ stroke: #a31f34; stroke-width: 2; fill: none; stroke-linecap: round; stroke-linejoin: round; }}
      .bio-axon     {{ stroke: #c9944a; stroke-width: 4; fill: none; stroke-linecap: round; }}
      .bio-myelin   {{ fill: #fdf3e6; stroke: #c9944a; stroke-width: 1.5; }}
      .bio-terminal {{ fill: #fce4e6; stroke: #a31f34; stroke-width: 1; }}
      
      .art-node     {{ fill: #e8f5e9; stroke: #2e8b57; stroke-width: 1.5; }}
      .art-big      {{ fill: #e8f5e9; stroke: #2e8b57; stroke-width: 2; }}
      .art-wire     {{ stroke: #2e8b57; stroke-width: 1.5; }}
      
      .conn-line    {{ fill: none; stroke-width: 1.5; stroke-dasharray: 4,3; opacity: 0.6; }}
      .conn-label   {{ font-size: 10px; font-weight: bold; fill: #555; background: white; }}
    </style>
    <marker id="arrowGray" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#bbb" />
    </marker>
    <marker id="arrowGreen" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#2e8b57" />
    </marker>
    <marker id="arrowConn" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#777" />
    </marker>
  </defs>

  <!-- BACKGROUND PANELS -->
  <rect x="10" y="10" width="400" height="360" class="panel-bg" />
  <rect x="490" y="10" width="400" height="360" class="panel-bg" />
  
  <text x="210" y="40" text-anchor="middle" class="label-title">Biological Neuron</text>
  <text x="690" y="40" text-anchor="middle" class="label-title">Artificial Neuron</text>

'''

    # --- BIOLOGICAL NEURON (Improved) ---
    # Soma (Cell Body) - more organic shape
    bio_soma = '''
    <path d="M 130,150 
             C 100,150 80,180 80,200 
             C 80,230 110,250 140,250 
             C 180,250 200,220 200,200 
             C 200,170 170,150 130,150 Z" class="bio-body" />
    <circle cx="140" cy="200" r="18" class="bio-nucleus" />
    '''
    
    # Dendrites - more branching
    # Main branches radiating from Soma
    dendrites = '''
    <!-- Top Left -->
    <path d="M 100,165 Q 80,140 60,120" class="bio-dendrite" />
    <path d="M 60,120 Q 50,100 45,90" class="bio-dendrite" />
    <path d="M 60,120 Q 70,100 75,90" class="bio-dendrite" />
    
    <!-- Left -->
    <path d="M 85,190 Q 50,180 30,180" class="bio-dendrite" />
    <path d="M 30,180 Q 20,170 15,165" class="bio-dendrite" />
    <path d="M 30,180 Q 20,190 15,195" class="bio-dendrite" />
    
    <!-- Bottom Left -->
    <path d="M 95,230 Q 70,260 50,280" class="bio-dendrite" />
    <path d="M 50,280 Q 40,300 35,310" class="bio-dendrite" />
    <path d="M 50,280 Q 60,300 65,310" class="bio-dendrite" />
    '''
    
    # Axon - clearer segmentation
    axon = '''
    <path d="M 195,200 Q 250,200 350,200" class="bio-axon" />
    <!-- Myelin Sheaths -->
    <rect x="210" y="192" width="30" height="16" rx="5" class="bio-myelin" />
    <rect x="250" y="192" width="30" height="16" rx="5" class="bio-myelin" />
    <rect x="290" y="192" width="30" height="16" rx="5" class="bio-myelin" />
    <rect x="330" y="192" width="30" height="16" rx="5" class="bio-myelin" />
    
    <!-- Terminals -->
    <path d="M 350,200 Q 370,180 380,170" class="bio-dendrite" />
    <path d="M 350,200 Q 370,200 385,200" class="bio-dendrite" />
    <path d="M 350,200 Q 370,220 380,230" class="bio-dendrite" />
    <circle cx="380" cy="170" r="5" class="bio-terminal" />
    <circle cx="385" cy="200" r="5" class="bio-terminal" />
    <circle cx="380" cy="230" r="5" class="bio-terminal" />
    '''
    
    # Labels for Bio
    bio_labels = '''
    <text x="60" y="80" class="label-part">Dendrites</text>
    <text x="140" y="270" text-anchor="middle" class="label-part">Cell Body (Soma)</text>
    <text x="280" y="180" text-anchor="middle" class="label-part">Axon</text>
    <text x="380" y="155" class="label-part">Synapses</text>
    '''

    # --- ARTIFICIAL NEURON ---
    # Inputs
    art_inputs = '''
    <g transform="translate(550, 100)">
      <circle cx="0" cy="0" r="18" class="art-node" />
      <text x="0" y="5" text-anchor="middle" class="label-var">x&#x2080;</text>
      <text x="25" y="5" class="label-small">=1</text>
      
      <circle cx="0" cy="60" r="18" class="art-node" />
      <text x="0" y="65" text-anchor="middle" class="label-var">x&#x2081;</text>
      
      <circle cx="0" cy="120" r="18" class="art-node" />
      <text x="0" y="125" text-anchor="middle" class="label-var">x&#x2082;</text>
      
      <text x="0" y="160" text-anchor="middle" font-size="16" fill="#555">...</text>
      
      <circle cx="0" cy="190" r="18" class="art-node" />
      <text x="0" y="195" text-anchor="middle" class="label-var">x&#x2099;</text>
    </g>
    <text x="550" y="80" text-anchor="middle" class="label-part">Inputs</text>
    '''
    
    # Processing Unit
    art_proc = '''
    <g transform="translate(730, 200)">
      <!-- Wires -->
      <line x1="-180" y1="-100" x2="-35" y2="-10" class="art-wire" />
      <line x1="-180" y1="-40" x2="-35" y2="-5" class="art-wire" />
      <line x1="-180" y1="20" x2="-35" y2="5" class="art-wire" />
      <line x1="-180" y1="90" x2="-35" y2="10" class="art-wire" />
      
      <!-- Weights Labels -->
      <text x="-120" y="-80" class="label-small" fill="#2e8b57">w0</text>
      <text x="-120" y="-35" class="label-small" fill="#2e8b57">w1</text>
      <text x="-120" y="15" class="label-small" fill="#2e8b57">w2</text>
      <text x="-120" y="75" class="label-small" fill="#2e8b57">wn</text>
    
      <!-- Main Node -->
      <circle cx="0" cy="0" r="35" class="art-big" />
      <line x1="0" y1="-35" x2="0" y2="35" stroke="#2e8b57" stroke-width="1" stroke-dasharray="3,3" />
      <text x="-15" y="8" text-anchor="middle" font-size="22" fill="#2e8b57" font-weight="700">z</text>
      <text x="15" y="8" text-anchor="middle" font-size="22" fill="#2e8b57" font-weight="700">f</text>
      
      <!-- Labels -->
      <text x="-20" y="50" text-anchor="middle" class="label-small">Sum</text>
      <text x="20" y="50" text-anchor="middle" class="label-small">Activation</text>
      
      <!-- Output -->
      <line x1="35" y1="0" x2="100" y2="0" stroke="#2e8b57" stroke-width="2" marker-end="url(#arrowGreen)" />
      <text x="110" y="5" class="label-var" font-size="16">y</text>
      <text x="80" y="25" text-anchor="middle" class="label-part">Output</text>
    </g>
    '''

    # --- CONNECTIONS (Simplified) ---
    # Cleaner, dashed lines without text boxes to reduce clutter
    connections = '''
    <!-- 1. Dendrites -> Inputs -->
    <path d="M 90,85 C 200,85 300,100 530,100" class="conn-line" stroke="#004c97" stroke-dasharray="5,5" />
    
    <!-- 2. Synapses -> Weights -->
    <!-- Synapses dots on dendrites -->
    <circle cx="60" cy="120" r="3" fill="#a31f34" />
    <circle cx="45" cy="90" r="3" fill="#a31f34" />
    <circle cx="30" cy="180" r="3" fill="#a31f34" />
    
    <path d="M 65,120 C 200,140 400,150 600,150" class="conn-line" stroke="#a31f34" stroke-dasharray="5,5" />

    <!-- 3. Soma -> Summation -->
    <path d="M 160,200 C 300,200 500,200 690,200" class="conn-line" stroke="#2e8b57" stroke-dasharray="5,5" />
    
    <!-- 4. Axon -> Output -->
    <path d="M 385,200 C 500,280 700,280 800,220" class="conn-line" stroke="#c9944a" stroke-dasharray="5,5" />
    '''
    
    svg_footer = '</svg>'
    
    return svg_header + bio_soma + dendrites + axon + bio_labels + art_inputs + art_proc + connections + svg_footer

with open('book/quarto/contents/vol1/nn_computation/images/svg/bio_nn2ai_nn_improved.svg', 'w') as f:
    f.write(generate_svg())
