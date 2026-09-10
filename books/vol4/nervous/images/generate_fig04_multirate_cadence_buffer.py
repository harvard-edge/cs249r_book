import os

svg_code = []
svg_code.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 650" width="900" height="650">')
svg_code.append('<defs>')
svg_code.append('<style>')
svg_code.append('.title { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 16px; font-weight: bold; fill: #1F407A; }')
svg_code.append('.label { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 14px; fill: #2D3748; }')
svg_code.append('.small-label { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 12px; fill: #2D3748; }')
svg_code.append('.tiny-label { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 10px; fill: #2D3748; }')
svg_code.append('.box-text { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 12px; fill: #FFFFFF; font-weight: bold; }')
svg_code.append('.line { stroke: #2D3748; stroke-width: 2; fill: none; }')
svg_code.append('.arrow { stroke: #1F407A; stroke-width: 2; fill: none; }')
svg_code.append('.arrow-red { stroke: #A51C30; stroke-width: 2; fill: none; }')
svg_code.append('</style>')
svg_code.append('<marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">')
svg_code.append('  <polygon points="0 0, 10 3.5, 0 7" fill="#1F407A" />')
svg_code.append('</marker>')
svg_code.append('<marker id="arrowhead-red" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">')
svg_code.append('  <polygon points="0 0, 10 3.5, 0 7" fill="#A51C30" />')
svg_code.append('</marker>')
svg_code.append('</defs>')

# Background
svg_code.append('<rect width="900" height="650" fill="#FFFFFF"/>')

# Panel A
svg_code.append('<text x="30" y="30" class="title">Panel A: Multi-Rate Execution Cadence</text>')

def ms_to_x(ms): return 120 + ms * 12

y_perc = 70
y_pol = 140
y_refl = 210

for y in [y_perc, y_pol, y_refl]:
    svg_code.append(f'<line x1="120" y1="{y}" x2="850" y2="{y}" class="line"/>')
    svg_code.append(f'<polygon points="850,{y-4} 858,{y} 850,{y+4}" fill="#2D3748"/>')

svg_code.append(f'<text x="110" y="{y_perc-15}" class="label" text-anchor="end">Perception</text>')
svg_code.append(f'<text x="110" y="{y_perc+5}" class="small-label" text-anchor="end">(50-100 Hz)</text>')
svg_code.append(f'<text x="110" y="{y_pol-15}" class="label" text-anchor="end">Policy</text>')
svg_code.append(f'<text x="110" y="{y_pol+5}" class="small-label" text-anchor="end">(50 Hz)</text>')
svg_code.append(f'<text x="110" y="{y_refl-15}" class="label" text-anchor="end">Reflex Control</text>')
svg_code.append(f'<text x="110" y="{y_refl+5}" class="small-label" text-anchor="end">(1000 Hz)</text>')

def draw_block(x_start, y, width, height, fill, stroke, text=""):
    svg_code.append(f'<rect x="{x_start}" y="{y}" width="{width}" height="{height}" fill="{fill}" stroke="{stroke}" stroke-width="1.5" rx="3"/>')
    if text:
        svg_code.append(f'<text x="{x_start + width/2}" y="{y + height/2 + 4}" class="box-text" text-anchor="middle">{text}</text>')

# Perception Blocks (10ms exposure)
draw_block(ms_to_x(0), y_perc-15, 120, 30, "#4A5568", "#2D3748", "Exposure")
draw_block(ms_to_x(20), y_perc-15, 120, 30, "#4A5568", "#2D3748", "Exposure")
draw_block(ms_to_x(40), y_perc-15, 120, 30, "#4A5568", "#2D3748", "Exposure")

# Policy Blocks (20ms) 
draw_block(ms_to_x(10), y_pol-20, 240, 40, "#1F407A", "#1a365d", "Policy Deliberation (20 ms)")
draw_block(ms_to_x(30), y_pol-20, 264, 40, "#A51C30", "#7b1523", "Tail Overrun (22 ms)")
draw_block(ms_to_x(52), y_pol-20, 96, 40, "#1F407A", "#1a365d", "Policy")

# Reflex Control Ticks (1ms)
for i in range(61):
    x = ms_to_x(i)
    h = 8 if i % 10 == 0 else 4
    svg_code.append(f'<line x1="{x}" y1="{y_refl-h}" x2="{x}" y2="{y_refl+h}" class="line"/>')
    if i % 10 == 0:
        svg_code.append(f'<text x="{x}" y="{y_refl+20}" class="small-label" text-anchor="middle">{i} ms</text>')

# Inset
svg_code.append('<rect x="550" y="250" width="300" height="70" fill="#F7FAFC" stroke="#CBD5E0" stroke-width="1.5" rx="4"/>')
svg_code.append('<text x="560" y="265" class="small-label" font-weight="bold">1 ms Slot Budgeting (Inset)</text>')
ins_x, ins_w = 570, 260
svg_code.append(f'<line x1="{ins_x}" y1="290" x2="{ins_x+ins_w}" y2="290" class="line"/>')
svg_code.append(f'<line x1="{ins_x}" y1="285" x2="{ins_x}" y2="295" class="line"/>')
svg_code.append(f'<line x1="{ins_x+ins_w}" y1="285" x2="{ins_x+ins_w}" y2="295" class="line"/>')
svg_code.append(f'<text x="{ins_x}" y="310" class="tiny-label" text-anchor="middle">0 \u03bcs</text>')
svg_code.append(f'<text x="{ins_x+ins_w}" y="310" class="tiny-label" text-anchor="middle">1000 \u03bcs (D)</text>')

c_w = ins_w * (240.0/1000.0)
svg_code.append(f'<rect x="{ins_x}" y="280" width="{c_w}" height="20" fill="#3182CE" stroke="#2B6CB0"/>')
svg_code.append(f'<text x="{ins_x+c_w/2}" y="294" class="tiny-label" fill="#FFFFFF" text-anchor="middle">C\u2264240\u03bcs</text>')
svg_code.append(f'<line x1="{ins_x+c_w}" y1="290" x2="{ins_x+ins_w}" y2="290" stroke="#A51C30" stroke-width="2" stroke-dasharray="2,2"/>')
svg_code.append(f'<text x="{ins_x+c_w + (ins_w-c_w)/2}" y="285" class="tiny-label" fill="#A51C30" text-anchor="middle">760 \u03bcs margin</text>')

# Link reflex axis to inset
svg_code.append(f'<line x1="{ms_to_x(52)}" y1="{y_refl+10}" x2="550" y2="250" stroke="#A0AEC0" stroke-width="1" stroke-dasharray="4,4"/>')
svg_code.append(f'<line x1="{ms_to_x(53)}" y1="{y_refl+10}" x2="850" y2="250" stroke="#A0AEC0" stroke-width="1" stroke-dasharray="4,4"/>')

# Path connections
svg_code.append(f'<path d="M {ms_to_x(10)} {y_perc+15} L {ms_to_x(10)} {y_pol-20}" stroke="#1F407A" stroke-width="1.5" stroke-dasharray="3,3" marker-end="url(#arrowhead)"/>')
svg_code.append(f'<path d="M {ms_to_x(30)} {y_perc+15} L {ms_to_x(30)} {y_pol-20}" stroke="#1F407A" stroke-width="1.5" stroke-dasharray="3,3" marker-end="url(#arrowhead)"/>')
svg_code.append(f'<path d="M {ms_to_x(30)} {y_pol+20} L {ms_to_x(30)} {y_refl-20}" stroke="#A51C30" stroke-width="1.5" stroke-dasharray="3,3" marker-end="url(#arrowhead-red)"/>')
svg_code.append(f'<path d="M {ms_to_x(52)} {y_pol+20} L {ms_to_x(52)} {y_refl-20}" stroke="#A51C30" stroke-width="1.5" stroke-dasharray="3,3" marker-end="url(#arrowhead-red)"/>')
svg_code.append(f'<text x="{ms_to_x(30)+5}" y="{y_refl-30}" class="tiny-label" fill="#A51C30">Write to Buffer</text>')


# Panel B
svg_code.append('<line x1="30" y1="350" x2="870" y2="350" stroke="#E2E8F0" stroke-width="2"/>')
svg_code.append('<text x="30" y="380" class="title">Panel B: SPSC Seqlock Triple-Buffer Architecture</text>')

# Architectures nodes
svg_code.append('<rect x="80" y="440" width="160" height="120" fill="#F7FAFC" stroke="#2D3748" stroke-width="2" rx="4"/>')
svg_code.append('<text x="160" y="465" class="label" font-weight="bold" text-anchor="middle">Application Processor</text>')
svg_code.append('<text x="160" y="485" class="small-label" text-anchor="middle">(Producer, 50 Hz)</text>')
svg_code.append('<rect x="100" y="500" width="120" height="40" fill="#1F407A" stroke="#1a365d" stroke-width="1.5" rx="3"/>')
svg_code.append('<text x="160" y="524" class="box-text" text-anchor="middle">Write Chunk</text>')

svg_code.append('<rect x="660" y="440" width="160" height="120" fill="#F7FAFC" stroke="#2D3748" stroke-width="2" rx="4"/>')
svg_code.append('<text x="740" y="465" class="label" font-weight="bold" text-anchor="middle">Real-Time MCU</text>')
svg_code.append('<text x="740" y="485" class="small-label" text-anchor="middle">(Consumer, 1000 Hz)</text>')
svg_code.append('<rect x="680" y="500" width="120" height="40" fill="#A51C30" stroke="#7b1523" stroke-width="1.5" rx="3"/>')
svg_code.append('<text x="740" y="524" class="box-text" text-anchor="middle">Latch Latest</text>')

# Triple Buffer 
svg_code.append('<rect x="340" y="420" width="220" height="170" fill="#EDF2F7" stroke="#4A5568" stroke-width="2" rx="4"/>')
svg_code.append('<text x="450" y="445" class="label" font-weight="bold" text-anchor="middle">Shared Memory (Triple Buffer)</text>')

b_w, b_h, b_x = 180, 30, 360
b_y1, b_y2, b_y3 = 460, 500, 540

svg_code.append(f'<rect x="{b_x}" y="{b_y1}" width="{b_w}" height="{b_h}" fill="#FFFFFF" stroke="#2D3748" stroke-width="1.5"/>')
svg_code.append(f'<text x="{b_x+10}" y="{b_y1+20}" class="small-label">Buffer 0:</text>')
svg_code.append(f'<text x="{b_x+80}" y="{b_y1+20}" class="small-label" fill="#A51C30" font-style="italic">Inactive (Write)</text>')

svg_code.append(f'<rect x="{b_x}" y="{b_y2}" width="{b_w}" height="{b_h}" fill="#E2E8F0" stroke="#2D3748" stroke-width="1.5"/>')
svg_code.append(f'<text x="{b_x+10}" y="{b_y2+20}" class="small-label">Buffer 1:</text>')
svg_code.append(f'<text x="{b_x+80}" y="{b_y2+20}" class="small-label" fill="#1F407A" font-weight="bold">Newest Committed</text>')

svg_code.append(f'<rect x="{b_x}" y="{b_y3}" width="{b_w}" height="{b_h}" fill="#FFFFFF" stroke="#2D3748" stroke-width="1.5"/>')
svg_code.append(f'<text x="{b_x+10}" y="{b_y3+20}" class="small-label">Buffer 2:</text>')
svg_code.append(f'<text x="{b_x+80}" y="{b_y3+20}" class="small-label" font-style="italic">Active (Read)</text>')

# Seqlock Box
svg_code.append('<rect x="390" y="605" width="120" height="30" fill="#F7FAFC" stroke="#2D3748" stroke-width="1.5" rx="3"/>')
svg_code.append('<text x="450" y="625" class="small-label" font-weight="bold" text-anchor="middle">seqlock: cseq</text>')

# Arrows 
svg_code.append('<path d="M 220 520 L 290 520 L 290 475 L 350 475" class="arrow" marker-end="url(#arrowhead)"/>')
svg_code.append('<text x="290" y="470" class="tiny-label" text-anchor="middle">Write Data</text>')

svg_code.append('<path d="M 160 540 L 160 620 L 380 620" stroke="#1F407A" stroke-width="1.5" stroke-dasharray="4,4" fill="none" marker-end="url(#arrowhead)"/>')
svg_code.append('<rect x="180" y="585" width="130" height="24" fill="#FFFFFF" stroke="#CBD5E0"/>')
svg_code.append('<text x="245" y="601" class="tiny-label" text-anchor="middle">1. atomic_store(odd)</text>')
svg_code.append('<rect x="180" y="625" width="130" height="24" fill="#FFFFFF" stroke="#CBD5E0"/>')
svg_code.append('<text x="245" y="641" class="tiny-label" text-anchor="middle">2. atomic_store(even)</text>')

svg_code.append('<path d="M 540 555 L 610 555 L 610 520 L 670 520" class="arrow-red" marker-end="url(#arrowhead-red)"/>')
svg_code.append('<text x="610" y="515" class="tiny-label" text-anchor="middle">Read Data</text>')

svg_code.append('<path d="M 740 540 L 740 620 L 520 620" stroke="#A51C30" stroke-width="1.5" stroke-dasharray="4,4" fill="none" marker-end="url(#arrowhead-red)"/>')
svg_code.append('<rect x="580" y="605" width="130" height="30" fill="#FFFFFF" stroke="#CBD5E0"/>')
svg_code.append('<text x="645" y="618" class="tiny-label" text-anchor="middle">Check cseq (even)</text>')
svg_code.append('<text x="645" y="630" class="tiny-label" text-anchor="middle">Verify no tearing</text>')

svg_code.append('</svg>')

os.makedirs('images/svg', exist_ok=True)
with open('images/svg/fig04_multirate_cadence_buffer.svg', 'w') as f:
    f.write('\\n'.join(svg_code))
print("Created images/svg/fig04_multirate_cadence_buffer.svg")
