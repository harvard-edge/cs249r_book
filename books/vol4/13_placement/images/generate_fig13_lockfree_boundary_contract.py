import xml.etree.ElementTree as ET

def build_svg():
    svg = ET.Element('svg', xmlns="http://www.w3.org/2000/svg", viewBox="0 0 1000 550", width="100%", height="100%")
    
    style = ET.SubElement(svg, 'style')
    style.text = """
        .bg { fill: #FFFFFF; }
        .title { font-family: sans-serif; font-size: 18px; font-weight: bold; fill: #1F407A; }
        .text { font-family: sans-serif; font-size: 14px; fill: #2D3748; }
        .text-bold { font-family: sans-serif; font-size: 14px; font-weight: bold; fill: #2D3748; }
        .small { font-family: sans-serif; font-size: 12px; fill: #2D3748; }
        .small-white { font-family: sans-serif; font-size: 12px; fill: #FFFFFF; }
        .warn { font-family: sans-serif; font-size: 14px; font-weight: bold; fill: #A51C30; }
        .box { fill: #E2E8F0; stroke: #2D3748; stroke-width: 1.5; }
        .box-blue { fill: #1F407A; stroke: #1F407A; stroke-width: 1.5; }
        .box-red { fill: #A51C30; stroke: #A51C30; stroke-width: 1.5; }
        .box-light-blue { fill: #EBF8FF; stroke: #1F407A; stroke-width: 1.5; }
        .box-light-red { fill: #FDE8E8; stroke: #A51C30; stroke-width: 1.5; }
        .line { stroke: #2D3748; stroke-width: 1.5; fill: none; }
        .line-thick { stroke: #2D3748; stroke-width: 2.5; fill: none; }
        .line-dashed { stroke: #2D3748; stroke-width: 1.5; stroke-dasharray: 4,4; fill: none; }
        .arrow { stroke: #1F407A; stroke-width: 2; fill: none; }
        .arrow-red { stroke: #A51C30; stroke-width: 2; fill: none; }
    """
    
    ET.SubElement(svg, 'rect', width="1000", height="550", class_="bg")
    
    defs = ET.SubElement(svg, 'defs')
    marker = ET.SubElement(defs, 'marker', id="arrow", markerWidth="10", markerHeight="10", refX="9", refY="3", orient="auto", markerUnits="strokeWidth")
    ET.SubElement(marker, 'path', d="M0,0 L0,6 L9,3 z", fill="#2D3748")
    
    marker_blue = ET.SubElement(defs, 'marker', id="arrow-blue", markerWidth="10", markerHeight="10", refX="9", refY="3", orient="auto", markerUnits="strokeWidth")
    ET.SubElement(marker_blue, 'path', d="M0,0 L0,6 L9,3 z", fill="#1F407A")
    
    marker_red = ET.SubElement(defs, 'marker', id="arrow-red", markerWidth="10", markerHeight="10", refX="9", refY="3", orient="auto", markerUnits="strokeWidth")
    ET.SubElement(marker_red, 'path', d="M0,0 L0,6 L9,3 z", fill="#A51C30")
    
    # ================= Panel (a) Protocol =================
    ET.SubElement(svg, 'text', x="20", y="30", class_="title").text = "(a) Inter-Core Lock-Free State Exchange (Seqlock)"
    
    # Writer
    ET.SubElement(svg, 'rect', x="20", y="60", width="220", height="160",, class_="box-light-blue")
    ET.SubElement(svg, 'text', x="130", y="90", class_="title", text_anchor="middle").text = "Proposal Writer"
    ET.SubElement(svg, 'text', x="130", y="110", class_="small", text_anchor="middle").text = "(Untrusted NPU/MPU)"
    
    ET.SubElement(svg, 'text', x="130", y="150", class_="text", text_anchor="middle").text = "1. S = S + 1 (odd)"
    ET.SubElement(svg, 'text', x="130", y="170", class_="text", text_anchor="middle").text = "2. Write Payload"
    ET.SubElement(svg, 'text', x="130", y="190", class_="text", text_anchor="middle").text = "3. S = S + 1 (even)"
    
    # Shared Memory
    ET.SubElement(svg, 'rect', x="350", y="60", width="280", height="240",, class_="box")
    ET.SubElement(svg, 'text', x="490", y="90", class_="title", text_anchor="middle").text = "Shared SRAM (Zero-Copy)"
    ET.SubElement(svg, 'text', x="490", y="110", class_="small", text_anchor="middle").text = "No mutex, no allocation"
    
    ET.SubElement(svg, 'rect', x="370", y="125", width="240", height="30", class_="box-blue")
    ET.SubElement(svg, 'text', x="490", y="145", class_="small-white", text_anchor="middle").text = "Monotonic Sequence Counter (S)"
    
    # Payload Box
    ET.SubElement(svg, 'rect', x="370", y="165", width="240", height="120", class_="box")
    ET.SubElement(svg, 'text', x="490", y="185", class_="text-bold", text_anchor="middle").text = "Self-Contained Payload (256 B)"
    ET.SubElement(svg, 'text', x="380", y="210", class_="small").text = "• Cmd Mass Flow (m_dot_cmd)"
    ET.SubElement(svg, 'text', x="380", y="230", class_="small").text = "• Sequence Index (k)"
    ET.SubElement(svg, 'text', x="380", y="250", class_="small").text = "• Hardware Timestamp (t_prod)"
    ET.SubElement(svg, 'text', x="380", y="270", class_="small").text = "• Lease Validity (Δt_valid), CRC-32"
    
    # Reader
    ET.SubElement(svg, 'rect', x="740", y="60", width="240", height="160",, class_="box-light-red")
    ET.SubElement(svg, 'text', x="860", y="90", class_="title", text_anchor="middle").text = "Safety Reader"
    ET.SubElement(svg, 'text', x="860", y="110", class_="small", text_anchor="middle").text = "(Deterministic MCU)"
    
    ET.SubElement(svg, 'text', x="860", y="150", class_="text", text_anchor="middle").text = "1. S1 = Read(S)"
    ET.SubElement(svg, 'text', x="860", y="170", class_="text", text_anchor="middle").text = "2. Read Payload"
    ET.SubElement(svg, 'text', x="860", y="190", class_="text", text_anchor="middle").text = "3. S2 = Read(S)"
    
    # Arrows for Memory Barriers
    # Writer to Shared
    ET.SubElement(svg, 'path', d="M240 140 L340 140", class_="arrow", marker_end="url(#arrow-blue)")
    ET.SubElement(svg, 'text', x="290", y="130", class_="small", text_anchor="middle").text = "Store-Release"
    
    # Shared to Reader
    ET.SubElement(svg, 'path', d="M630 140 L730 140", class_="arrow-red", marker_end="url(#arrow-red)")
    ET.SubElement(svg, 'text', x="680", y="130", class_="small", text_anchor="middle").text = "Load-Acquire"
    
    # ================= Panel (b) Preemption Matrix =================
    ET.SubElement(svg, 'text', x="20", y="340", class_="title").text = "(b) Writer Preemption Matrix & Deterministic Fallbacks"
    
    # Table headers
    ET.SubElement(svg, 'rect', x="20", y="360", width="220", height="30", class_="box-blue")
    ET.SubElement(svg, 'text', x="130", y="380", class_="small-white", text_anchor="middle").text = "Failure Class"
    
    ET.SubElement(svg, 'rect', x="240", y="360", width="380", height="30", class_="box-blue")
    ET.SubElement(svg, 'text', x="430", y="380", class_="small-white", text_anchor="middle").text = "Detection Condition (Safety Reader)"
    
    ET.SubElement(svg, 'rect', x="620", y="360", width="360", height="30", class_="box-red")
    ET.SubElement(svg, 'text', x="800", y="380", class_="small-white", text_anchor="middle").text = "Bounded Real-Time Fallback (<5 µs)"
    
    # Rows
    rows = [
        ("Torn / In-Flight", "S1 % 2 != 0  OR  S1 != S2", "Trap: Discard snapshot, use previous safe state"),
        ("Malformed", "CRC-32 fails  OR  m_dot_cmd out of bounds", "Trap: Reject command, clamp valve"),
        ("Stale", "t_now > t_prod + Δt_valid", "Timeout: Clamp valve (transient), ramp down if > 10ms"),
        ("Missing", "k_new > k_prev + 1", "Skip detected: Log fault, hold state")
    ]
    
    y_offset = 390
    for fail, cond, fallback in rows:
        ET.SubElement(svg, 'rect', x="20", y=str(y_offset), width="220", height="30", class_="box")
        ET.SubElement(svg, 'text', x="130", y=str(y_offset+20), class_="text-bold", text_anchor="middle").text = fail
        
        ET.SubElement(svg, 'rect', x="240", y=str(y_offset), width="380", height="30", class_="box")
        ET.SubElement(svg, 'text', x="430", y=str(y_offset+20), class_="text", text_anchor="middle").text = cond
        
        ET.SubElement(svg, 'rect', x="620", y=str(y_offset), width="360", height="30", class_="box-light-red")
        ET.SubElement(svg, 'text', x="800", y=str(y_offset+20), class_="small", text_anchor="middle").text = fallback
        
        y_offset += 30
        
    ET.SubElement(svg, 'text', x="500", y="530", class_="warn", text_anchor="middle").text = "Guarantees continuous 1000 Hz actuator regulation even during host crashes"
    
    with open("svg/fig13_lockfree_boundary_contract.svg", "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(ET.tostring(svg, encoding="unicode"))

if __name__ == "__main__":
    build_svg()
