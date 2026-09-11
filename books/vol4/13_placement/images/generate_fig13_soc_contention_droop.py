import xml.etree.ElementTree as ET

def build_svg():
    svg = ET.Element('svg', xmlns="http://www.w3.org/2000/svg", viewBox="0 0 1000 450", width="100%", height="100%")
    
    # Styles
    style = ET.SubElement(svg, 'style')
    style.text = """
        .bg { fill: #FFFFFF; }
        .title { font-family: sans-serif; font-size: 18px; font-weight: bold; fill: #1F407A; }
        .text { font-family: sans-serif; font-size: 14px; fill: #2D3748; }
        .small { font-family: sans-serif; font-size: 12px; fill: #2D3748; }
        .small-white { font-family: sans-serif; font-size: 12px; fill: #FFFFFF; }
        .warn { font-family: sans-serif; font-size: 14px; font-weight: bold; fill: #A51C30; }
        .box { fill: #E2E8F0; stroke: #2D3748; stroke-width: 1.5; }
        .box-blue { fill: #1F407A; stroke: #1F407A; stroke-width: 1.5; }
        .box-red { fill: #A51C30; stroke: #A51C30; stroke-width: 1.5; }
        .box-light-blue { fill: #EBF8FF; stroke: #1F407A; stroke-width: 1.5; }
        .box-light-red { fill: #FDE8E8; stroke: #A51C30; stroke-width: 1.5; }
        .line { stroke: #2D3748; stroke-width: 1.5; fill: none; }
        .line-dashed { stroke: #2D3748; stroke-width: 1.5; stroke-dasharray: 4,4; fill: none; }
        .line-red { stroke: #A51C30; stroke-width: 2; fill: none; }
        .line-blue { stroke: #1F407A; stroke-width: 2; fill: none; }
    """
    
    # Background
    ET.SubElement(svg, 'rect', width="1000", height="450", class_="bg")
    
    # Defs for arrows
    defs = ET.SubElement(svg, 'defs')
    marker = ET.SubElement(defs, 'marker', id="arrow", markerWidth="10", markerHeight="10", refX="9", refY="3", orient="auto", markerUnits="strokeWidth")
    ET.SubElement(marker, 'path', d="M0,0 L0,6 L9,3 z", fill="#2D3748")
    
    # ================= Panel (a) Memory Contention =================
    ET.SubElement(svg, 'text', x="20", y="30", class_="title").text = "(a) On-Chip Memory Bus & Cache Contention"
    
    # NPU
    ET.SubElement(svg, 'rect', x="20", y="60", width="160", height="60",, class_="box-light-blue")
    ET.SubElement(svg, 'text', x="100", y="85", class_="title", text_anchor="middle").text = "Proposal NPU"
    ET.SubElement(svg, 'text', x="100", y="105", class_="small", text_anchor="middle").text = "Streams 256 kB bursts"
    
    # MCU
    ET.SubElement(svg, 'rect', x="20", y="150", width="160", height="60",, class_="box-light-red")
    ET.SubElement(svg, 'text', x="100", y="175", class_="title", text_anchor="middle").text = "Real-Time MCU"
    ET.SubElement(svg, 'text', x="100", y="195", class_="small", text_anchor="middle").text = "Sensor read (5.0 µs)"
    
    # Connections to Bus
    ET.SubElement(svg, 'path', d="M180 90 L230 90", class_="line", marker_end="url(#arrow)")
    ET.SubElement(svg, 'path', d="M180 180 L230 180", class_="line", marker_end="url(#arrow)")
    
    # Shared Interconnect & Queue
    ET.SubElement(svg, 'rect', x="240", y="50", width="160", height="170",, class_="box")
    ET.SubElement(svg, 'text', x="320", y="75", class_="title", text_anchor="middle").text = "DRAM Queue"
    
    # Queue Items
    ET.SubElement(svg, 'rect', x="250", y="90", width="140", height="25", class_="box-blue")
    ET.SubElement(svg, 'text', x="320", y="107", class_="small-white", text_anchor="middle").text = "NPU Burst (256 kB)"
    
    ET.SubElement(svg, 'rect', x="250", y="120", width="140", height="25", class_="box-blue")
    ET.SubElement(svg, 'text', x="320", y="137", class_="small-white", text_anchor="middle").text = "NPU Burst (256 kB)"
    
    ET.SubElement(svg, 'rect', x="250", y="180", width="140", height="25", class_="box-red")
    ET.SubElement(svg, 'text', x="320", y="197", class_="small-white", text_anchor="middle").text = "MCU Read (64 kB)"
    
    ET.SubElement(svg, 'text', x="410", y="180", class_="warn").text = "Wait: 27.6 µs"
    
    # Timing Waterfall
    ET.SubElement(svg, 'text', x="20", y="270", class_="title").text = "Timing Waterfall (1.0 ms control loop)"
    
    # Axis
    ET.SubElement(svg, 'path', d="M20 290 L480 290", class_="line")
    ET.SubElement(svg, 'text', x="20", y="305", class_="small").text = "0"
    ET.SubElement(svg, 'text', x="360", y="305", class_="small", text_anchor="middle").text = "120 µs Budget"
    ET.SubElement(svg, 'path', d="M360 285 L360 350", class_="line-dashed")
    
    # Nominal
    ET.SubElement(svg, 'text', x="20", y="325", class_="text").text = "Nominal:"
    ET.SubElement(svg, 'rect', x="80", y="310", width="250", height="20", class_="box-light-red")
    ET.SubElement(svg, 'text', x="205", y="325", class_="small", text_anchor="middle").text = "Execution (Includes 5.0 µs read)"
    
    # Contended
    ET.SubElement(svg, 'text', x="20", y="355", class_="text").text = "Contended:"
    ET.SubElement(svg, 'rect', x="80", y="340", width="80", height="20", class_="box")
    ET.SubElement(svg, 'text', x="120", y="355", class_="small", text_anchor="middle").text = "Stall: 27.6 µs"
    ET.SubElement(svg, 'rect', x="160", y="340", width="250", height="20", class_="box-light-red")
    ET.SubElement(svg, 'text', x="285", y="355", class_="small", text_anchor="middle").text = "Execution shifted"
    
    ET.SubElement(svg, 'text', x="415", y="355", class_="warn").text = "+7.6 µs Overrun"
    
    # Vertical Divider
    ET.SubElement(svg, 'path', d="M500 20 L500 430", class_="line-dashed")
    
    # ================= Panel (b) PDN Interference =================
    ET.SubElement(svg, 'text', x="520", y="30", class_="title").text = "(b) Power Distribution Network & Voltage Droop"
    
    # Current Plot
    ET.SubElement(svg, 'text', x="520", y="80", class_="text").text = "Current (A)"
    ET.SubElement(svg, 'path', d="M580 140 L640 140 L650 60 L920 60", class_="line-red")
    ET.SubElement(svg, 'text', x="700", y="80", class_="warn").text = "ΔI = 6.0 A, dI/dt = 3.0 A/ns"
    ET.SubElement(svg, 'text', x="700", y="100", class_="text").text = "(NPU Array Activation)"
    
    # Voltage Plot
    ET.SubElement(svg, 'text', x="520", y="210", class_="text").text = "Voltage (V)"
    # V_nom = 0.85V -> y=180, V_min = 0.78V -> y=250, Droop = 138mV -> y=260
    ET.SubElement(svg, 'path', d="M580 180 L640 180 L655 260 L680 190 L710 200 L740 185 L920 185", class_="line-blue")
    ET.SubElement(svg, 'path', d="M580 250 L920 250", class_="line-dashed")
    ET.SubElement(svg, 'text', x="930", y="255", class_="warn").text = "Vmin = 0.78 V"
    ET.SubElement(svg, 'text', x="520", y="185", class_="text").text = "0.85 V"
    
    ET.SubElement(svg, 'text', x="660", y="280", class_="warn").text = "ΔV = 138 mV Droop (16.2% collapse)"
    
    # Consequence Box
    ET.SubElement(svg, 'rect', x="580", y="310", width="340", height="70",, class_="box-light-red")
    ET.SubElement(svg, 'text', x="750", y="340", class_="warn", text_anchor="middle").text = "Gate Delay Stretch: +38%"
    ET.SubElement(svg, 'text', x="750", y="360", class_="warn", text_anchor="middle").text = "Emergency Clock Halving (1.2 GHz → 600 MHz)"
    
    # Write to file
    with open("svg/fig13_soc_contention_droop.svg", "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(ET.tostring(svg, encoding="unicode"))

if __name__ == "__main__":
    build_svg()
