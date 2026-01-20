// Satellite Configuration
const satelliteConfig = {
    radius: 90,
    rotationSpeed: 0.005,
    tilt: 0.3
};

let satelliteGroup;
let satelliteBody;
let solarPanels;
let satUsers = [];
let tooltipSelection;
let currentAngle = 0;
let originX = 0;
let originY = 0;

export function initCloud(svgSelection, width, height, tooltipSel) {
    // Store origin for animation loop
    originX = width - 180;
    originY = height / 2;
    tooltipSelection = tooltipSel;

    satelliteGroup = svgSelection.append('g')
        .attr("transform", `translate(${originX}, ${originY})`)
        .style("cursor", "pointer");

    // Draw Satellite Structure (Outline Style)
    solarPanels = satelliteGroup.append("g").attr("class", "solar-panels");

    const panelWidth = 75;
    const panelHeight = 30;

    // Left Panel
    solarPanels.append("rect")
        .attr("x", -panelWidth - 35)
        .attr("y", -panelHeight/2)
        .attr("width", panelWidth)
        .attr("height", panelHeight)
        .attr("fill", "rgba(255,255,255,0.1)")
        .attr("stroke", "#333")
        .attr("stroke-width", 1.5)
        .attr("rx", 2);

    // Right Panel
    solarPanels.append("rect")
        .attr("x", 35)
        .attr("y", -panelHeight/2)
        .attr("width", panelWidth)
        .attr("height", panelHeight)
        .attr("fill", "rgba(255,255,255,0.1)")
        .attr("stroke", "#333")
        .attr("stroke-width", 1.5)
        .attr("rx", 2);

    // Connecting Rod
    solarPanels.append("line")
        .attr("x1", -panelWidth - 35)
        .attr("x2", panelWidth + 35)
        .attr("y1", 0)
        .attr("y2", 0)
        .attr("stroke", "#555")
        .attr("stroke-width", 2);

    // Central Body
    satelliteBody = satelliteGroup.append("circle")
        .attr("r", 35)
        .attr("fill", "#fff")
        .attr("stroke", "#333")
        .attr("stroke-width", 2);

    // Detail Ring
    satelliteGroup.append("circle")
        .attr("r", 18)
        .attr("fill", "none")
        .attr("stroke", "#777")
        .attr("stroke-width", 1)
        .attr("stroke-dasharray", "3,3");

    // Hint Line Group (Vertical line + label) - Static
    const hintGroup = svgSelection.append("g")
        .attr("transform", `translate(${originX - 130}, ${originY})`); // Position to the left

    // Vertical line
    hintGroup.append("line")
        .attr("x1", 0).attr("y1", -40)
        .attr("x2", 0).attr("y2", 40)
        .attr("stroke", "#ccc")
        .attr("stroke-width", 1);

    // Top Serif
    hintGroup.append("line")
        .attr("x1", -3).attr("y1", -40)
        .attr("x2", 3).attr("y2", -40)
        .attr("stroke", "#ccc")
        .attr("stroke-width", 1);

    // Bottom Serif
    hintGroup.append("line")
        .attr("x1", -3).attr("y1", 40)
        .attr("x2", 3).attr("y2", 40)
        .attr("stroke", "#ccc")
        .attr("stroke-width", 1);

    // Location Update Hint Box
    const noteGroup = hintGroup.append("g")
        .attr("transform", "translate(-105, -23)");

    // Shadow Rectangle (matching Hello World style)
    noteGroup.append("rect")
        .attr("x", 2)
        .attr("y", 2)
        .attr("width", 100)
        .attr("height", 46)
        .attr("fill", "rgba(255, 102, 0, 0.3)");

    // Background Box
    noteGroup.append("rect")
        .attr("width", 100)
        .attr("height", 46)
        .attr("fill", "#fff")
        .attr("stroke", "#555")
        .attr("stroke-width", 0.8)
        .attr("rx", 0);

    // Text Content
    const textElem = noteGroup.append("text")
        .attr("x", 5)
        .attr("y", 12)
        .style("font-family", "Courier New")
        .style("font-size", "7px")
        .style("font-weight", "bold")
        .style("fill", "#333")
        .style("pointer-events", "all");

    textElem.append("tspan")
        .attr("x", 5)
        .attr("dy", 0)
        .style("fill", "#777")
        .text("Launched: 12.2025 🔥");

    textElem.append("tspan")
        .attr("x", 5)
        .attr("dy", 12)
        .style("fill", "#333")
        .text("Update location in");

    const linkTspan = textElem.append("tspan")
        .attr("x", 5)
        .attr("dy", 10)
        .style("fill", "#ff6600")
        .style("text-decoration", "underline")
        .style("cursor", "pointer")
        .text("your profile")
        .on("click", () => {
            const authBtn = document.getElementById('authBtn');
            if (authBtn) authBtn.click();
        });

    textElem.append("tspan")
        .style("fill", "#333")
        .style("text-decoration", "none")
        .text("");
}

export function updateCloudUsers(users) {
    satUsers = users.map((d, i) => {
        // Random distribution inside the habitat
        const angle = Math.random() * Math.PI * 2;
        const radius = Math.random() * 25; // Keep within body r=35

        return {
            ...d,
            lx: Math.cos(angle) * radius,
            ly: Math.sin(angle) * radius
        };
    });

    const markers = satelliteGroup.selectAll('circle.sat-marker')
        .data(satUsers, d => d.user);

    markers.exit().remove();

    markers.enter()
        .append('circle')
        .attr('class', 'sat-marker')
        .attr('r', 3)
        .attr("fill", "#2ecc71")
        .attr("stroke", "#fff")
        .attr("stroke-width", 1)
        .style("cursor", "pointer")
        .on("mouseover", function(event, d) {
             d3.select(this).attr("r", 5);
             showSatTooltip(event, d);
        })
        .on("mouseout", function() {
             d3.select(this).attr("r", 3);
             tooltipSelection.style("opacity", 0);
        });
}

export function animateCloud() {
    currentAngle += satelliteConfig.rotationSpeed;

    // Rocking animation
    const deg = Math.sin(currentAngle) * 15;

    // Use stored origin coordinates to prevent jumping
    satelliteGroup.attr("transform", `translate(${originX}, ${originY}) rotate(${deg})`);

    // Update markers
    const markers = satelliteGroup.selectAll('circle.sat-marker');
    markers
        .attr("cx", d => d.lx)
        .attr("cy", d => d.ly);
}

function showSatTooltip(event, d) {
    tooltipSelection.html(`
        <h3>${d.displayName}</h3>
        <div class="info-row">Status: <span class="highlight">${d.completed}</span></div>
        <div class="info-row"><i>${d.institution}</i></div>
        <div class="info-row" style="color:#2ecc71;">🛰️ Unknown Location</div>
    `);

    tooltipSelection
        .style("left", (event.pageX + 15) + "px")
        .style("top", (event.pageY - 15) + "px")
        .style("opacity", 1);
}
