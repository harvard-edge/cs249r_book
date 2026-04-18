import * as d3 from 'd3';

export class FlashcardVisualization {
    constructor(shadowRoot) {
        this.shadowRoot = shadowRoot;
        this.width = 600;
        this.height = 400;
        this.margin = { top: 20, right: 20, bottom: 30, left: 40 };
        this.svg = null;
        this.tooltip = null;
        this.simulation = null;
        this.initialized = false;
        this.currentView = 'network';
        this.chapterSets = null;  // Store chapter data
        this.reviewData = null;   // Store review data
        this.isDark = false;      // Theme state
    }

    // Helper method to get theme-aware text color
    getTextColor() {
        return this.isDark ? '#9ca3af' : '#6b7280';
    }

    // Update theme state
    updateTheme() {
        const hostElement = this.shadowRoot.host;
        const currentTheme = hostElement?.getAttribute('data-socratiq-theme') || 'light';
        this.isDark = currentTheme === 'dark';
    }

    showNetworkView() {
        this.currentView = 'network';
        this.updateTheme();
        if (this.chapterSets) {
            this.svg.selectAll('*').remove();
            this.drawNetworkGraph();
        }
    }

    showHeatmapView() {
        this.currentView = 'heatmap';
        this.updateTheme();
        if (this.reviewData) {
            this.svg.selectAll('*').remove();
            this.drawHeatmap();
        }
    }

    update(data) {
        if (!this.initialized && !this.initialize()) return;
        
        this.updateTheme();
        
        // Store data based on current view
        if (this.currentView === 'network') {
            this.chapterSets = data;
            this.drawNetworkGraph();
        } else {
            this.reviewData = data;
            this.drawHeatmap();
        }
    }

    drawHeatmap() {
        if (!this.reviewData) return;
        
        // Initialize constants
        const cellSize = 10;
        const cellPadding = 2;
        const today = new Date();
        const weekWidth = cellSize + cellPadding;

        // Calculate dates for 6 months before and 6 months after current month
        const startDate = new Date(today);
        startDate.setMonth(today.getMonth() - 6);
        startDate.setDate(1); // Start from beginning of month
        const endDate = new Date(today);
        endDate.setMonth(today.getMonth() + 6);

        // Clear existing content
        this.svg.selectAll('*').remove();

        // Create container with padding for labels
        const container = this.svg.append('g')
            .attr('transform', `translate(${this.margin.left + 30}, ${this.margin.top + 20})`);

        // Add title/explanation
        container.append('text')
            .attr('class', 'text-xs font-mono')
            .attr('x', 0)
            .attr('y', -30)
            .attr('fill', this.getTextColor())
            .text('Daily review activity (centered on current month)');

        // Calculate month positions relative to current month
        const monthPositions = [];
        let monthCount = -6; // Start 6 months before
        while (monthCount <= 6) { // Go until 6 months after
            const date = new Date(today);
            date.setMonth(today.getMonth() + monthCount);
            monthPositions.push({
                month: date.toLocaleString('default', { month: 'short' }),
                x: (monthCount + 6) * 4.5 * weekWidth, // Center current month
                current: monthCount === 0
            });
            monthCount++;
        }

        // Add month labels with current month highlighted
        container.selectAll('.month-label')
            .data(monthPositions)
            .enter()
            .append('text')
            .attr('class', d => `month-label text-xs font-mono ${d.current ? 'font-bold' : ''}`)
            .attr('x', d => d.x)
            .attr('y', -5)
            .attr('fill', this.getTextColor())
            .text(d => d.month);

        // Add day labels
        const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
        container.selectAll('.day-label')
            .data(days)
            .enter()
            .append('text')
            .attr('class', 'day-label text-xs font-mono')
            .attr('x', -25)
            .attr('y', (d, i) => i * (cellSize + cellPadding) + cellSize)
            .attr('fill', this.getTextColor())
            .style('text-anchor', 'start')
            .text(d => d);

        // Create color scale
        const colorScale = d3.scaleQuantize()
            .domain([0, 10])
            .range([
                '#ebedf0', // 0
                '#9be9a8', // 1-3
                '#40c463', // 4-6
                '#30a14e', // 7-9
                '#216e39'  // 10+
            ]);

        // Generate cells for the entire date range
        const cells = [];
        const dateFormat = d3.timeFormat('%Y-%m-%d');
        
        for (let d = new Date(startDate); d <= endDate; d.setDate(d.getDate() + 1)) {
            const dateStr = dateFormat(d);
            cells.push({
                date: new Date(d),
                count: this.reviewData[dateStr] || 0,
                dateStr: dateStr
            });
        }

        // Draw cells
        container.selectAll('.day')
            .data(cells)
            .enter()
            .append('rect')
            .attr('class', 'day')
            .attr('width', cellSize)
            .attr('height', cellSize)
            .attr('rx', 2)
            .attr('x', d => {
                const weeksSinceStart = d3.timeWeek.count(startDate, d.date);
                return weeksSinceStart * weekWidth;
            })
            .attr('y', d => d.date.getDay() * (cellSize + cellPadding))
            .attr('fill', d => colorScale(d.count))
            .append('title')
            .text(d => `${d.date.toDateString()}\n${d.count} reviews`);

        // Add legend right under the graph
        const legendY = 7 * (cellSize + cellPadding) + 20; // Position below the cells
        const legend = container.append('g')
            .attr('transform', `translate(${this.width - 250}, ${legendY})`);

        legend.append('text')
            .attr('class', 'text-xs font-mono')
            .attr('x', 0)
            .attr('y', 8)
            .attr('fill', this.getTextColor())
            .text('Less');

        // Add color boxes
        const legendColors = colorScale.range();
        legendColors.forEach((color, i) => {
            legend.append('rect')
                .attr('x', 35 + i * 25)
                .attr('y', 0)
                .attr('width', 20)
                .attr('height', 10)
                .attr('fill', color);
        });

        legend.append('text')
            .attr('class', 'text-xs font-mono')
            .attr('x', 35 + legendColors.length * 25 + 5)
            .attr('y', 8)
            .attr('fill', this.getTextColor())
            .text('More');

        // Add explanation text below legend
        const explanation = container.append('g')
            .attr('transform', `translate(0, ${legendY + 40})`);

        explanation.append('text')
            .attr('class', 'text-xs font-mono')
            .attr('x', 0)
            .attr('y', 0)
            .attr('fill', this.getTextColor())
            .selectAll('tspan')
            .data([
                'How to use this heatmap:',
                '• Darker colors indicate more review sessions',
                '• Look for patterns in your study habits',
                '• Regular, spaced practice (consistent color) is better than cramming (dark spots)',
                '• Aim for consistent daily reviews rather than long gaps'
            ])
            .enter()
            .append('tspan')
            .attr('x', 0)
            .attr('dy', (d, i) => i === 0 ? 0 : '1.2em')
            .text(d => d);
    }

    initialize() {
        console.log("Initializing visualization...");
        
        try {
            const container = this.shadowRoot.querySelector('#visualization-container');
            if (!container) {
                console.error("Visualization container not found");
                return false;
            }

            // Get theme detection
            const hostElement = this.shadowRoot.host;
            const currentTheme = hostElement?.getAttribute('data-socratiq-theme') || 'light';
            this.isDark = currentTheme === 'dark';

            // Clear container first
            container.innerHTML = '';
            
            // Create SVG element with zoom support
            const svg = d3.select(container)
                .append('svg')
                .attr('width', '100%')
                .attr('height', '100%')
                .attr('viewBox', `0 0 ${this.width} ${this.height}`)
                .attr('class', 'rounded-lg shadow-lg')
                .style('background-color', this.isDark ? '#0d1117' : '#ffffff');

            // Add zoom behavior
            const zoom = d3.zoom()
                .scaleExtent([0.1, 4])
                .on('zoom', (event) => {
                    mainGroup.attr('transform', event.transform);
                });

            svg.call(zoom);

            // Create main group for zooming
            const mainGroup = svg.append('g');
            this.svg = mainGroup;

            // Create tooltip
            this.tooltip = d3.select(container)
                .append('div')
                .attr('class', 'absolute hidden p-2 rounded shadow-lg text-sm z-50')
                .style('pointer-events', 'none')
                .style('background-color', this.isDark ? '#0d1117' : '#ffffff')
                .style('color', this.isDark ? '#e6edf3' : '#1f2328')
                .style('border', `1px solid ${this.isDark ? '#30363d' : '#d0d7de'}`);

            // Initialize force simulation
            this.simulation = d3.forceSimulation()
                .force("link", d3.forceLink().id(d => d.id).distance(100))
                .force("charge", d3.forceManyBody().strength(-400))
                .force("center", d3.forceCenter(this.width / 2, this.height / 4))
                .force("collision", d3.forceCollide().radius(d => d.radius + 5));

            this.initialized = true;
            console.log("Visualization initialized successfully");
            return true;

        } catch (error) {
            console.error("Error initializing visualization:", error);
            return false;
        }
    }

    drawNetworkGraph() {
        console.log("Drawing network graph with chapters:", this.chapterSets);
        
        // Clear any existing simulation
        if (this.simulation) {
            this.simulation.stop();
        }
    
        const container = this.svg.append("g")
            .attr("transform", `translate(${this.margin.left}, ${this.margin.top})`);
    
        // Process data for network graph
        const nodes = [];
        const links = [];
    
        // Add chapter nodes first
        for (const [chapterNum, cards] of this.chapterSets) {
            nodes.push({
                id: `chapter-${chapterNum}`,
                type: 'chapter',
                name: `Chapter ${chapterNum}`,
                radius: 30,
                chapter: chapterNum,
                cardCount: cards.length
            });

            // Add card nodes and links for this chapter
            cards.forEach(card => {
                nodes.push({
                    id: `card-${card.id}`,
                    type: 'card',
                    card: card,
                    radius: 8,
                    quality: card.quality || 0,
                    chapter: chapterNum
                });

                links.push({
                    source: `chapter-${chapterNum}`,
                    target: `card-${card.id}`
                });
            });
        }

        console.log("Processed nodes:", nodes);
    
        // Create links first (so they appear under nodes)
        const link = container.append("g")
            .attr("class", "links")
            .lower()
            .selectAll("line")
            .data(links)
            .enter().append("line")
            .attr("stroke", "#999")
            .attr("stroke-opacity", 0.3)
            .attr("stroke-width", 1);

        // Create nodes with click handlers
        const nodeGroup = container.append("g")
            .attr("class", "nodes")
            .selectAll("g")
            .data(nodes)
            .enter()
            .append("g")
            .attr("class", "node-group")
            .style("cursor", "pointer")
            .on("click", (event, d) => {
                console.log("Node clicked:", d);
                if (d.type === 'chapter') {
                    console.log("Chapter node clicked:", {
                        chapter: d.chapter,
                        name: d.name,
                        cardCount: d.cardCount
                    });
                } else {
                    console.log("Card node clicked:", {
                        question: d.card.question,
                        quality: d.quality,
                        chapter: d.chapter
                    });
                    
                    // Dispatch custom event to show the card
                    const showCardEvent = new CustomEvent('show-card', {
                        detail: {
                            chapterNum: d.chapter,
                            cardIndex: d.index-1,
                            card: d.card
                        },
                        bubbles: true,
                        composed: true
                    });
                    this.shadowRoot.dispatchEvent(showCardEvent);
                }
            });

        // Add circle for each node
        nodeGroup.append("circle")
            .attr("r", d => d.radius)
            .attr("fill", d => this.getNodeColor(d));

        // Add text labels
        nodeGroup.append("text")
            .attr("class", "text-[10px] fill-gray-600 dark:fill-gray-300 pointer-events-none")
            .attr("dy", d => d.type === 'chapter' ? "0.35em" : "1.5em")
            .attr("text-anchor", "middle")
            .text(d => {
                if (d.type === 'chapter') {
                    return this.truncateText(`Chapter ${d.chapter} (${d.cardCount})`, 20);
                } else {
                    const cleanQuestion = d.card.question.replace(/#\w+/g, '').trim();
                    return this.truncateText(cleanQuestion, 30);
                }
            });

        // Add legend with vertical stacking
        const legend = this.svg.append("g")
            .attr("transform", `translate(${this.margin.left}, ${this.height - 160})`);

        // Add explanation
        legend.append("text")
            .attr("class", "text-xs font-mono")
            .attr("y", -20)
            .attr("fill", this.getTextColor())
            .text("Flashcard network: Chapters (large) connected to their cards (small)");

        // Update quality levels to show blue progression
        const qualityLevels = [
            { label: "Not Started", quality: 0 },
            { label: "Just Started", quality: 1 },
            { label: "Learning", quality: 2 },
            { label: "Almost Done", quality: 3 }
        ];

        const legendSpacing = 25;
        qualityLevels.forEach((level, i) => {
            const group = legend.append("g")
                .attr("transform", `translate(0, ${i * legendSpacing})`);

            group.append("circle")
                .attr("r", 6)
                .attr("fill", this.getNodeColor({ type: 'card', quality: level.quality }));

            group.append("text")
                .attr("class", "text-[10px]")
                .attr("x", 15)
                .attr("y", 4)
                .attr("fill", this.getTextColor())
                .text(level.label);
        });

        // Update force simulation
        this.simulation
            .nodes(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-400))
            .force("center", d3.forceCenter(this.width / 2, this.height / 3))
            .force("collision", d3.forceCollide().radius(d => d.radius + 5))
            .force("y", d3.forceY(d => d.type === 'chapter' ? this.height / 4 : this.height / 2).strength(0.1))
            .on("tick", () => {
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);

                nodeGroup
                    .attr("transform", d => `translate(${d.x},${d.y})`);
            });

        this.simulation.alpha(1).restart();

        nodeGroup.call(d3.drag()
            .on("start", this.dragstarted.bind(this))
            .on("drag", this.dragged.bind(this))
            .on("end", this.dragended.bind(this)));
    }

    // Updated node coloring method
    getNodeColor(d) {
        if (d.type === 'chapter') {
            return 'rgba(75, 85, 99, 0.7)'; // Gray for chapter nodes
        }
        
        // Blue color scale for cards
        const blueScale = {
            0: 'rgba(219, 234, 254, 0.8)',  // lightest blue
            1: 'rgba(147, 197, 253, 0.8)',
            2: 'rgba(59, 130, 246, 0.8)',
            3: 'rgba(29, 78, 216, 0.8)',
            4: 'rgba(30, 58, 138, 0.8)'     // darkest blue
        };
        
        return blueScale[d.quality] || blueScale[0];
    }

    // Helper methods
    dragstarted(event, d) {
        if (!event.active) this.simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    dragended(event, d) {
        if (!event.active) this.simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    getQualityColor(quality, opacity = 1) {
        const colors = {
            0: `rgba(239, 68, 68, ${opacity})`,   // red
            1: `rgba(249, 115, 22, ${opacity})`,  // orange
            2: `rgba(234, 179, 8, ${opacity})`,   // yellow
            3: `rgba(59, 130, 246, ${opacity})`,  // blue
            4: `rgba(99, 102, 241, ${opacity})`,  // indigo
            5: `rgba(34, 197, 94, ${opacity})`    // green
        };
        return colors[quality] || colors[0];
    }

    getQualityLabel(quality) {
        const labels = {
            0: 'Not Started',
            1: 'Just Started',
            2: 'Learning',
            3: 'Reviewing',
            4: 'Almost Learned',
            5: 'Learned'
        };
        return labels[quality] || 'Not Started';
    }

    // Add helper method for text truncation
    truncateText(text, maxLength) {
        return text.length > maxLength ? text.substring(0, maxLength - 3) + '...' : text;
    }

    // Add method to update review count for today
    updateTodayReview(updatedHistory) {
        if (this.currentView === 'heatmap') {
            const today = new Date().toDateString();
            const todayCount = (updatedHistory && updatedHistory[today.split('T')[0]]) || 0;
            
            const todayCell = this.svg.select(`.day[data-date="${today}"]`);
            if (todayCell.node()) {
                todayCell
                    .attr('data-count', todayCount)
                    .transition()
                    .duration(300)
                    .attr('fill', this.colorScale(todayCount));
            }
        }
    }

    // Add helper method to get chapter title
    getChapterTitle(chapter) {
        const chapterSets = JSON.parse(localStorage.getItem('chapter_card_sets') || '{}');
        return `Deck ${chapter}`;
    }

    // Add new specific method for heatmap data
    updateHeatmapData(reviewData) {
        this.reviewData = reviewData;
        if (this.currentView === 'heatmap') {
            this.drawHeatmap();
        }
    }
} 