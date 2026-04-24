import * as d3 from 'd3';
import { enableTooltip } from '../tooltip/tooltip.js';
import { showPopover } from '../../libs/utils/utils.js';
import { generateProgressReport } from '../progressReport/progressReport.js';
import { getDBInstance } from '../../libs/utils/indexDb.js';
import { getAllChapterMapEntries } from '../../libs/utils/tocExtractor.js';

let modalInstance = null;

export class KnowledgeGraph {
    constructor() {
        this.width = window.innerWidth * 0.9;
        this.height = window.innerHeight * 0.8;
        this.nodes = [];
        this.links = [];
        this.simulation = null;
        this.currentZoomLevel = 0;
        this.tieredData = {};
        this.selectedNodes = new Set();
        
        // Zoom level configuration - optimized for performance
        this.ZOOM_LEVEL_CONFIG = {
            0: { // Bird's Eye - Only H1s
                maxLevel: 1,
                description: "Major Topics",
                nodeSize: "large",
                showDetails: false,
                maxNodes: 50
            },
            1: { // Medium Overview - H1s + H2s  
                maxLevel: 2,
                description: "Topics + Sections",
                nodeSize: "medium",
                showDetails: "minimal",
                maxNodes: 200
            },
            2: { // Detailed View - H1s + H2s + H3s
                maxLevel: 3,
                description: "Topics + Sections + Subsections", 
                nodeSize: "small",
                showDetails: "moderate",
                maxNodes: 500
            },
            3: { // Deep Dive - Up to H4s
                maxLevel: 4,
                description: "Full Content Structure",
                nodeSize: "tiny",
                showDetails: "full",
                maxNodes: 1000
            }
        };
        
        // Performance settings
        this.MAX_NODES_PER_LEVEL = 1000;
        this.VIEWPORT_BUFFER = 200; // pixels outside viewport to render
        
        this.colors = {
            chapter: '#2c5282',
            section: '#4299e1',
            selected: '#63b3ed',
            chapterLink: '#cbd5e0',
            linkGreen: '#48bb78',
            linkYellow: '#ecc94b',
            linkRed: '#f56565',
            linkBlue: '#4299e1',
            // Domain-based colors
            domainColors: {
                'example.com': '#2c5282',
                'github.com': '#24292e',
                'stackoverflow.com': '#f48024',
                'default': '#4299e1'
            }
        };
    }
    clearVisualization(container) {
        d3.select(container).selectAll('svg').remove();
    }

    async getQuizScores() {
        try {
            const dbManager = await getDBInstance();
            if (!dbManager) {
                throw new Error('Database not initialized');
            }
            
            const scores = {};
            const quizScores = await dbManager.getAll('quizHighScores');
            
            quizScores.forEach(data => {
                scores[data.quizTitle] = data.percentageScore;
            });
            
            return scores;
        } catch (error) {
            console.error('Error getting quiz scores:', error);
            return {};
        }
    }

    async getChapterSummaries() {
        try {
            const dbManager = await getDBInstance();
            if (!dbManager) {
                throw new Error('Database not initialized');
            }
            
            const summaries = {};
            const allSummaries = await dbManager.getAll('chapterSummaries');
            
            allSummaries.forEach(summary => {
                summaries[summary.chapterId] = summary;
            });
            
            return summaries;
        } catch (error) {
            console.error('Error getting chapter summaries:', error);
            return {};
        }
    }

    async initData() {
        const savedSelectedNodes = localStorage.getItem('knowledgeGraphSelectedNodes');
        if (savedSelectedNodes) {
            this.selectedNodes = new Set(JSON.parse(savedSelectedNodes));
        }
        
        
        // Build hierarchical tree data
        this.treeData = await this.buildHierarchicalTreeData();
        
    }

    async buildHierarchicalTreeData(showAllPages = false, forListView = false) {
        
        // Get all TOC entries from chapterMap
        const allTOCEntries = await getAllChapterMapEntries();
        
        // 🔍 DETAILED LOGGING: Log all chapterMap entries and their TOC content
        // Uncomment below for detailed analysis
        /*
        console.group('📚 CHAPTERMAP DATA ANALYSIS');
        allTOCEntries.forEach((entry, index) => {
            console.log(`📖 Entry ${index + 1}:`, {
                url: entry.url,
                title: entry.title,
                tocDataCount: entry.tocData ? entry.tocData.length : 0,
                lastUpdated: entry.lastUpdated
            });
            
            if (entry.tocData && entry.tocData.length > 0) {
                console.log(`📋 TOC Content for "${entry.title}":`);
                entry.tocData.forEach((tocItem, tocIndex) => {
                    console.log(`  ${tocIndex + 1}. [Level ${tocItem.level}] ${tocItem.text}`);
                    console.log(`     Content: "${tocItem.content ? tocItem.content.substring(0, 100) + '...' : 'NO CONTENT'}"`);
                    console.log(`     Position: ${tocItem.position}, Index: ${tocItem.index}`);
                });
            } else {
                console.log(`❌ No TOC data found for "${entry.title}"`);
            }
        });
        console.groupEnd();
        */
        
        // Create root node
        const root = {
            name: "Knowledge Graph",
            children: [],
            level: 0,
            id: "root",
            pageUrl: "",
            pageTitle: "",
            domain: "",
            position: 0,
            isExpanded: false,
            content: '' // Root node doesn't have content
        };
        
        // For list view, show ALL pages and ALL levels
        // For radial graph, limit for performance
        const MAX_PAGES = (showAllPages || forListView) ? allTOCEntries.length : 6;
        allTOCEntries.slice(0, MAX_PAGES).forEach(pageEntry => {
        // LOG: Check pageEntry structure and tocData
        console.log('🔍 PROCESSING PAGE ENTRY:', pageEntry.title);
        console.log('  - tocData length:', pageEntry.tocData ? pageEntry.tocData.length : 0);
        if (pageEntry.tocData && pageEntry.tocData[0]) {
            console.log('  - First tocData item:', pageEntry.tocData[0]);
            console.log('  - Has content:', 'content' in pageEntry.tocData[0]);
        }
            
            const pageNode = {
                name: pageEntry.title,
                children: [],
                level: 0,
                id: `page-${pageEntry.url}`,
                pageUrl: pageEntry.url,
                pageTitle: pageEntry.title,
                domain: pageEntry.domain,
                position: 0,
                isExpanded: false,
                content: pageEntry.content || '' // Include page content if available
            };
            
            // Build hierarchy for this page
            this.buildPageHierarchy(pageEntry.tocData, pageNode, forListView);
            
            if (pageNode.children.length > 0) {
                root.children.push(pageNode);
            }
        });

        // Add truncation node if there are more pages and we're not showing all pages
        if (!showAllPages && allTOCEntries.length > MAX_PAGES) {
            const truncatedPageNode = {
                name: `... ${allTOCEntries.length - MAX_PAGES} more pages`,
                children: [],
                level: 0,
                id: 'truncated-pages',
                pageUrl: '',
                pageTitle: '',
                domain: '',
                position: 0,
                isExpanded: false,
                isTruncated: true,
                content: '' // Truncated nodes don't have content
            };
            root.children.push(truncatedPageNode);
        }
        
        return root;
    }

    buildPageHierarchy(tocData, parentNode, forListView = false) {
        const stack = [{ node: parentNode, level: 0 }];
        // For list view, show ALL levels; for radial graph, limit to 2 levels for performance
        const MAX_LEVEL = forListView ? 6 : 2; // Show H1-H6 for list view, H1-H2 for radial graph
        const MAX_CHILDREN = forListView ? 1000 : 5; // Show all children for list view, limit for radial graph
        
        // Filter out duplicate headings (like "Table of contents" that appears in nav)
        const filteredTocData = this.filterDuplicateHeadings(tocData);
        
        filteredTocData.forEach(heading => {
            // Skip if beyond max level
            if (heading.level > MAX_LEVEL) {
                return;
            }
            
            // LOG: Check if heading has content
            console.log('🔍 BUILDING HEADING NODE:', heading.text);
            console.log('  - Has content:', 'content' in heading);
            console.log('  - Content length:', heading.content ? heading.content.length : 0);
            
            const headingNode = {
                name: heading.text,
                children: [],
                level: heading.level,
                id: `${parentNode.pageUrl}#${heading.id}`,
                pageUrl: parentNode.pageUrl,
                pageTitle: parentNode.pageTitle,
                domain: parentNode.domain,
                position: heading.position,
                isExpanded: false,
                isTruncated: false,
                content: heading.content || '' // Include content from tocData
            };
            
            // LOG: Verify content was added to headingNode
            console.log('✅ HEADING NODE CREATED:', headingNode.name, 'Content length:', headingNode.content ? headingNode.content.length : 0);
            
            // Find the correct parent in the stack
            while (stack.length > 1 && stack[stack.length - 1].level >= heading.level) {
                stack.pop();
            }
            
            const currentParent = stack[stack.length - 1].node;
            
            // Check if we need to truncate
            if (currentParent.children.length >= MAX_CHILDREN) {
                // Add truncation node if not already present
                if (!currentParent.children.some(child => child.isTruncated)) {
                    const truncatedNode = {
                        name: `... ${this.countRemainingChildren(tocData, heading.position)} more`,
                        children: [],
                        level: heading.level,
                        id: `${parentNode.pageUrl}#truncated-${currentParent.children.length}`,
                        pageUrl: parentNode.pageUrl,
                        pageTitle: parentNode.pageTitle,
                        domain: parentNode.domain,
                        position: heading.position,
                        isExpanded: false,
                        isTruncated: true,
                        content: '' // Truncated nodes don't have content
                    };
                    currentParent.children.push(truncatedNode);
                }
                return; // Skip adding this node
            }
            
            currentParent.children.push(headingNode);
            
            // Add to stack for potential children
            stack.push({ node: headingNode, level: heading.level });
        });
    }

    filterDuplicateHeadings(tocData) {
        // Common navigation/duplicate headings to filter out
        const duplicatePatterns = [
            /^table\s+of\s+contents$/i,
            /^toc$/i,
            /^navigation$/i,
            /^nav$/i,
            /^menu$/i,
            /^sidebar$/i,
            /^footer$/i,
            /^header$/i,
            /^skip\s+to\s+content$/i,
            /^skip\s+navigation$/i,
            /^breadcrumb/i,
            /^back\s+to\s+top$/i,
            /^scroll\s+to\s+top$/i,
            /^page\s+contents$/i,
            /^contents$/i,
            /^index$/i,
            /^site\s+map$/i,
            /^sitemap$/i
        ];
        
        // Track headings we've seen to detect exact duplicates
        const seenHeadings = new Set();
        const filteredData = [];
        
        for (const heading of tocData) {
            const headingText = heading.text.trim().toLowerCase();
            
            // Skip if it matches any duplicate pattern
            const isDuplicatePattern = duplicatePatterns.some(pattern => pattern.test(headingText));
            if (isDuplicatePattern) {
                console.log(`🚫 Filtering out duplicate pattern: "${heading.text}"`);
                continue;
            }
            
            // Skip if we've seen this exact heading before
            if (seenHeadings.has(headingText)) {
                console.log(`🚫 Filtering out duplicate heading: "${heading.text}"`);
                continue;
            }
            
            // Skip very short headings that are likely navigation elements
            if (headingText.length < 3) {
                console.log(`🚫 Filtering out very short heading: "${heading.text}"`);
                continue;
            }
            
            // Skip headings that are just numbers or symbols
            if (/^[\d\s\-_\.]+$/.test(headingText)) {
                console.log(`🚫 Filtering out numeric/symbol heading: "${heading.text}"`);
                continue;
            }
            
            seenHeadings.add(headingText);
            filteredData.push(heading);
        }
        
        return filteredData;
    }

    countRemainingChildren(tocData, currentPosition) {
        // Count how many more children would be added after truncation
        let count = 0;
        let foundCurrent = false;
        
        for (const heading of tocData) {
            if (heading.position === currentPosition) {
                foundCurrent = true;
                continue;
            }
            if (foundCurrent && heading.position > currentPosition) {
                count++;
            }
        }
        
        return count;
    }

    buildInitialLinks(nodes) {
        const links = [];
        
        // Build simple sequential links between H1 nodes
        const sortedNodes = nodes.sort((a, b) => a.position - b.position);
        
        for (let i = 0; i < sortedNodes.length - 1; i++) {
            const currentNode = sortedNodes[i];
            const nextNode = sortedNodes[i + 1];
            
            links.push({
                source: currentNode.id,
                target: nextNode.id,
                type: 'sequential',
                level: 1,
                weight: 1,
                color: '#cbd5e0'
            });
        }
        
        return links;
    }

    createNodeFromTOC(heading, pageEntry) {
        return {
            // Core identifiers
            id: `${pageEntry.url}#${heading.id}`,
            headingId: heading.id,
            pageUrl: pageEntry.url,
            pageTitle: pageEntry.title,
            domain: pageEntry.domain,
            
            // Content
            label: heading.text,
            level: heading.level,
            position: heading.position,
            index: heading.index,
            
            // Visual properties based on level
            size: this.calculateNodeSize(heading.level),
            color: this.calculateNodeColor(heading.level, pageEntry.domain),
            opacity: this.calculateNodeOpacity(heading.level),
            
            // Metadata
            isVisible: true,
            hasChildren: this.hasChildHeadings(heading, pageEntry.tocData),
            isExpanded: false,
            
            // Interaction properties
            clickable: true,
            expandable: heading.level < 6,
            
            // Position for layout
            x: null, // Will be set by D3 force simulation
            y: null,
            fx: null, // Fixed position if needed
            fy: null
        };
    }

    calculateNodeSize(level) {
        // Base sizes for each heading level
        const levelSizes = {
            1: 40,  // H1 - Largest
            2: 30,  // H2
            3: 20,  // H3
            4: 15,  // H4
            5: 12,  // H5
            6: 10   // H6 - Smallest
        };
        
        return levelSizes[level] || 10;
    }

    calculateNodeColor(level, domain) {
        // Simple color scheme based on heading level
        const levelColors = {
            1: '#1a365d', // Dark blue for H1s
            2: '#2c5282', // Darker blue for H2s
            3: '#3182ce', // Medium blue for H3s
            4: '#4299e1', // Light blue for H4s
            5: '#63b3ed', // Lighter blue for H5s
            6: '#90cdf4'  // Lightest blue for H6s
        };
        
        return levelColors[level] || '#4299e1';
    }

    calculateNodeOpacity(level) {
        // Higher levels (H1, H2) are more opaque
        const levelOpacities = {
            1: 1.0,   // H1 - Fully opaque
            2: 0.9,   // H2
            3: 0.8,   // H3
            4: 0.7,   // H4
            5: 0.6,   // H5
            6: 0.5    // H6 - More transparent
        };
        
        return levelOpacities[level] || 0.5;
    }

    hasChildHeadings(heading, allHeadings) {
        return allHeadings.some(h => 
            h.level > heading.level && 
            h.position > heading.position &&
            h.index > heading.index
        );
    }

    buildPageLinks(nodes, zoomLevel) {
        const links = [];
        const config = this.ZOOM_LEVEL_CONFIG[zoomLevel];
        
        // Sort nodes by position for sequential links
        const sortedNodes = nodes.sort((a, b) => a.position - b.position);
        
        // Build hierarchical links (parent-child)
        sortedNodes.forEach(node => {
            const parentNode = this.findParentNode(node, sortedNodes);
            if (parentNode) {
                links.push({
                    source: parentNode.id,
                    target: node.id,
                    type: 'hierarchical',
                    level: node.level,
                    weight: this.calculateLinkWeight(node.level, 'hierarchical'),
                    color: this.calculateLinkColor(node.level, 'hierarchical')
                });
            }
        });
        
        // Build sequential links (same level, adjacent)
        for (let i = 0; i < sortedNodes.length - 1; i++) {
            const currentNode = sortedNodes[i];
            const nextNode = sortedNodes[i + 1];
            
            // Only link if they're close in position and same level
            if (nextNode.position - currentNode.position < 5000 && // Within 5000px
                currentNode.level === nextNode.level) {
                links.push({
                    source: currentNode.id,
                    target: nextNode.id,
                    type: 'sequential',
                    level: currentNode.level,
                    weight: this.calculateLinkWeight(currentNode.level, 'sequential'),
                    color: this.calculateLinkColor(currentNode.level, 'sequential')
                });
            }
        }
        
        return links;
    }

    findParentNode(node, allNodes) {
        // Find the closest parent heading (lower level number)
        const candidates = allNodes.filter(n => 
            n.level < node.level && 
            n.position < node.position &&
            n.pageUrl === node.pageUrl
        );
        
        if (candidates.length === 0) return null;
        
        // Return the closest parent (highest level number among candidates)
        return candidates.reduce((closest, current) => 
            current.level > closest.level ? current : closest
        );
    }

    calculateLinkWeight(level, type) {
        const baseWeights = {
            hierarchical: 2,
            sequential: 1,
            'cross-page': 0.5
        };
        
        const levelMultipliers = {
            1: 1.0,
            2: 0.8,
            3: 0.6,
            4: 0.4,
            5: 0.3,
            6: 0.2
        };
        
        return baseWeights[type] * levelMultipliers[level];
    }

    calculateLinkColor(level, type) {
        const typeColors = {
            hierarchical: '#4299e1',
            sequential: '#cbd5e0',
            'cross-page': '#e2e8f0'
        };
        
        return typeColors[type] || '#cbd5e0';
    }

    buildCrossPageLinks(allNodes, zoomLevel) {
        const links = [];
        
        // Group nodes by level
        const nodesByLevel = {};
        allNodes.forEach(node => {
            if (!nodesByLevel[node.level]) {
                nodesByLevel[node.level] = [];
            }
            nodesByLevel[node.level].push(node);
        });
        
        // Create cross-page links for each level
        Object.entries(nodesByLevel).forEach(([level, nodes]) => {
            // Sort by domain and position
            const sortedNodes = nodes.sort((a, b) => {
                if (a.domain !== b.domain) {
                    return a.domain.localeCompare(b.domain);
                }
                return a.position - b.position;
            });
            
            // Link H1s across pages (major topic flow)
            if (parseInt(level) === 1) {
                for (let i = 0; i < sortedNodes.length - 1; i++) {
                    const currentNode = sortedNodes[i];
                    const nextNode = sortedNodes[i + 1];
                    
                    // Only link if different pages
                    if (currentNode.pageUrl !== nextNode.pageUrl) {
                        links.push({
                            source: currentNode.id,
                            target: nextNode.id,
                            type: 'cross-page',
                            level: parseInt(level),
                            weight: 0.5, // Lighter weight for cross-page links
                            color: '#cbd5e0' // Light gray for cross-page
                        });
                    }
                }
            }
        });
        
        return links;
    }

    calculateZoomLevel(scale) {
        // More gradual zoom level changes for better area expansion
        if (scale < 0.8) return 0;      // Bird's eye
        if (scale < 1.5) return 1;      // Medium detail
        if (scale < 3.0) return 2;      // Detailed
        if (scale < 6.0) return 3;      // Very detailed
        return 3;                      // Cap at level 3 for performance
    }

    updateGraphForZoomLevel(newZoomLevel) {
        const config = this.ZOOM_LEVEL_CONFIG[newZoomLevel];
        const data = this.tieredData[newZoomLevel];
        
        console.log(`🔄 Updating graph to zoom level ${newZoomLevel}: ${config.description}`);
        
        // Get visible nodes (limited for performance)
        const visibleNodes = this.getVisibleNodes(data.nodes);
        const visibleLinks = this.getVisibleLinks(data.links, visibleNodes);
        
        console.log(`📊 Rendering ${visibleNodes.length} visible nodes out of ${data.nodes.length} total`);
        
        // Clear existing nodes and links first
        this.nodeGroup.selectAll('circle').remove();
        this.linkGroup.selectAll('line').remove();
        
        // Add new links first (so they appear behind nodes)
        this.linkGroup.selectAll('line')
            .data(visibleLinks, d => `${d.source.id || d.source}-${d.target.id || d.target}`)
            .join('line')
            .attr('class', 'link')
            .attr('stroke-width', d => Math.max(2, d.weight * 2))
            .attr('stroke', d => d.color || '#2d3748')
            .attr('stroke-opacity', 0.8);
        
        // Add new nodes
        this.nodeGroup.selectAll('circle')
            .data(visibleNodes, d => d.id)
            .join('circle')
            .attr('class', 'node')
            .attr('r', d => d.size)
            .attr('fill', d => {
                if (this.selectedNodes.has(d.id)) {
                    return this.colors.selected;
                }
                return d.color;
            })
            .attr('stroke', d => {
                if (this.selectedNodes.has(d.id)) {
                    return '#ffd700';
                }
                return 'none';
            })
            .attr('stroke-width', d => {
                if (this.selectedNodes.has(d.id)) {
                    return 3;
                }
                return 0;
            })
            .style('opacity', d => d.opacity)
            .on('click', (event, d) => {
                this.handleNodeClick(event, d);
            });
        
        // Update simulation with visible data only
        this.simulation.nodes(visibleNodes);
        this.simulation.force('link').links(visibleLinks);
        this.simulation.alpha(0.3).restart();
        
        // Update zoom level indicator
        this.updateZoomLevelIndicator(newZoomLevel, config);
    }

    updateLinksForZoomLevel(zoomLevel, visibleLinks = null) {
        const links = visibleLinks || this.tieredData[zoomLevel].links;
        
        // Update links
        const linkUpdate = this.linkGroup.selectAll('line')
            .data(links, d => `${d.source.id || d.source}-${d.target.id || d.target}`);
        
        // Remove old links
        linkUpdate.exit()
            .transition()
            .duration(300)
            .style('opacity', 0)
            .remove();
        
        // Add new links
        const linkEnter = linkUpdate.enter()
            .append('line')
            .attr('class', 'link')
            .style('opacity', 0);
        
        // Update all links
        const linkMerge = linkEnter.merge(linkUpdate);
        
        linkMerge
            .transition()
            .duration(300)
            .style('stroke', d => d.color)
            .style('stroke-width', d => d.weight)
            .style('opacity', 0.6);
    }

    getVisibleNodes(allNodes) {
        // For performance, limit nodes per zoom level
        const config = this.ZOOM_LEVEL_CONFIG[this.currentZoomLevel];
        const maxNodes = config.maxNodes || 200;
        
        if (allNodes.length <= maxNodes) {
            return allNodes;
        }
        
        // Prioritize nodes by level (H1s first, then H2s, etc.)
        const sortedNodes = allNodes.sort((a, b) => {
            // First sort by level (lower level = higher priority)
            if (a.level !== b.level) {
                return a.level - b.level;
            }
            // Then by position for same level
            return a.position - b.position;
        });
        
        // Return only the most important nodes
        return sortedNodes.slice(0, maxNodes);
    }

    getVisibleLinks(allLinks, visibleNodes) {
        if (!visibleNodes || visibleNodes.length <= 100) {
            return allLinks; // Return all links if small dataset
        }
        
        const visibleNodeIds = new Set(visibleNodes.map(n => n.id));
        
        // Only include links where both source and target are visible
        return allLinks.filter(link => {
            const sourceId = link.source.id || link.source;
            const targetId = link.target.id || link.target;
            return visibleNodeIds.has(sourceId) && visibleNodeIds.has(targetId);
        });
    }


    showLoader() {
        if (this.loader) return;
        
        this.loader = document.createElement('div');
        this.loader.style.cssText = `
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(255, 255, 255, 0.9);
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            z-index: 1000;
            display: flex;
            align-items: center;
            gap: 10px;
        `;
        
        const spinner = document.createElement('div');
        spinner.style.cssText = `
            width: 20px;
            height: 20px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #4299e1;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        `;
        
        const text = document.createElement('span');
        text.textContent = 'Loading...';
        text.style.cssText = `
            font-size: 14px;
            color: #2d3748;
        `;
        
        this.loader.appendChild(spinner);
        this.loader.appendChild(text);
        
        // Add spinner animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        `;
        this.loader.appendChild(style);
        
        this.container.appendChild(this.loader);
    }

    hideLoader() {
        if (this.loader) {
            this.loader.remove();
            this.loader = null;
        }
    }

    updateZoomLevelIndicator(zoomLevel, config) {
        let indicator = this.container.querySelector('.zoom-level-indicator');
        
        if (!indicator) {
            indicator = document.createElement('div');
            indicator.className = 'zoom-level-indicator';
            indicator.style.cssText = `
                position: absolute;
                top: 10px;
                right: 10px;
                background: rgba(255,255,255,0.9);
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 12px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                z-index: 10;
            `;
            this.container.appendChild(indicator);
        }
        
        indicator.innerHTML = `
            <div><strong>Detail Level ${zoomLevel}</strong></div>
            <div>${config.description}</div>
            <div>Showing H1-H${config.maxLevel}</div>
            <div>${this.tieredData[zoomLevel].metadata.totalNodes} nodes</div>
        `;
    }

    createVisualization(container) {
        console.log("createVisualization container:", container)
        // Store the whole container reference for shadow DOM access
        this.wholeContainer = container;
        
        // Create in-memory dictionary for sidebar nodes
        this.sidebarNodes = new Map(); // key: nodeId, value: node data
        
        // Define updateButtonState at the beginning of createVisualization
        const updateButtonState = () => {
            const generateButton = container.querySelector('#generate-summative-btn');
            if (!generateButton) return;
            
            const hasSelectedNodes = this.selectedNodes.size > 0;
            generateButton.disabled = !hasSelectedNodes;
            generateButton.style.backgroundColor = hasSelectedNodes ? '#4299e1' : '#CBD5E0';
            generateButton.style.cursor = hasSelectedNodes ? 'pointer' : 'not-allowed';
            
            // Save selected nodes to localStorage
            localStorage.setItem('knowledgeGraphSelectedNodes', 
                JSON.stringify(Array.from(this.selectedNodes)));
        };

        // Create wrapper div for graph, sidebar, and right panel
        const wrapper = document.createElement('div');
        wrapper.style.cssText = `
            display: flex;
            width: 100%;
            height: 100%;
            position: relative;
        `;
        container.appendChild(wrapper);

        // Create graph container with explicit dimensions
        const graphContainer = document.createElement('div');
        graphContainer.style.cssText = `
            flex: 1;
            position: relative;
            min-height: 500px; // Ensure minimum height
            height: 100%;
        `;
        wrapper.appendChild(graphContainer);

        // Right panel for progress analysis will be created by the existing HTML structure

        // Create sidebar
        const sidebar = document.createElement('div');
        sidebar.className = 'sidebar'; // Add class name for identification
        sidebar.style.cssText = `
            width: 250px;
            background: #f7fafc;
            padding: 20px;
            border-left: 1px solid #e2e8f0;
            position: relative;
            height: 100%;
            display: flex;
            flex-direction: column;
        `;
        wrapper.appendChild(sidebar);
        
        // Add header section to sidebar
        const sidebarHeader = document.createElement('div');
        sidebarHeader.className = 'sidebar-header';
        sidebarHeader.style.cssText = `
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #e2e8f0;
        `;
        
        const headerTitle = document.createElement('div');
        headerTitle.textContent = 'Selected Components';
        headerTitle.style.cssText = `
            font-weight: bold;
            font-size: 14px;
            color: #2d3748;
            margin-bottom: 8px;
        `;
        
        const explanatoryText = document.createElement('div');
        explanatoryText.textContent = 'Selecting nodes will display here, allowing you to create custom quiz';
        explanatoryText.style.cssText = `
            font-size: 11px;
            color: #718096;
            line-height: 1.4;
            margin-bottom: 8px;
        `;
        
        sidebarHeader.appendChild(headerTitle);
        sidebarHeader.appendChild(explanatoryText);
        sidebar.appendChild(sidebarHeader);
        
        // Add scrollable container for quiz items
        const quizItemsContainer = document.createElement('div');
        quizItemsContainer.className = 'quiz-items-container';
        quizItemsContainer.style.cssText = `
            flex: 1;
            overflow-y: auto;
            margin-bottom: 80px; /* Space for fixed button container */
            max-height: calc(100% - 120px); /* Adjust based on header and button space */
        `;
        sidebar.appendChild(quizItemsContainer);
    
        // Add button container fixed to bottom
        const buttonContainer = document.createElement('div');
        buttonContainer.style.cssText = `
            position: absolute;
            bottom: 20px;
            left: 20px;
            right: 20px;
            padding-top: 10px;
            border-top: 1px solid #e2e8f0;
            background: #f7fafc;
        `;
    
        const generateButton = document.createElement('button');
        generateButton.id = 'generate-summative-btn';
        generateButton.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="16" y1="13" x2="8" y2="13"></line>
                <line x1="16" y1="17" x2="8" y2="17"></line>
                <polyline points="10 9 9 9 8 9"></polyline>
            </svg>
            Generate Custom Quiz
        `;
        generateButton.style.cssText = `
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            padding: 8px 16px;
            background-color: #CBD5E0;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            cursor: not-allowed;
            transition: background-color 0.2s;
            gap: 8px;
        `;
        generateButton.disabled = true;
        enableTooltip(generateButton, "Select nodes to generate custom quiz", container);
    
        generateButton.addEventListener('mouseover', () => {
            generateButton.style.backgroundColor = '#3182ce';
        });
    
        generateButton.addEventListener('mouseout', () => {
            generateButton.style.backgroundColor = '#4299e1';
        });
    
        generateButton.addEventListener('click', async () => {
            console.log('Generate Custom Quiz clicked, extracting data from sidebar DOM...');
            
            // Extract data directly from sidebar DOM elements - use wholeContainer for shadow DOM access
            const sidebar = this.wholeContainer.querySelector('.sidebar');
            if (!sidebar) {
                console.error('Sidebar not found in wholeContainer');
                alert('Sidebar not found. Please try again.');
                return;
            }
            
            const quizItems = sidebar.querySelectorAll('[data-quiz-id]');
            console.log(`Found ${quizItems.length} quiz items in sidebar`);
            
            if (quizItems.length === 0) {
                console.error('No quiz items found in sidebar');
                alert('No items selected. Please select some nodes and try again.');
                return;
            }
            
            // Extract data from each quiz item
            const selectedNodesData = Array.from(quizItems).map(quizItem => {
                const nodeId = quizItem.dataset.quizId;
                const content = quizItem.dataset.content || '';
                const pageUrl = quizItem.dataset.pageUrl || '';
                const domain = quizItem.dataset.domain || '';
                const level = parseInt(quizItem.dataset.level) || 0;
                const position = parseFloat(quizItem.dataset.position) || 0;
                
                // Get the display name from the DOM
                const nameElement = quizItem.querySelector('div[style*="font-weight: 500"]');
                const label = nameElement ? nameElement.textContent.trim() : 'Unknown';
                
                console.log('Extracted node data:', {
                    id: nodeId,
                    label: label,
                    content: content.substring(0, 100) + '...', // Log first 100 chars
                    pageUrl: pageUrl,
                    domain: domain,
                    level: level,
                    position: position
                });
                
                return {
                    id: nodeId,
                    label: label,
                    content: content,
                    pageUrl: pageUrl,
                    domain: domain,
                    level: level,
                    position: position
                };
            });

            console.log('Selected nodes data extracted from sidebar:', selectedNodesData);

            if (selectedNodesData.length === 0) {
                console.error('No valid selected nodes found');
                alert('No valid nodes selected. Please select some nodes and try again.');
                return;
            }

            // Create title string from selected node labels
            const title = selectedNodesData
                .map(node => node.label)
                .join(' | ');

            // Create content with source mapping for reference tooltips
            const contentWithSources = selectedNodesData.map((node, index) => ({
                sourceId: `source-${index}`,
                label: node.label,
                content: node.content || `Section: ${node.label}\nContent: Please visit this section to generate content.`,
                pageUrl: node.pageUrl,
                domain: node.domain,
                level: node.level,
                position: node.position
            }));

            // Combine all content into one text with enhanced formatting
            const quizPromptText = contentWithSources
                .map(source => `## ${source.label}\n\n${source.content}`)
                .join('\n\n---\n\n');

            console.log('Generated quiz prompt text:', quizPromptText);
            console.log('Content with sources:', contentWithSources);

            const event = new CustomEvent('aiActionCompleted', {
                detail: {
                    type: "summative",
                    title: title, // Add the combined title
                    text: quizPromptText, // Add the combined content for quiz generation
                    selectedNodesData: selectedNodesData, // Keep the original data if needed
                    contentWithSources: contentWithSources // NEW: Preserve source mapping for tooltips
                },
                bubbles: true,
                composed: true
            });
            container.dispatchEvent(event);

            // Close the modal after dispatching the event
            const modal = container.closest('.modal') || container.parentElement;
            if (modal) {
                modal.style.display = 'none';
                console.log('Modal closed after custom quiz generation');
            }
        });
    
        buttonContainer.appendChild(generateButton);

        // Add the new Analyze Progress button
        const analyzeButton = document.createElement('button');
        analyzeButton.id = 'analyze-progress-btn';
        analyzeButton.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path>
                <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path>
            </svg>
            Analyze my Progress
        `;
        analyzeButton.style.cssText = `
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            padding: 8px 16px;
            background-color: #4299e1;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: background-color 0.2s;
            gap: 8px;
            margin-top: 8px;
        `;

        analyzeButton.addEventListener('mouseover', () => {
            analyzeButton.style.backgroundColor = '#3182ce';
        });

        analyzeButton.addEventListener('mouseout', () => {
            analyzeButton.style.backgroundColor = '#4299e1';
        });

        analyzeButton.addEventListener('click', async () => {
            console.log("analyze button clicked");

            let progressReport;
            try {
                progressReport = await generateProgressReport();
                console.log("Full progress report data:", progressReport); // Log full report
                
                if (progressReport.success) {
                    console.log('Progress Report Text:', progressReport.report);
                    console.log('Progress Report Charts:', {
                        xyChart: progressReport.charts?.xyChart,
                        quadrantChart: progressReport.charts?.quadrantChart
                    });
                    
                    // Verify chart data before dispatching
                    if (!progressReport.charts?.xyChart || !progressReport.charts?.quadrantChart) {
                        console.warn('Missing chart data:', {
                            hasXYChart: !!progressReport.charts?.xyChart,
                            hasQuadrantChart: !!progressReport.charts?.quadrantChart
                        });
                    }

                    container.dispatchEvent(new CustomEvent('aiActionCompleted', {
                        bubbles: true,
                        composed: true,
                        detail: {
                            text: progressReport.report,
                            type: "progress_report",
                            xyChart: progressReport.charts?.xyChart || '',
                            quadrantChart: progressReport.charts?.quadrantChart || ''
                        }
                    }));

                } else {
                    console.log('Progress Report Status:', progressReport.message);
                }
            } catch (error) {
                console.error('Error generating progress report:', error);
                return;
            }

            const modal = container.closest('.modal') || container.parentElement;
            if (modal) {
                modal.style.display = 'none';
            }
        });

        enableTooltip(analyzeButton, "View your learning progress analysis", container);
        buttonContainer.appendChild(analyzeButton);
        sidebar.appendChild(buttonContainer);

        // Update dimensions based on actual container size
        const containerRect = graphContainer.getBoundingClientRect();
        this.width = containerRect.width;
        this.height = containerRect.height;

        // Store the graph container reference
        this.graphContainer = graphContainer;
        
        // Add view toggle switch
        this.addViewToggle(graphContainer);
        
        // Create radial tree visualization
        this.createRadialTree(graphContainer);

        // Add legend with detail level buttons
        const legend = document.createElement('div');
        legend.style.cssText = `
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(255, 255, 255, 0.9);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            z-index: 2;
            min-width: 200px;
        `;

        const legendTitle = document.createElement('h3');
        legendTitle.textContent = 'Knowledge Graph';
        legendTitle.style.cssText = `
            margin: 0 0 15px 0;
            font-size: 14px;
            color: #2d3748;
        `;
        legend.appendChild(legendTitle);



        const legendItems = [
            { color: '#4299e1', label: 'Pages & Topics (clickable)' },
            { color: '#999', label: 'Sections & Subsections' },
            { color: '#ff6b6b', label: 'Truncated (too many items)' },
            { color: '#ffd700', label: 'Selected for Quiz' },
            { color: '#555', label: 'Tree Connections' }
        ];

        legendItems.forEach(item => {
            const itemDiv = document.createElement('div');
            itemDiv.style.cssText = `
                display: flex;
                align-items: center;
                margin-bottom: 5px;
            `;

            const colorBox = document.createElement('div');
            colorBox.style.cssText = `
                width: 15px;
                height: 15px;
                background: ${item.color};
                margin-right: 8px;
                border-radius: 3px;
            `;

            const label = document.createElement('span');
            label.textContent = item.label;
            label.style.cssText = `
                font-size: 12px;
                color: #4a5568;
            `;

            itemDiv.appendChild(colorBox);
            itemDiv.appendChild(label);
            legend.appendChild(itemDiv);
        });

        graphContainer.appendChild(legend);

        // After adding the legend to graphContainer, add the explanation card
        const explanation = document.createElement('div');
        explanation.style.cssText = `
            position: absolute;
            top: ${legend.offsetHeight + 40}px;
            left: 20px;
            background: rgba(255, 255, 255, 0.9);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            z-index: 2;
            max-width: 300px;
            font-family: monospace;
            font-size: 12px;
            line-height: 1.4;
        `;

        explanation.innerHTML = `
            <h3 style="margin: 0 0 10px 0; font-size: 14px; color: #2d3748;">Radial Knowledge Tree</h3>
            <p style="margin: 0 0 8px 0; color: #4a5568;">
                This radial tree shows your learning structure:<br>
                • Center = Root of knowledge<br>
                • Branches = Pages & Topics<br>
                • Leaves = Sections & Details
            </p>
            <p style="margin: 0 0 8px 0; color: #4a5568;">
                Interactive Features:<br>
                • Click blue nodes to expand<br>
                • Click gray nodes to select<br>
                • Generate quizzes from selections
            </p>
            <p style="margin: 0; color: #4a5568;">
                Tip: Use zoom to explore<br>
                different branches of your<br>
                knowledge tree.
            </p>
        `;

        graphContainer.appendChild(explanation);
    }

    addViewToggle(container) {
        // Create toggle container
        const toggleContainer = document.createElement('div');
        toggleContainer.style.cssText = `
            position: absolute;
            top: 20px;
            right: 20px;
            z-index: 100;
            background: rgba(255, 255, 255, 0.9);
            padding: 8px 12px;
            border-radius: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            color: #2d3748;
        `;

        // Add labels
        const treeLabel = document.createElement('span');
        treeLabel.textContent = 'Tree';
        treeLabel.style.cssText = `
            color: #4299e1;
            font-weight: bold;
        `;

        const listLabel = document.createElement('span');
        listLabel.textContent = 'List';
        listLabel.style.cssText = `
            color: #718096;
        `;

        // Create toggle switch
        const toggleSwitch = document.createElement('div');
        toggleSwitch.style.cssText = `
            position: relative;
            width: 40px;
            height: 20px;
            background: #4299e1;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
        `;

        const toggleSlider = document.createElement('div');
        toggleSlider.style.cssText = `
            position: absolute;
            top: 2px;
            left: 2px;
            width: 16px;
            height: 16px;
            background: white;
            border-radius: 50%;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        `;

        toggleSwitch.appendChild(toggleSlider);
        toggleContainer.appendChild(treeLabel);
        toggleContainer.appendChild(toggleSwitch);
        toggleContainer.appendChild(listLabel);

        // Add click handler
        toggleSwitch.addEventListener('click', () => {
            this.toggleView();
        });

        container.appendChild(toggleContainer);
        
        // Store references
        this.toggleContainer = toggleContainer;
        this.toggleSwitch = toggleSwitch;
        this.toggleSlider = toggleSlider;
        this.treeLabel = treeLabel;
        this.listLabel = listLabel;
        this.currentView = 'tree'; // Start with tree view
    }

    toggleView() {
        if (this.currentView === 'tree') {
            this.switchToListView();
        } else {
            this.switchToTreeView();
        }
    }

    switchToListView() {
        console.log('Switching to list view...');
        this.currentView = 'list';
        
        // Update toggle appearance
        this.toggleSwitch.style.background = '#718096';
        this.toggleSlider.style.left = '22px';
        this.treeLabel.style.color = '#718096';
        this.listLabel.style.color = '#4299e1';
        this.listLabel.style.fontWeight = 'bold';
        this.treeLabel.style.fontWeight = 'normal';
        
        // Hide tree view
        const svg = this.graphContainer ? this.graphContainer.querySelector('svg') : this.container.querySelector('svg');
        if (svg) {
            console.log('Hiding SVG tree view');
            svg.style.display = 'none';
        } else {
            console.log('No SVG found to hide');
        }
        
        // Ensure tree data is available
        if (!this.treeData) {
            console.log('Tree data not available, attempting to rebuild...');
            this.initData().then(() => {
                this.createListView();
            }).catch(error => {
                console.error('Failed to initialize tree data:', error);
            });
        } else {
            console.log('Tree data available, creating list view...');
            // Show list view
            this.createListView();
        }
    }

    switchToTreeView() {
        this.currentView = 'tree';
        
        // Update toggle appearance
        this.toggleSwitch.style.background = '#4299e1';
        this.toggleSlider.style.left = '2px';
        this.treeLabel.style.color = '#4299e1';
        this.treeLabel.style.fontWeight = 'bold';
        this.listLabel.style.color = '#718096';
        this.listLabel.style.fontWeight = 'normal';
        
        // Hide list view
        const listContainer = this.container.querySelector('#list-view');
        if (listContainer) {
            listContainer.remove();
        }
        
        // Show tree view
        const svg = this.container.querySelector('svg');
        if (svg) {
            svg.style.display = 'block';
        }
    }

    createListView() {
        console.log('Creating list view...');
        
        // Prevent multiple simultaneous list view creations
        if (this.isCreatingListView) {
            console.log('List view creation already in progress, skipping...');
            return;
        }
        
        this.isCreatingListView = true;
        
        // Remove existing list view if any
        const existingList = this.container.querySelector('#list-view');
        if (existingList) {
            console.log('Removing existing list view');
            existingList.remove();
        }
        
        // Create list container
        const listContainer = document.createElement('div');
        listContainer.id = 'list-view';
        listContainer.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: white;
            overflow-y: auto;
            padding: 20px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            z-index: 10;
        `;
        
        // Rebuild tree data specifically for list view (showing ALL nodes)
        console.log('Rebuilding tree data for list view with all nodes...');
        this.buildHierarchicalTreeData(true, true).then(treeData => {
            this.treeData = treeData;
            this.renderListViewContent(listContainer);
            this.isCreatingListView = false; // Reset the guard
        }).catch(error => {
            console.error('Failed to build tree data for list view:', error);
            listContainer.innerHTML = '<div style="text-align: center; margin-top: 50px; color: #e53e3e;">Error loading data</div>';
            this.container.appendChild(listContainer);
            this.isCreatingListView = false; // Reset the guard
        });
    }

    renderListViewContent(listContainer) {
        // Check if tree data exists
        if (!this.treeData) {
            console.error('Tree data not available for list view');
            listContainer.innerHTML = '<div style="text-align: center; margin-top: 50px; color: #718096;">Loading...</div>';
            this.container.appendChild(listContainer);
            return;
        }
        
        console.log('Tree data available, building hierarchical list...');
        console.log('Tree data:', this.treeData);
        
        // Add info panel at the top
        const infoPanel = document.createElement('div');
        infoPanel.style.cssText = `
            background: #f7fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 20px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            line-height: 1.4;
            color: #2d3748;
        `;
        
        infoPanel.innerHTML = `
            <div style="font-weight: bold; margin-bottom: 8px; color: #2c5282;">KNOWLEDGE GRAPH LIST VIEW - ALL NODES</div>
            <div style="margin-bottom: 6px;">• Shows ALL pages and ALL heading levels (H1-H6)</div>
            <div style="margin-bottom: 6px;">• Click pages to expand topics</div>
            <div style="margin-bottom: 6px;">• Click H1 topics to expand sections</div>
            <div style="margin-bottom: 6px;">• Click H2+ sections to select for quiz</div>
            <div style="margin-bottom: 6px;">• Selected items appear in sidebar</div>
            <div style="margin-bottom: 6px;">• Use "Generate Custom Quiz" to create quiz</div>
            
            <div style="margin-top: 12px; margin-bottom: 8px; font-weight: bold; color: #2c5282;">ICON LEGEND:</div>
            <div style="display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 8px;">
                <div style="display: flex; align-items: center; gap: 4px;">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path><path d="M8 2v4"></path><path d="M12 2v4"></path><path d="M16 2v4"></path></svg>
                    <span style="font-size: 10px;">Page</span>
                </div>
                <div style="display: flex; align-items: center; gap: 4px;">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="9" y1="9" x2="15" y2="9"></line><line x1="9" y1="13" x2="15" y2="13"></line><line x1="9" y1="17" x2="13" y2="17"></line></svg>
                    <span style="font-size: 10px;">H1</span>
                </div>
                <div style="display: flex; align-items: center; gap: 4px;">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>
                    <span style="font-size: 10px;">H2</span>
                </div>
                <div style="display: flex; align-items: center; gap: 4px;">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 12l2 2 4-4"></path><path d="M21 12c-1 0-3-1-3-3s2-3 3-3 3 1 3 3-2 3-3 3"></path><path d="M3 12c1 0 3-1 3-3s-2-3-3-3-3 1-3 3 2 3 3 3"></path><path d="M12 3c0 1-1 3-3 3s-3-2-3-3 1-3 3-3 3 2 3 3"></path><path d="M12 21c0-1 1-3 3-3s3 2 3 3-1 3-3 3-3-2-3-3"></path></svg>
                    <span style="font-size: 10px;">H3</span>
                </div>
                <div style="display: flex; align-items: center; gap: 4px;">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="M12 1v6m0 6v6m11-7h-6m-6 0H1"></path></svg>
                    <span style="font-size: 10px;">H4</span>
                </div>
                <div style="display: flex; align-items: center; gap: 4px;">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"></path></svg>
                    <span style="font-size: 10px;">H5</span>
                </div>
                <div style="display: flex; align-items: center; gap: 4px;">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>
                    <span style="font-size: 10px;">H6</span>
                </div>
            </div>
            
            <div style="color: #718096; font-size: 11px; margin-top: 8px;">
                Duplicate navigation elements (like "Table of contents") are automatically filtered out
            </div>
        `;
        
        listContainer.appendChild(infoPanel);
        
        // Add column headers for the list items
        const listHeaders = document.createElement('div');
        listHeaders.style.cssText = `
            background: #f8f9fa;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 8px 12px;
            margin-bottom: 10px;
            font-size: 11px;
            color: #718096;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        `;
        
        listHeaders.innerHTML = `
            <span>Component</span>
            <span>Last Visited</span>
        `;
        
        listContainer.appendChild(listHeaders);
        
        // Create hierarchical list
        this.buildHierarchicalList(listContainer, this.treeData);
        
        console.log('Adding list container to DOM...');
        console.log('Container:', this.container);
        console.log('Graph container:', this.graphContainer);
        
        // Use the graph container instead of the main container
        if (this.graphContainer) {
            this.graphContainer.appendChild(listContainer);
            console.log('List view added to graph container');
        } else {
            this.container.appendChild(listContainer);
            console.log('List view added to main container');
        }
        console.log('List view created successfully');
    }

    buildHierarchicalList(container, data, level = 0) {
        if (!data) {
            console.log('buildHierarchicalList: No data provided');
            return;
        }
        
        console.log(`buildHierarchicalList: Building level ${level} with data:`, data);
        
        const item = document.createElement('div');
        item.className = 'list-item';
        item.style.cssText = `
            margin-left: ${level * 20}px;
            margin-bottom: 8px;
            padding: 8px 12px;
            border-radius: 6px;
            cursor: pointer;
            transition: background-color 0.2s;
            display: flex;
            align-items: center;
            justify-content: space-between;
        `;
        
        // Store URL data in data attributes for accessibility by other components
        // URLs are hidden from display to reduce clutter but remain accessible via data attributes
        if (data.pageUrl) {
            item.setAttribute('data-page-url', data.pageUrl);
        }
        if (data.id) {
            item.setAttribute('data-node-id', data.id);
        }
        if (data.domain) {
            item.setAttribute('data-domain', data.domain);
        }
        if (data.pageTitle) {
            item.setAttribute('data-page-title', data.pageTitle);
        }
        
        // Add click handler
        item.addEventListener('click', () => {
            this.handleListItemClick(data, item);
        });
        
        // Add hover effect
        item.addEventListener('mouseenter', () => {
            item.style.backgroundColor = '#f7fafc';
        });
        
        item.addEventListener('mouseleave', () => {
            item.style.backgroundColor = 'transparent';
        });
        
        // Create left content with icon and title
        const leftContent = document.createElement('div');
        leftContent.style.cssText = `
            display: flex;
            align-items: center;
            flex: 1;
        `;
        
        // Add icon based on level
        const icon = document.createElement('span');
        icon.style.cssText = `
            margin-right: 8px;
            font-size: 14px;
            display: inline-flex;
            align-items: center;
        `;
        icon.innerHTML = this.getLevelIcon(data.level, level);
        
        const title = document.createElement('span');
        title.textContent = data.name;
        title.style.cssText = `
            font-weight: ${level <= 1 ? 'bold' : 'normal'};
            font-size: ${level === 0 ? '16px' : level === 1 ? '14px' : '12px'};
            color: #2d3748;
        `;
        
        leftContent.appendChild(icon);
        leftContent.appendChild(title);
        
        // Create right content with date and expand icon
        const rightContent = document.createElement('div');
        rightContent.style.cssText = `
            display: flex;
            align-items: center;
            gap: 8px;
        `;
        
        // Add visit date if available
        if (data.pageUrl && data.pageUrl !== '') {
            const visitDate = document.createElement('span');
            visitDate.style.cssText = `
                font-size: 11px;
                color: #718096;
                font-family: 'Courier New', monospace;
            `;
            visitDate.textContent = `Last visited: ${this.getVisitDate(data.pageUrl)}`;
            rightContent.appendChild(visitDate);
        }
        
        // Add expand icon for items with children
        if (data.children && data.children.length > 0) {
            const expandIcon = document.createElement('span');
            expandIcon.style.cssText = `
                font-size: 12px;
                color: #718096;
            `;
            expandIcon.textContent = data.isExpanded ? '▼' : '▶';
            rightContent.appendChild(expandIcon);
        }
        
        item.appendChild(leftContent);
        item.appendChild(rightContent);
        
        // Metadata section removed to reduce clutter - page title data still available in data attributes
        
        container.appendChild(item);
        console.log(`Added item to container: ${data.name} (level ${level})`);
        
        // Add children if expanded
        if (data.children && data.children.length > 0) {
            console.log(`Adding ${data.children.length} children for ${data.name}`);
            data.children.forEach(child => {
                const childElement = this.buildHierarchicalList(container, child, level + 1);
                // Hide children initially if parent is not expanded
                if (!data.isExpanded && childElement) {
                    childElement.style.display = 'none';
                }
            });
        }
        
        return item;
    }

    getLevelIcon(nodeLevel, hierarchyLevel) {
        // Unique SVG icons for each heading level
        if (hierarchyLevel === 0) {
            return '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path><path d="M8 2v4"></path><path d="M12 2v4"></path><path d="M16 2v4"></path></svg>'; // Root/Page level - Book with pages
        } else if (nodeLevel === 1) {
            return '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="9" y1="9" x2="15" y2="9"></line><line x1="9" y1="13" x2="15" y2="13"></line><line x1="9" y1="17" x2="13" y2="17"></line></svg>'; // H1 - Document with lines
        } else if (nodeLevel === 2) {
            return '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>'; // H2 - File with content
        } else if (nodeLevel === 3) {
            return '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 12l2 2 4-4"></path><path d="M21 12c-1 0-3-1-3-3s2-3 3-3 3 1 3 3-2 3-3 3"></path><path d="M3 12c1 0 3-1 3-3s-2-3-3-3-3 1-3 3 2 3 3 3"></path><path d="M12 3c0 1-1 3-3 3s-3-2-3-3 1-3 3-3 3 2 3 3"></path><path d="M12 21c0-1 1-3 3-3s3 2 3 3-1 3-3 3-3-2-3-3"></path></svg>'; // H3 - Network/Connection nodes
        } else if (nodeLevel === 4) {
            return '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="M12 1v6m0 6v6m11-7h-6m-6 0H1"></path></svg>'; // H4 - Target/Crosshair
        } else if (nodeLevel === 5) {
            return '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"></path></svg>'; // H5 - Star
        } else if (nodeLevel === 6) {
            return '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>'; // H6 - Cube/3D shape
        } else {
            return '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>'; // Default - File with content
        }
    }

    getVisitDate(pageUrl) {
        // Try to get visit date from localStorage or return placeholder
        try {
            const visitData = localStorage.getItem(`visit_${pageUrl}`);
            if (visitData) {
                const data = JSON.parse(visitData);
                return data.date || 'Unknown';
            }
        } catch (error) {
            console.log('Could not get visit date:', error);
        }
        
        // Return a placeholder date for now
        return new Date().toLocaleDateString();
    }

    /**
     * Helper method to get URL data from list items for other components
     * @param {HTMLElement} listItem - The list item element
     * @returns {Object} Object containing URL-related data
     */
    getListItemUrlData(listItem) {
        return {
            pageUrl: listItem.getAttribute('data-page-url'),
            nodeId: listItem.getAttribute('data-node-id'),
            domain: listItem.getAttribute('data-domain'),
            pageTitle: listItem.getAttribute('data-page-title')
        };
    }

    /**
     * Helper method to get all list items with their URL data
     * @returns {Array} Array of objects containing list items and their URL data
     */
    getAllListItemsWithUrlData() {
        const listItems = this.container.querySelectorAll('.list-item');
        return Array.from(listItems).map(item => ({
            element: item,
            urlData: this.getListItemUrlData(item)
        }));
    }

    handleListItemClick(data, element) {
        if (data.isTruncated && data.id === 'truncated-pages') {
            // Special handling for "... X more pages" - show all pages
            console.log('Expanding truncated pages - rebuilding tree with all pages');
            this.buildHierarchicalTreeData(true, true); // Force rebuild with all pages and all levels for list view
            const listContainer = this.container.querySelector('#list-view');
            listContainer.innerHTML = '';
            this.createListView(); // Recreate the entire list view
            return;
        }
        
        if (data.children && data.children.length > 0) {
            // Toggle expansion
            data.isExpanded = !data.isExpanded;
            
            // Update icon
            const icon = element.querySelector('span:last-child');
            icon.textContent = data.isExpanded ? '▼' : '▶';
            
            // Instead of recreating the entire list, just toggle the children visibility
            this.toggleChildrenVisibility(element, data);
        } else {
            // For leaf nodes, add to summative selection
            this.toggleNodeSelection(data);
        }
    }

    toggleChildrenVisibility(element, data) {
        // Find the next sibling elements that are children of this item
        let nextElement = element.nextElementSibling;
        const childrenElements = [];
        
        // Collect all child elements (they have higher margin-left)
        const currentMarginLeft = parseInt(element.style.marginLeft) || 0;
        
        while (nextElement && nextElement.classList && nextElement.classList.contains('list-item')) {
            const nextMarginLeft = parseInt(nextElement.style.marginLeft) || 0;
            
            // If the next element has a higher margin-left, it's a child
            if (nextMarginLeft > currentMarginLeft) {
                childrenElements.push(nextElement);
                nextElement = nextElement.nextElementSibling;
            } else {
                // We've reached a sibling or parent level, stop
                break;
            }
        }
        
        // Toggle visibility of children
        childrenElements.forEach(childElement => {
            if (data.isExpanded) {
                childElement.style.display = 'flex';
            } else {
                childElement.style.display = 'none';
            }
        });
    }

    createRadialTree(container) {
        // Specify the chart's dimensions
        const width = Math.min(this.width, this.height);
        const height = width;
        const cx = width * 0.5;
        const cy = height * 0.5;
        const radius = Math.min(width, height) / 2 - 40;

        // Create a radial tree layout
        const tree = d3.tree()
            .size([2 * Math.PI, radius])
            .separation((a, b) => (a.parent == b.parent ? 1 : 2) / a.depth);

        // Create the SVG container
        const svg = d3.select(container)
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .attr('viewBox', [-cx, -cy, width, height])
            .attr('style', 'width: 100%; height: auto; font: 12px sans-serif;');

        // Add zoom behavior
        const g = svg.append('g');
        
        const zoom = d3.zoom()
            .scaleExtent([0.5, 3]) // Prevent zooming out too much
            .on('zoom', (event) => {
                g.attr('transform', event.transform);
            });

        svg.call(zoom);

        // Store references
        this.svg = svg;
        this.container = container;
        this.g = g;

        // Create the initial tree visualization
        this.updateRadialTree();
    }

    updateRadialTree() {
        if (!this.treeData) return;

        // Sort the tree and apply the layout with optimized separation
        const root = d3.tree()
            .size([2 * Math.PI, Math.min(this.width, this.height) / 2 - 40])
            .separation((a, b) => {
                // Reduce separation for better performance
                if (a.parent == b.parent) {
                    return 0.5 / Math.max(a.depth, 1);
                }
                return 1 / Math.max(a.depth, 1);
            })
            (d3.hierarchy(this.treeData)
                .sort((a, b) => d3.ascending(a.data.name, b.data.name)));

        // Clear existing content
        this.g.selectAll('*').remove();

        // Append links
        this.g.append('g')
            .attr('fill', 'none')
            .attr('stroke', '#555')
            .attr('stroke-opacity', 0.4)
            .attr('stroke-width', 1.5)
            .selectAll('path')
            .data(root.links())
            .join('path')
            .attr('d', d3.linkRadial()
                .angle(d => d.x)
                .radius(d => d.y));

        // Append nodes
        const nodes = this.g.append('g')
            .selectAll('g')
            .data(root.descendants())
            .join('g')
            .attr('transform', d => `rotate(${d.x * 180 / Math.PI - 90}) translate(${d.y},0)`);

        // Add circles for nodes
        nodes.append('circle')
            .attr('fill', d => {
                if (this.selectedNodes.has(d.data.id)) {
                    return '#ffd700';
                } else if (d.data.isTruncated) {
                    return '#ff6b6b'; // Truncated nodes
                } else if (d.children) {
                    return '#4299e1'; // Expandable nodes
                } else {
                    return '#999'; // Leaf nodes
                }
            })
            .attr('stroke', d => {
                if (this.selectedNodes.has(d.data.id)) {
                    return '#ff6b6b';
                } else if (d.data.isTruncated) {
                    return '#e53e3e';
                } else if (d.children) {
                    return '#2c5282';
                }
                return 'none';
            })
            .attr('stroke-width', d => {
                if (this.selectedNodes.has(d.data.id)) {
                    return 3;
                } else if (d.data.isTruncated) {
                    return 2;
                } else if (d.children) {
                    return 2;
                }
                return 0;
            })
            .attr('r', d => {
                if (d.depth === 0) return 8; // Root
                if (d.depth === 1) return 6; // Pages
                if (d.depth === 2) return 4; // H1s
                return 3; // H2s and below
            })
            .style('cursor', d => d.data.isTruncated ? 'not-allowed' : 'pointer')
            .on('mouseover', (event, d) => {
                this.showNodeTooltip(event, d);
            })
            .on('mouseout', (event, d) => {
                this.hideNodeTooltip();
            })
            .on('click', (event, d) => {
                this.handleRadialNodeClick(event, d);
            });

        // Add labels
        nodes.append('text')
            .attr('dy', '0.31em')
            .attr('x', d => d.x < Math.PI === !d.children ? 6 : -6)
            .attr('text-anchor', d => d.x < Math.PI === !d.children ? 'start' : 'end')
            .attr('transform', d => d.x >= Math.PI ? 'rotate(180)' : null)
            .attr('paint-order', 'stroke')
            .attr('stroke', 'white')
            .attr('stroke-width', 3)
            .attr('fill', 'currentColor')
            .style('font-size', d => {
                if (d.depth === 0) return '14px';
                if (d.depth === 1) return '12px';
                return '10px';
            })
            .style('font-weight', d => d.depth <= 2 ? 'bold' : 'normal')
            .text(d => {
                // Truncate long labels more aggressively
                const maxLength = d.depth <= 1 ? 12 : 8; // Much shorter
                return d.data.name.length > maxLength ? 
                    d.data.name.substring(0, maxLength) + '...' : 
                    d.data.name;
            });
    }

    showNodeTooltip(event, d) {
        // Hide any existing tooltip
        this.hideNodeTooltip();
        
        // Create tooltip
        const tooltip = document.createElement('div');
        tooltip.id = 'node-tooltip';
        tooltip.style.cssText = `
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 12px;
            border-radius: 8px;
            font-size: 12px;
            max-width: 300px;
            z-index: 1000;
            pointer-events: none;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        `;
        
        // Build tooltip content
        let content = `<div style="font-weight: bold; margin-bottom: 8px; color: #4299e1;">${d.data.name}</div>`;
        
        if (d.data.isTruncated) {
            content += `<div style="color: #ff6b6b; font-style: italic;">Truncated content</div>`;
        } else if (d.children && d.children.length > 0) {
            content += `<div style="margin-bottom: 6px; color: #cbd5e0;">Children (${d.children.length}):</div>`;
            d.children.forEach(child => {
                const childName = child.data.name.length > 25 ? 
                    child.data.name.substring(0, 25) + '...' : 
                    child.data.name;
                content += `<div style="margin-left: 12px; margin-bottom: 2px;">• ${childName}</div>`;
            });
        } else if (d.data.pageTitle) {
            content += `<div style="color: #cbd5e0;">Page: ${d.data.pageTitle}</div>`;
            if (d.data.domain) {
                content += `<div style="color: #cbd5e0;">Domain: ${d.data.domain}</div>`;
            }
        }
        
        tooltip.innerHTML = content;
        
        // Position tooltip
        const rect = this.container.getBoundingClientRect();
        tooltip.style.left = (event.pageX - rect.left + 10) + 'px';
        tooltip.style.top = (event.pageY - rect.top - 10) + 'px';
        
        // Add to container
        this.container.appendChild(tooltip);
    }

    hideNodeTooltip() {
        const tooltip = this.container.querySelector('#node-tooltip');
        if (tooltip) {
            tooltip.remove();
        }
    }

    handleRadialNodeClick(event, d) {
        event.stopPropagation();
        
        console.log('Radial node clicked:', d.data);

        // Don't allow interaction with truncated nodes
        if (d.data.isTruncated) {
            console.log('Cannot interact with truncated node');
            return;
        }
        
        console.log('Node is not truncated, proceeding with interaction...');

        if (d.children) {
            // Toggle expansion
            if (d.data.isExpanded) {
                // Collapse
                d.data.isExpanded = false;
                d.children = null;
            } else {
                // Expand - load more children if needed
                this.expandNodeWithMoreChildren(d);
            }
            this.updateRadialTree();
        } else {
            // Toggle selection for leaf nodes
            this.toggleNodeSelection(d.data);
        }
    }

    async expandNodeWithMoreChildren(node) {
        // If this is a page node, load more H1 topics
        if (node.data.level === 0 && node.data.pageUrl) {
            await this.loadMoreChildrenForPage(node);
        }
        // If this is an H1 node, load more H2 sections
        else if (node.data.level === 1) {
            await this.loadMoreChildrenForTopic(node);
        }
        
        node.data.isExpanded = true;
    }

    async loadMoreChildrenForPage(pageNode) {
        // Get the original TOC data for this page
        const allTOCEntries = await getAllChapterMapEntries();
        const pageEntry = allTOCEntries.find(entry => entry.url === pageNode.data.pageUrl);
        
        if (pageEntry) {
            // Load more H1 topics (beyond the initial 5)
            const h1Topics = pageEntry.tocData.filter(heading => heading.level === 1);
            const additionalTopics = h1Topics.slice(5); // Get topics beyond the first 5
            
            additionalTopics.forEach(heading => {
                const topicNode = {
                    name: heading.text,
                    children: [],
                    level: heading.level,
                    id: `${pageNode.data.pageUrl}#${heading.id}`,
                    pageUrl: pageNode.data.pageUrl,
                    pageTitle: pageNode.data.pageTitle,
                    domain: pageNode.data.domain,
                    position: heading.position,
                    isExpanded: false,
                    isTruncated: false,
                    content: heading.content || '' // Include content from tocData
                };
                
                // Load H2 sections for this topic
                const h2Sections = pageEntry.tocData.filter(h => 
                    h.level === 2 && h.position > heading.position && 
                    h.position < this.findNextH1Position(pageEntry.tocData, heading.position)
                );
                
                h2Sections.slice(0, 5).forEach(section => {
                    topicNode.children.push({
                        name: section.text,
                        children: [],
                        level: section.level,
                        id: `${pageNode.data.pageUrl}#${section.id}`,
                        pageUrl: pageNode.data.pageUrl,
                        pageTitle: pageNode.data.pageTitle,
                        domain: pageNode.data.domain,
                        position: section.position,
                        isExpanded: false,
                        isTruncated: false,
                        content: section.content || '' // Include content from tocData
                    });
                });
                
                pageNode.children.push(topicNode);
            });
        }
    }

    async loadMoreChildrenForTopic(topicNode) {
        // Get the original TOC data for this page
        const allTOCEntries = await getAllChapterMapEntries();
        const pageEntry = allTOCEntries.find(entry => entry.url === topicNode.data.pageUrl);
        
        if (pageEntry) {
            // Load more H2 sections for this topic
            const h2Sections = pageEntry.tocData.filter(h => 
                h.level === 2 && h.position > topicNode.data.position && 
                h.position < this.findNextH1Position(pageEntry.tocData, topicNode.data.position)
            );
            
            const additionalSections = h2Sections.slice(5); // Get sections beyond the first 5
            
            additionalSections.forEach(section => {
                topicNode.children.push({
                    name: section.text,
                    children: [],
                    level: section.level,
                    id: `${topicNode.data.pageUrl}#${section.id}`,
                    pageUrl: topicNode.data.pageUrl,
                    pageTitle: topicNode.data.pageTitle,
                    domain: topicNode.data.domain,
                    position: section.position,
                    isExpanded: false,
                    isTruncated: false,
                    content: section.content || '' // Include content from tocData
                });
            });
        }
    }

    findNextH1Position(tocData, currentPosition) {
        const nextH1 = tocData.find(h => h.level === 1 && h.position > currentPosition);
        return nextH1 ? nextH1.position : Infinity;
    }

    handleNodeClick(event, d) {
        console.log('Node clicked:', d);

        // Check if node has children and is expandable
        if (d.hasChildren && !d.isExpanded) {
            this.expandNode(d);
        } else if (d.isExpanded) {
            this.collapseNode(d);
        } else {
            // For nodes without children, just select them
            this.toggleNodeSelection(d);
        }
    }

    expandNode(node) {
        console.log(`Expanding node: ${node.label}`);
        
        // Show loader
        this.showLoader();
        
        // Find child nodes from the same page
        const childNodes = this.allTOCData.filter(child => 
            child.pageUrl === node.pageUrl && 
            child.level === node.level + 1 &&
            child.position > node.position &&
            child.position < this.findNextSiblingPosition(node)
        );
        
        if (childNodes.length > 0) {
            // Position child nodes around the parent
            this.positionChildNodesAroundParent(node, childNodes);
            
            // Add child nodes to the graph
            this.nodes.push(...childNodes);
            
            // Create links from parent to children
            const newLinks = childNodes.map(child => ({
                source: node.id,
                target: child.id,
                type: 'hierarchical',
                level: child.level,
                weight: 2,
                color: '#4299e1'
            }));
            
            this.links.push(...newLinks);
            
            // Mark node as expanded
            node.isExpanded = true;
            
            // Update the visualization with better spacing
            this.updateGraphVisualization();
            
            console.log(`Added ${childNodes.length} child nodes`);
        }
        
        // Hide loader
        setTimeout(() => this.hideLoader(), 300);
    }

    positionChildNodesAroundParent(parentNode, childNodes) {
        const parentX = parentNode.x;
        const parentY = parentNode.y;
        const childCount = childNodes.length;
        
        if (childCount === 0) return;
        
        // Calculate positions in a circle around the parent
        const radius = Math.max(150, childCount * 30); // Minimum radius, scales with child count
        
        childNodes.forEach((child, index) => {
            // Distribute children in a circle around the parent
            const angle = (2 * Math.PI * index) / childCount;
            const childX = parentX + radius * Math.cos(angle);
            const childY = parentY + radius * Math.sin(angle);
            
            // Add some randomness to prevent perfect circles
            const randomOffset = 20;
            child.x = childX + (Math.random() - 0.5) * randomOffset;
            child.y = childY + (Math.random() - 0.5) * randomOffset;
            
            console.log(`Positioned child ${child.label} at (${child.x}, ${child.y})`);
        });
    }

    collapseNode(node) {
        console.log(`Collapsing node: ${node.label}`);
        
        // Find all child nodes to remove
        const childNodes = this.nodes.filter(n => 
            n.pageUrl === node.pageUrl && 
            n.level > node.level &&
            n.position > node.position &&
            n.position < this.findNextSiblingPosition(node)
        );
        
        if (childNodes.length > 0) {
            // Remove child nodes
            this.nodes = this.nodes.filter(n => !childNodes.includes(n));
            
            // Remove links involving child nodes
            this.links = this.links.filter(link => {
                const sourceId = link.source.id || link.source;
                const targetId = link.target.id || link.target;
                return !childNodes.some(child => child.id === sourceId || child.id === targetId);
            });
            
            // Mark node as collapsed
            node.isExpanded = false;
            
            // Update the visualization
            this.updateGraphVisualization();
            
            console.log(`Removed ${childNodes.length} child nodes`);
        }
    }

    findNextSiblingPosition(node) {
        // Find the position of the next sibling at the same level
        const siblings = this.allTOCData.filter(n => 
            n.pageUrl === node.pageUrl && 
            n.level === node.level &&
            n.position > node.position
        );
        
        if (siblings.length > 0) {
            return Math.min(...siblings.map(s => s.position));
        }
        
        return Infinity;
    }

    toggleNodeSelection(node) {
        const sidebar = this.container.querySelector('.sidebar') || this.container;
        console.log("Sidebar found:", sidebar)
        if (this.selectedNodes.has(node.id)) {
            // Deselect node
            this.selectedNodes.delete(node.id);
            
            const listItem = sidebar.querySelector(`[data-id="${node.id}"]`);
            if (listItem) {
                listItem.style.opacity = '0';
                setTimeout(() => listItem.remove(), 300);
            }
            
            // Hide right panel if no selections
            if (this.selectedNodes.size === 0) {
                if (rightPanel) {
                    rightPanel.style.display = 'none';
                }
            }
        } else {
            // Select node
            this.selectedNodes.add(node.id);
            
            // Add to sidebar
            const listItem = document.createElement('div');
            listItem.dataset.id = node.id;
            listItem.textContent = node.name || node.label;
            listItem.style.cssText = `
                padding: 8px;
                margin: 5px 0;
                background: #edf2f7;
                border-radius: 4px;
                opacity: 0;
                transition: opacity 0.3s ease;
            `;
            sidebar.appendChild(listItem);
            setTimeout(() => listItem.style.opacity = '1', 10);
            
            // Show right panel with node details
            this.showNodeDetails(node);
        }
        
        // Update button state
        const generateButton = this.wholeContainer.querySelector('#generate-summative-btn');
        if (generateButton) {
            const hasSelectedNodes = this.selectedNodes.size > 0;
            generateButton.disabled = !hasSelectedNodes;
            generateButton.style.backgroundColor = hasSelectedNodes ? '#4299e1' : '#CBD5E0';
            generateButton.style.cursor = hasSelectedNodes ? 'pointer' : 'not-allowed';
        }
        
        // Save selected nodes to localStorage
        localStorage.setItem('knowledgeGraphSelectedNodes', 
            JSON.stringify(Array.from(this.selectedNodes)));
    }

    showNodeDetails(node) {
        // Just call selectNodeForQuiz directly since we're using sidebar only
        this.selectNodeForQuiz(node);
    }

    selectNodeForQuiz(node) {
        console.log('selectNodeForQuiz called with node:', node);
        console.log('Current sidebar dictionary size:', this.sidebarNodes.size);
        
        // Add the node to quiz selection if not already selected
        if (!this.selectedNodes.has(node.id)) {
            this.selectedNodes.add(node.id);
        } else {
            console.log('Node already in selected nodes:', node.id);
        }
        
        // Just add the selected item to the sidebar list with delete option
        const sidebar = this.wholeContainer.querySelector('.sidebar');
        console.log('Adding node to sidebar:', node.id);
        
        // Check if this item is already in the sidebar
        const existingItem = sidebar.querySelector(`[data-quiz-id="${node.id}"]`);
        console.log('Existing item:', existingItem);
        if (existingItem) {
            console.log('Item already exists, returning');
            return; // Already added
        }
        
        // Log node data being added to sidebar
        console.log('Adding node to sidebar:', {
            id: node.id,
            name: node.name || node.label,
            content: node.content ? node.content.substring(0, 100) + '...' : 'No content',
            pageUrl: node.pageUrl,
            level: node.level
        });
        
        // Create a new item for the quiz selection list
        const quizItem = document.createElement('div');
        // PREVIOUS UNDERSTANDING: Content was stored in dataset attributes (lines 2386-2390)
        // These dataset attributes are used for data extraction when generating quiz
        quizItem.dataset.quizId = node.id;
        quizItem.dataset.content = node.content || ''; // Store content as data attribute
        quizItem.dataset.pageUrl = node.pageUrl || '';
        quizItem.dataset.domain = node.domain || '';
        quizItem.dataset.level = node.level || 0;
        quizItem.dataset.position = node.position || 0; // Store position for scrolling
        quizItem.style.cssText = `
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 12px;
            margin: 5px 0;
            background: #e6fffa;
            border: 1px solid #38b2ac;
            border-radius: 6px;
            font-size: 12px;
            color: #2d3748;
        `;
        
        // Get additional info for the component
        const pageUrl = node.pageUrl || '';
        const domain = node.domain || '';
        const visitDate = this.getVisitDate(pageUrl);
        
        // LOG: Verify content is being added to data attributes
        console.log('🔍 ADDING CONTENT TO SIDEBAR ITEM:');
        console.log('  - Node ID:', node.id);
        console.log('  - Node Name:', node.name || node.label);
        console.log('  - Content Length:', node.content ? node.content.length : 0);
        console.log('  - Content Preview:', node.content ? node.content.substring(0, 50) + '...' : 'NO CONTENT');
        console.log('  - Page URL:', pageUrl);
        console.log('  - Domain:', domain);
        console.log('  - Level:', node.level);
        
        quizItem.innerHTML = `
            <div style="flex: 1; margin-right: 8px;">
                <div style="font-weight: 500; margin-bottom: 2px;">${node.name || node.label}</div>
                <div style="font-size: 10px; color: #718096; margin-bottom: 1px;">${domain}</div>
                <div style="font-size: 9px; color: #a0aec0;">${visitDate}</div>
            </div>
            <button class="remove-quiz-item" style="
                background: #e53e3e;
                color: white;
                border: none;
                border-radius: 3px;
                width: 20px;
                height: 20px;
                cursor: pointer;
                font-size: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
            ">×</button>
        `;
        
        // LOG: Verify data attributes were set correctly
        console.log('✅ DATA ATTRIBUTES SET:');
        console.log('  - data-quiz-id:', quizItem.dataset.quizId);
        console.log('  - data-content length:', quizItem.dataset.content ? quizItem.dataset.content.length : 0);
        console.log('  - data-page-url:', quizItem.dataset.pageUrl);
        console.log('  - data-domain:', quizItem.dataset.domain);
        console.log('  - data-level:', quizItem.dataset.level);
        console.log('  - data-position:', quizItem.dataset.position);
        
        // Add click handler for remove button
        const removeBtn = quizItem.querySelector('.remove-quiz-item');
        removeBtn.addEventListener('click', () => {
            this.selectedNodes.delete(node.id);
            quizItem.remove();
            
            console.log('Removed node from sidebar:', node.id);
            
            // Update button state
            const generateButton = this.wholeContainer.querySelector('#generate-summative-btn');
            if (generateButton) {
                const hasSelectedNodes = this.selectedNodes.size > 0;
                generateButton.disabled = !hasSelectedNodes;
                generateButton.style.backgroundColor = hasSelectedNodes ? '#4299e1' : '#CBD5E0';
                generateButton.style.cursor = hasSelectedNodes ? 'pointer' : 'not-allowed';
            }
        });
        
        // Insert into the scrollable quiz items container
        const quizItemsContainer = sidebar.querySelector('.quiz-items-container');
        console.log('Quiz items container found:', quizItemsContainer);
        
        if (quizItemsContainer) {
            quizItemsContainer.appendChild(quizItem);
            console.log('Quiz item appended to quiz items container');
        } else {
            // Fallback: insert before button container
            const buttonContainer = sidebar.querySelector('div[style*="position: absolute; bottom"]');
            if (buttonContainer) {
                sidebar.insertBefore(quizItem, buttonContainer);
                console.log('Quiz item inserted before button container (fallback)');
            } else {
                sidebar.appendChild(quizItem);
                console.log('Quiz item appended to sidebar (fallback)');
            }
        }
        
        // Update button state
        const generateButton = this.wholeContainer.querySelector('#generate-summative-btn');
        if (generateButton) {
            const hasSelectedNodes = this.selectedNodes.size > 0;
            generateButton.disabled = !hasSelectedNodes;
            generateButton.style.backgroundColor = hasSelectedNodes ? '#4299e1' : '#CBD5E0';
            generateButton.style.cursor = hasSelectedNodes ? 'pointer' : 'not-allowed';
        }
        
        // Save selected nodes to localStorage
        localStorage.setItem('knowledgeGraphSelectedNodes', 
            JSON.stringify(Array.from(this.selectedNodes)));
    }

    resetToH1Level() {
        console.log('Resetting to H1 level');
        
        // Show loader
        this.showLoader();
        
        // Reset to only H1 nodes
        this.nodes = this.allTOCData.filter(node => node.level === 1);
        this.links = this.buildInitialLinks(this.nodes);
        
        // Reset all expansion states
        this.allTOCData.forEach(node => {
            node.isExpanded = false;
        });
        
        // Clear selected nodes
        this.selectedNodes.clear();
        
        // Update the visualization
        this.updateGraphVisualization();
        
        // Hide loader
        setTimeout(() => this.hideLoader(), 300);
    }

    updateGraphVisualization() {
        // Update simulation with new data
        this.simulation.nodes(this.nodes);
        this.simulation.force('link').links(this.links);
        
        // Update existing nodes and links
        this.updateNodesAndLinks();
        
        // Adjust simulation forces based on number of nodes
        const nodeCount = this.nodes.length;
        if (nodeCount > 20) {
            // For many nodes, reduce forces to prevent chaos
            this.simulation.force('charge').strength(-1500);
            this.simulation.force('collision').radius(d => d.size + 60);
        } else {
            // For fewer nodes, use stronger forces for better spacing
            this.simulation.force('charge').strength(-3000);
            this.simulation.force('collision').radius(d => d.size + 80);
        }
        
        // Restart simulation with higher energy for better spacing
        this.simulation.alpha(0.5).restart();
    }

    updateNodesAndLinks() {
        // Update links
        this.linkGroup.selectAll('line')
            .data(this.links, d => `${d.source.id || d.source}-${d.target.id || d.target}`)
            .join(
                enter => enter.append('line')
                    .attr('class', 'link')
                    .attr('stroke-width', d => Math.max(2, d.weight * 2))
                    .attr('stroke', d => d.color || '#2d3748')
                    .attr('stroke-opacity', 0.8),
                update => update,
                exit => exit.remove()
            );
        
        // Update nodes
        const nodeUpdate = this.nodeGroup.selectAll('g')
            .data(this.nodes, d => d.id);
        
        const nodeEnter = nodeUpdate.enter()
            .append('g')
            .each(function(d, i, nodes) {
                // Only set position for new nodes that don't have one
                if (d.x === null || d.y === null) {
                    const width = this.parentNode.parentNode.clientWidth || 1200;
                    const height = this.parentNode.parentNode.clientHeight || 800;
                    
                    // Use a more spread out grid layout
                    const totalNodes = nodes.length;
                    const cols = Math.ceil(Math.sqrt(totalNodes));
                    const rows = Math.ceil(totalNodes / cols);
                    const colWidth = width / (cols + 1);
                    const rowHeight = height / (rows + 1);
                    
                    const col = i % cols;
                    const row = Math.floor(i / cols);
                    
                    // Add more spacing between nodes
                    d.x = (col + 1) * colWidth + (Math.random() - 0.5) * colWidth * 0.2;
                    d.y = (row + 1) * rowHeight + (Math.random() - 0.5) * rowHeight * 0.2;
                }
            });
        
        // Add circles to new nodes
        nodeEnter.append('circle')
            .attr('r', d => d.size)
            .attr('fill', d => {
                if (this.selectedNodes.has(d.id)) {
                    return this.colors.selected;
                }
                return d.color;
            })
            .attr('stroke', d => {
                if (this.selectedNodes.has(d.id)) {
                    return '#ffd700';
                } else if (d.hasChildren && !d.isExpanded) {
                    return '#4299e1'; // Blue border for expandable nodes
                } else if (d.isExpanded) {
                    return '#2c5282'; // Darker border for expanded nodes
                }
                return 'none';
            })
            .attr('stroke-width', d => {
                if (this.selectedNodes.has(d.id)) {
                    return 3;
                } else if (d.hasChildren) {
                    return 2; // Thicker border for expandable nodes
                }
                return 0;
            })
            .style('opacity', d => d.opacity)
            .on('click', (event, d) => {
                this.handleNodeClick(event, d);
            });
        
        // Add labels to new nodes
        nodeEnter.append('text')
            .text(d => d.label)
            .attr('x', 15)
            .attr('y', 5)
            .style('font-size', '12px')
            .style('fill', '#2d3748');
        
        // Remove old nodes
        nodeUpdate.exit().remove();
        
        // Update node labels
        this.nodeLabelGroup.selectAll('text')
            .data(this.nodes, d => d.id)
            .join(
                enter => enter.append('text')
                    .attr('class', 'node-label')
                    .text(d => d.label)
                    .attr('font-size', d => Math.max(10, d.size * 0.4))
                    .attr('font-weight', d => d.level === 1 ? 'bold' : 'normal')
                    .attr('text-anchor', 'middle')
                    .attr('dy', d => d.size + 15)
                    .style('pointer-events', 'none')
                    .style('fill', '#2d3748'),
                update => update,
                exit => exit.remove()
            );
    }

    fitToView() {
        if (!this.svg || !this.nodes || !this.nodes.length) return;
        
        // Wait for nodes to have positions
        const nodesWithPositions = this.nodes.filter(node => 
            node.x !== null && node.y !== null && 
            !isNaN(node.x) && !isNaN(node.y)
        );
        
        if (nodesWithPositions.length === 0) {
            // If no nodes have positions yet, try again later
            setTimeout(() => this.fitToView(), 500);
            return;
        }
        
        // Calculate bounds of positioned nodes
        const bounds = nodesWithPositions.reduce((acc, node) => {
            acc.minX = Math.min(acc.minX, node.x);
            acc.maxX = Math.max(acc.maxX, node.x);
            acc.minY = Math.min(acc.minY, node.y);
            acc.maxY = Math.max(acc.maxY, node.y);
            return acc;
        }, { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity });
        
        if (bounds.minX === Infinity) return;
        
        // Add padding
        const padding = 100;
        const width = bounds.maxX - bounds.minX + padding * 2;
        const height = bounds.maxY - bounds.minY + padding * 2;
        
        // Calculate scale to fit
        const scaleX = this.width / width;
        const scaleY = this.height / height;
        const scale = Math.min(scaleX, scaleY, 1); // Don't zoom in beyond 1:1
        
        // Calculate center
        const centerX = (bounds.minX + bounds.maxX) / 2;
        const centerY = (bounds.minY + bounds.maxY) / 2;
        
        // Calculate translate to center the graph
        const translateX = this.width / 2 - centerX * scale;
        const translateY = this.height / 2 - centerY * scale;
        
        // Apply transform
        const transform = d3.zoomIdentity
            .translate(translateX, translateY)
            .scale(scale);
        
        this.svg.transition()
            .duration(1000)
            .call(d3.zoom().transform, transform);
    }

    // Updated drag handlers with simulation reference
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

    async getSelectedNodesSummaries() {
        const summaries = [];
        
        try {
            const dbManager = await getDBInstance();
            if (!dbManager || !dbManager.db) {
                throw new Error('Database not properly initialized');
            }

            for (const nodeId of this.selectedNodes) {
                const node = this.nodes.find(n => n.id === nodeId);
                if (!node) continue;

                console.log('Processing node:', node);
                console.log('Node chapter:', node.chapter);
                console.log('Node pageUrl:', node.pageUrl);

                try {
                    // Extract the normalized URL from the node ID for chapterMap lookup
                    let chapterKey = node.chapter;
                    
                    // If node.chapter is not available, try to extract from node.id or node.pageUrl
                    if (!chapterKey) {
                        if (node.id && node.id.startsWith('http')) {
                            // Extract the base URL from the node ID (remove hash fragment)
                            const url = new URL(node.id);
                            chapterKey = `${url.protocol}//${url.host}${url.pathname}`;
                            console.log('Extracted chapterKey from node.id:', chapterKey);
                        } else if (node.pageUrl) {
                            chapterKey = node.pageUrl;
                            console.log('Using node.pageUrl as chapterKey:', chapterKey);
                        }
                    } else {
                        console.log('Using node.chapter as chapterKey:', chapterKey);
                    }
                    
                    console.log(`Looking for chapterMap data with key: ${chapterKey}`);
                    const chapterMapData = await dbManager.getByKey('chapterMap', chapterKey);

                    if (!chapterMapData) {
                        console.log(`No chapterMap data found for chapter ${chapterKey} - you may need to visit this chapter first`);
                        
                        // Try to find chapterMap entries that might match
                        const allChapterMapEntries = await dbManager.getAll('chapterMap');
                        console.log('Available chapterMap entries:', allChapterMapEntries.map(entry => ({
                            url: entry.url,
                            title: entry.title,
                            domain: entry.domain
                        })));
                        continue;
                    }

                    console.log('Retrieved chapterMap data:', chapterMapData);
                    
                    // 🔍 DETAILED LOGGING: Log the TOC data structure and content
                    console.log(`📚 Processing TOC for "${chapterMapData.title}" (${chapterMapData.tocData ? chapterMapData.tocData.length : 0} entries)`);
                    // Uncomment below for detailed TOC analysis
                    /*
                    console.group(`📚 TOC DATA ANALYSIS for "${chapterMapData.title}"`);
                    console.log('📊 ChapterMap Entry Details:', {
                        url: chapterMapData.url,
                        originalUrl: chapterMapData.originalUrl,
                        title: chapterMapData.title,
                        tocDataCount: chapterMapData.tocData ? chapterMapData.tocData.length : 0,
                        lastUpdated: chapterMapData.lastUpdated
                    });
                    
                    if (chapterMapData.tocData && chapterMapData.tocData.length > 0) {
                        console.log('📋 Full TOC Data Structure:');
                        chapterMapData.tocData.forEach((tocItem, index) => {
                            console.log(`  ${index + 1}. [Level ${tocItem.level}] "${tocItem.text}"`);
                            console.log(`     ID: ${tocItem.id}`);
                            console.log(`     Position: ${tocItem.position}`);
                            console.log(`     Index: ${tocItem.index}`);
                            console.log(`     Content Length: ${tocItem.content ? tocItem.content.length : 0} chars`);
                            console.log(`     Content Preview: "${tocItem.content ? tocItem.content.substring(0, 150) + '...' : 'NO CONTENT'}"`);
                            console.log(`     Content Full: "${tocItem.content || 'NO CONTENT'}"`);
                            console.log('     ---');
                        });
                    } else {
                        console.log('❌ No TOC data found in chapterMap entry');
                    }
                    console.groupEnd();
                    */

                    if (node.type === 'chapter') {
                        // For chapter nodes, use the page title and URL
                        const content = `Chapter: ${chapterMapData.title}\nURL: ${chapterMapData.originalUrl || chapterMapData.url}`;
                        summaries.push({
                            title: node.label,
                            content: content
                        });
                        console.log('Added chapter content:', content);
                    } else if (node.type === 'section') {
                        // 🔍 DETAILED LOGGING: Section content matching process
                        console.log(`🔍 Matching section "${node.label}"`);
                        // Uncomment below for detailed matching analysis
                        /*
                        console.group(`🔍 SECTION CONTENT MATCHING for "${node.label}"`);
                        console.log('🎯 Looking for TOC entry matching node label:', node.label);
                        console.log('📋 Available TOC entries to match against:');
                        chapterMapData.tocData.forEach((entry, index) => {
                            console.log(`  ${index + 1}. "${entry.text}" (Level ${entry.level})`);
                        });
                        */
                        
                        // Find matching TOC entry
                        const tocEntry = chapterMapData.tocData.find(entry => 
                            entry.text === node.label || 
                            entry.text.includes(node.label) ||
                            node.label.includes(entry.text)
                        );

                        if (tocEntry) {
                            console.log(`✅ Found TOC match: "${tocEntry.text}" (${tocEntry.content ? tocEntry.content.length : 0} chars)`);
                            /*
                            console.log('✅ Found matching TOC entry:', {
                                text: tocEntry.text,
                                level: tocEntry.level,
                                id: tocEntry.id,
                                contentLength: tocEntry.content ? tocEntry.content.length : 0
                            });
                            console.log('📄 TOC Entry Content:', tocEntry.content);
                            */
                            
                            let content = tocEntry.content;
                            
                            // If no content, use URL fallback
                            if (!content || content.trim() === '') {
                                content = `Section: ${tocEntry.text}\nURL: ${chapterMapData.originalUrl || chapterMapData.url}`;
                                console.log('⚠️ Using URL fallback (no content):', tocEntry.text);
                            } else {
                                console.log('✅ Using content snippet:', tocEntry.text);
                            }
                            
                            summaries.push({
                                title: node.label,
                                content: content
                            });
                        } else {
                            // No matching TOC entry found, use URL fallback
                            console.log('❌ No TOC match found for:', node.label);
                            /*
                            console.log('🔍 Attempted matches:');
                            chapterMapData.tocData.forEach((entry, index) => {
                                const exactMatch = entry.text === node.label;
                                const includesLabel = entry.text.includes(node.label);
                                const labelIncludesEntry = node.label.includes(entry.text);
                                console.log(`  ${index + 1}. "${entry.text}" - exact: ${exactMatch}, includes: ${includesLabel}, reverse: ${labelIncludesEntry}`);
                            });
                            */
                            
                            const content = `Section: ${node.label}\nURL: ${chapterMapData.originalUrl || chapterMapData.url}`;
                            summaries.push({
                                title: node.label,
                                content: content
                            });
                            console.log('⚠️ Using URL fallback for unmatched section:', node.label);
                        }
                        // console.groupEnd();
                    }
                } catch (error) {
                    console.error(`Error processing node ${nodeId}:`, error);
                    // Continue with other nodes instead of breaking the entire process
                }
            }
        } catch (error) {
            console.error('Database error:', error);
            showPopover(document, error.message || 'Failed to load summaries. Please try visiting some chapters first.', 'error');
            return [];
        }
        
        console.log('Final summaries:', summaries);
        return summaries;
    }
}

function createLoadingSpinner() {
    const spinner = document.createElement('div');
    spinner.id = 'knowledge-graph-spinner';
    spinner.style.cssText = `
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 50px;
        height: 50px;
        border: 5px solid #f3f3f3;
        border-top: 5px solid #3498db;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    `;

    // Add keyframes for spinner animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes spin {
            0% { transform: translate(-50%, -50%) rotate(0deg); }
            100% { transform: translate(-50%, -50%) rotate(360deg); }
        }
    `;
    spinner.appendChild(style);
    return spinner;
}
export function showKnowledgeGraph(shadowRoot) {
    if (!modalInstance) {
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        `;

        const content = document.createElement('div');
        content.style.cssText = `
            background: white;
            padding: 20px;
            border-radius: 10px;
            width: 90%;
            height: 90%;
            position: relative;
            display: flex;
            flex-direction: column;
        `;

        // Add click handlers for modal
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.style.display = 'none';
            }
        });

        // Add escape key handler
        const handleEscape = (e) => {
            if (e.key === 'Escape') {
                modal.style.display = 'none';
            }
        };
        document.addEventListener('keydown', handleEscape);

        // Clean up event listener when modal is hidden
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'attributes' && mutation.attributeName === 'style') {
                    if (modal.style.display === 'none') {
                        document.removeEventListener('keydown', handleEscape);
                    }
                }
            });
        });
        observer.observe(modal, { attributes: true });

        const closeButton = document.createElement('button');
        closeButton.textContent = '×';
        closeButton.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            border: none;
            background: none;
            font-size: 24px;
            cursor: pointer;
            z-index: 1;
        `;

        content.appendChild(closeButton);
        modal.appendChild(content);
        shadowRoot.appendChild(modal);

        closeButton.onclick = () => {
            modal.style.display = 'none';
        };

        modalInstance = { modal, content };
    }

    const { modal, content } = modalInstance;
    
    // Clear previous content except close button
    const closeButton = content.firstChild;
    content.innerHTML = '';
    content.appendChild(closeButton);

    // Show modal and add spinner
    modal.style.display = 'flex';
    const spinner = createLoadingSpinner();
    content.appendChild(spinner);

    // Create new graph
    const graph = new KnowledgeGraph();
    
    // Initialize and create visualization
    graph.initData().then(() => {
        if (spinner && spinner.parentNode === content) {
            content.removeChild(spinner);
        }
        graph.createVisualization(content);
    }).catch(error => {
        console.error('Error creating knowledge graph:', error);
        if (spinner && spinner.parentNode === content) {
            content.removeChild(spinner);
        }
        
        const errorMsg = document.createElement('div');
        errorMsg.style.cssText = `
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: red;
            text-align: center;
        `;
        errorMsg.textContent = 'Error loading knowledge graph. Please try again.';
        content.appendChild(errorMsg);
        
        setTimeout(() => {
            if (errorMsg && errorMsg.parentNode === content) {
                content.removeChild(errorMsg);
            }
        }, 3000);
    });
}

export function initKnowledgeGraph(shadowRoot) {
    const knowledgeGraphBtn = shadowRoot.querySelector('#knowledge-graph-btn');
    enableTooltip(knowledgeGraphBtn, "View the knowledge graph", shadowRoot);
    knowledgeGraphBtn.addEventListener('click', () => {
        showKnowledgeGraph(shadowRoot);
    });
}
