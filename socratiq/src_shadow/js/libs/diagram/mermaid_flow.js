// src_shadow/js/components/markdown/mermaid.js

// import { elements } from "chart.js";
import mermaid from "mermaid";

// Initialize mermaid with default config
mermaid.initialize({
  startOnLoad: false,
  theme: "default",
  securityLevel: "loose",
  flowchart: {
    htmlLabels: true,
    curve: "basis",
    width: 320, 
  },
});

// Add common syntax fixes
const commonFixes = {
  classDiagram: {
    patterns: [
      // Fix missing curly braces
      { 
        find: /^(\s*class\s+\w+)(?!\s*{)/gm,
        replace: '$1 {'
      },
      // Fix relationship arrows
      {
        find: /(\w+)\s*--\s*(\w+)\s*:/g,
        replace: '$1 --> $2 :'
      },
      // Fix method syntax (convert +method : return to +method() return)
      {
        find: /([+-])(\w+)\s*:\s*(\w+)/g,
        replace: '$1$2() $3'
      },
      // Fix parameter syntax
      {
        find: /([+-])(\w+)\((\w+)\s*:\s*(\w+)\)\s*:\s*(\w+)/g,
        replace: '$1$2($3 : $4) $5'
      },
      // Add missing closing braces
      {
        find: /(class\s+\w+\s*{[^}]*?)(?=\s*class|\s*$)/g,
        replace: '$1\n}'
      },
      // Fix visibility modifiers
      {
        find: /(\s*)([\+\-])(\w+)/g,
        replace: '$1$2 $3'
      },
      // Fix return type syntax
      {
        find: /\)\s*:(\w+)/g,
        replace: ') $1'
      }
    ]
  },
  flowchart: {
    patterns: [
      // Fix arrow syntax
      {
        find: /(--)(?!>)/g,
        replace: '-->'
      }
    ]
  }
};

// Helper function to extract mermaid code and caption from response
function extractMermaidAndCaption(text) {
  try {
    let content = text;
    // ProviderSingle returns plain message text. Retain compatibility with old
    // response envelopes without requiring a JSON wrapper around fenced code.
    if (typeof text === 'string' && /^[{[]/.test(text.trim())) {
      try { content = JSON.parse(text); } catch { content = text; }
    }
    if (typeof content !== 'string') {
      content = content?.choices?.[0]?.message?.content ||
        content?.[0]?.generated_text || content?.generated_text || content?.text || '';
    }
    const mermaidMatch = content.match(/```(?:mermaid)?\s*([\s\S]*?)```/);
    const captionMatch = content.match(/\$\$([\s\S]*?)\$\$/);
    const code = (mermaidMatch ? mermaidMatch[1] : content).trim();
    if (!/^(graph|sequenceDiagram|classDiagram|stateDiagram|erDiagram|gantt|pie|flowchart|mindmap|timeline|xychart|quadrantChart)\b/.test(code)) {
      throw new Error('No valid Mermaid diagram found in provider response');
    }
    return { code, caption: captionMatch?.[1]?.trim() || '', success: true };
  } catch (error) {
    return { code: '', caption: '', success: false, error: error.message };
  }
}

// Modified renderMermaidDiagram to handle both code and caption
export const renderMermaidFlow = async (text) => {
  try {
    const id = `mermaid-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    // Extract both code and caption
    const { code, caption, success, error } = extractMermaidAndCaption(text);
    let workingCode = code;
    
    if (!success) {
      throw new Error(error || 'Failed to extract valid mermaid code');
    }

    // Validate mermaid syntax
    try {
      await mermaid.parse(workingCode);
    } catch (parseError) {
      console.log("Initial parse failed, attempting fixes...");
      const fixedCode = attemptCodeFix(workingCode);
      if (fixedCode === workingCode) {
        throw parseError;
      }
      await mermaid.parse(fixedCode);
      workingCode = fixedCode;
    }
    
    // Render the diagram
    const { svg } = await mermaid.render(id, workingCode);
    
    // Create container with figure and caption
    const figure = document.createElement("figure");
    figure.className = "mermaid-figure";
    figure.style.cssText = `
      margin: 1em 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      background: #f8f9fa;
      padding: 1em;
      border-radius: 4px;
      width: 320px;
    `;

    const diagramContainer = document.createElement("div");
    diagramContainer.className = "mermaid-diagram";
    diagramContainer.innerHTML = svg;
    figure.appendChild(diagramContainer);

    if (caption) {
      const figcaption = document.createElement("figcaption");
      figcaption.textContent = caption;
      figcaption.style.cssText = `
        margin-top: 0.5em;
        text-align: left;
        color: #666;
      `;
      figure.appendChild(figcaption);
    }

    console.log("I am returning figure", figure);
    console.log("I am returning svg", svg);
    console.log("I am returning code", workingCode);
    console.log("I am returning caption", caption);

    return { 
      success: true, 
      element: figure, 
      svg,
      code: workingCode,
      caption 
    };
  } catch (error) {
    console.error("Error in renderMermaidDiagram:", error);
    return { 
      success: false, 
      error: error.message 
    };
  }
};

export function containsWordReference(text, keywords = KEYWORDS) {
  const lowerCaseText = text.toLowerCase();
  const foundKeywords = [];

  for (const keyword of keywords) {
    // Check for exact word matches first (with word boundaries)
    const exactRegex = new RegExp(`\\b${keyword.toLowerCase()}\\b`, 'g');
    const exactMatches = lowerCaseText.match(exactRegex);
    
    if (exactMatches) {
      foundKeywords.push({
        keyword: keyword,
        count: exactMatches.length,
        matches: exactMatches,
        type: 'exact'
      });
      continue; // Skip to next keyword if we found exact matches
    }

    // If no exact match, check for partial matches (plurals, embedded words, etc)
    const partialRegex = new RegExp(keyword.toLowerCase(), 'g');
    const partialMatches = lowerCaseText.match(partialRegex);
    
    if (partialMatches) {
      foundKeywords.push({
        keyword: keyword,
        count: partialMatches.length,
        matches: partialMatches,
        type: 'partial'
      });
    }
  }

  // Return true if we found any matches (exact or partial)
  return foundKeywords.length > 0;
}

// Define diagram-related keywords with their variations
const KEYWORDS = [
  "diagram",    // Will match: diagrams, diagrammatic, etc.
  "flowchart",  // Will match: flowcharts, flowcharting, etc.
  "sequence",   // Will match: sequences, sequential, etc.
  "class",      // Will match: classes, classification, etc.
  "er",         // Will match: erd, er-diagram, etc.
  "graph",      // Will match: graphs, graphical, etc.
  "mindmap",    // Will match: mindmaps, mindmapping, etc.
  "timeline",   // Will match: timelines, etc.
  "gantt"       // Will match: ganttchart, etc.
];
