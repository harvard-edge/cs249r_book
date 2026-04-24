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
  },
  xyChart: {
    width: 320,
    height: 500
  },
  quadrantChart: {
    width: 320,
    height: 500,
    titlePadding: 10,
    quadrantPadding: 5
  }
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
  },
  xyChart: {
    patterns: [
      {
        find: /xychart(?!-beta)/g,
        replace: 'xychart-beta'
      }
    ]
  },
  quadrantChart: {
    patterns: []
  }
};

// // Helper function to extract mermaid code and caption from response
// function extractMermaidAndCaption(text) {
//   try {
//     // If the text is a JSON string, try to parse it
//     if (typeof text === 'string' && (text.startsWith('{') || text.startsWith('['))) {
//       const parsed = JSON.parse(text);
      
//       const content = parsed.choices?.[0]?.message?.content || 
//                      parsed[0]?.generated_text || 
//                      parsed.generated_text ||
//                      parsed.text ||
//                      text;

//       // Extract mermaid code and caption with improved regex
//       const mermaidMatch = content.match(/```(?:mermaid)?\s*([\s\S]*?)```/);
//       const captionMatch = content.match(/\$\$([\s\S]*?)\$\$/);

//       let mermaidCode = mermaidMatch ? mermaidMatch[1].trim() : '';
//       const caption = captionMatch ? captionMatch[1].trim() : '';

//       // Clean up the mermaid code
//       if (mermaidCode) {
//         // Remove excessive indentation
//         mermaidCode = mermaidCode.split('\n')
//           .map(line => line.trimStart())  // Remove leading whitespace
//           .join('\n')
//           .trim();
//       }

//       // Validate and fix diagram type
//       let fixedCode = mermaidCode;
//       if (mermaidCode && !mermaidCode.match(/^(graph|sequenceDiagram|classDiagram|stateDiagram|erDiagram|gantt|pie|flowchart)/)) {
//         if (mermaidCode.includes('[') && mermaidCode.includes(']') &&
//             (mermaidCode.includes('-->') || mermaidCode.includes('--'))) {
//             fixedCode = 'graph TD\n' + mermaidCode;
//         }
//       }

//       console.log("I am fixed code", fixedCode);

//       return {
//         code: fixedCode,
//         caption: caption,
//         success: !!fixedCode
//       };
//     }

//     return {
//       code: '',
//       caption: '',
//       success: false
//     };
//   } catch (e) {
//     console.warn('Error parsing response:', e);
//     return {
//       code: '',
//       caption: '',
//       success: false,
//       error: e.message
//     };
//   }
// }

// Modified renderMermaidDiagram to handle both code and caption
export const renderMermaidDiagram = async (text) => {
  try {
    const id = `mermaid-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    // Check if the input is already mermaid syntax
    const isMermaidSyntax = text.includes('```mermaid');
    let code, caption;

    

    if (isMermaidSyntax) {
      // Extract code and caption from mermaid block
      const mermaidMatch = text.match(/```mermaid\s*([\s\S]*?)```/);
      const captionMatch = text.match(/\$\$([\s\S]*?)\$\$/);
      
      code = mermaidMatch ? mermaidMatch[1].trim() : '';
      caption = captionMatch ? captionMatch[1].trim() : '';
    } else {
      // Treat the input as direct mermaid code
      code = text.trim();
    }

    if (!code) {
      throw new Error('No valid mermaid code found');
    }

    // Apply chart-specific fixes
    if (code.includes('quadrantChart')) {
      code = applyFixes(code, commonFixes.quadrantChart);
    } else if (code.includes('xychart')) {
      code = applyFixes(code, commonFixes.xyChart);
    }

    console.log("I am fixed code", code);

    // Validate mermaid syntax
    try {
      await mermaid.parse(code);
    } catch (parseError) {
      console.log("Parse error:", parseError);
      throw parseError;
    }
    
    // Render the diagram
    const { svg } = await mermaid.render(id, code);
    
    // Create container with figure and caption
    const figure = document.createElement("figure");
    figure.className = "mermaid-figure";
    // figure.className = "mermaid-figure zoomable-image"; 
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
    diagramContainer.className = "mermaid-diagram zoomable-image";
    diagramContainer.innerHTML = svg;
    figure.appendChild(diagramContainer);

    if (caption) {
      const figcaption = document.createElement("figcaption");
      figcaption.textContent = caption;
      figcaption.style.cssText = `
        margin-top: 0.5em;
        text-align: left;
        color: #666;
        width: 100%;
      `;
      figure.appendChild(figcaption);
    }

    return { 
      success: true, 
      element: figure, 
      svg,
      code,
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

// Helper function to apply fixes
function applyFixes(code, patterns) {
  if (!patterns || !patterns.patterns) return code;
  
  let fixedCode = code;
  patterns.patterns.forEach(pattern => {
    fixedCode = fixedCode.replace(pattern.find, pattern.replace);
  });
  return fixedCode;
}

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

// Specialized render function for progress report charts
// export const renderProgressChart = async (chartType, code) => {
//     try {
//         const id = `mermaid-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        
//         // Clean and validate the code
//         let cleanCode = code.trim();
//         if (!cleanCode.includes('```mermaid')) {
//             cleanCode = '```mermaid\n' + cleanCode + '\n```';
//         }

//         // Extract the actual mermaid code
//         const mermaidCode = cleanCode.replace(/```mermaid\n?/, '').replace(/\n?```$/, '').trim();
        
//         console.log(`Rendering ${chartType} chart with code:`, mermaidCode);

//         // Validate mermaid syntax
//         try {
//             await mermaid.parse(mermaidCode);
//         } catch (parseError) {
//             console.log(`Parse error in ${chartType}:`, parseError);
//             throw parseError;
//         }
        
//         // Render the diagram
//         const { svg } = await mermaid.render(id, mermaidCode);
        
//         // Create container with figure and caption
//         const figure = document.createElement("figure");
//         figure.className = `mermaid-${chartType}-figure`;
//         figure.style.cssText = `
//             margin: 1em 0;
//             display: flex;
//             flex-direction: column;
//             align-items: center;
//             background: #f8f9fa;
//             padding: 1em;
//             border-radius: 4px;
//             width: ${chartType === 'quadrant' ? '600px' : '400px'};
//         `;

//         const diagramContainer = document.createElement("div");
//         diagramContainer.className = `mermaid-${chartType}-diagram`;
//         diagramContainer.innerHTML = svg;
//         figure.appendChild(diagramContainer);

//         // Extract and add caption if present
//         const captionMatch = code.match(/\$\$([\s\S]*?)\$\$/);
//         if (captionMatch) {
//             const figcaption = document.createElement("figcaption");
//             figcaption.textContent = captionMatch[1].trim();
//             figcaption.style.cssText = `
//                 margin-top: 0.5em;
//                 text-align: left;
//                 color: #666;
//                 width: 100%;
//             `;
//             figure.appendChild(figcaption);
//         }

//         return { 
//             success: true, 
//             element: figure, 
//             svg,
//             code: mermaidCode
//         };
//     } catch (error) {
//         console.error(`Error in renderProgressChart (${chartType}):`, error);
//         return { 
//             success: false, 
//             error: error.message 
//         };
//     }
// };
