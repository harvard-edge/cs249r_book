// src_shadow/configs/prompt_templater.js

/**
 * Replaces template variables in a prompt with corresponding values
 * @param {string} prompt - Original prompt with {{variable}} placeholders
 * @param {Object} data - Object containing replacement values
 * @returns {string} - Processed prompt with replacements
 * 
 * Example usage:
 * const data = {
 *   progress_report: "Student progress...",
 *   understanding_level: "intermediate",
 *   topic: "ML Systems"
 * }
 * const processed = fillPromptTemplate(prompt, data)
 */
export function fillPromptTemplate(prompt, data) {
    // Early return if no data or prompt
    if (!prompt || !data) {
        console.warn('Missing prompt or data for template filling');
        return prompt;
    }

    let filledPrompt = prompt;
    
    // Find all template variables in the prompt
    const templateVariables = prompt.match(/\{\{([^}]+)\}\}/g) || [];
    
    // Track which replacements were made
    const replacementsMade = new Set();
    
    // Process each template variable
    templateVariables.forEach(template => {
        // Extract variable name without brackets
        const variableName = template.replace(/\{\{|\}\}/g, '').trim();
        
        // Check if we have data for this variable
        if (data.hasOwnProperty(variableName)) {
            filledPrompt = filledPrompt.replace(
                new RegExp(template, 'g'), 
                data[variableName]
            );
            replacementsMade.add(variableName);
        } else {
            console.warn(`No data provided for template variable: ${variableName}`);
        }
    });

    // Log which replacements were made and which data wasn't used
    const unusedData = Object.keys(data).filter(key => !replacementsMade.has(key));
    if (unusedData.length > 0) {
        console.info('Unused data keys:', unusedData);
    }

    return filledPrompt;
}

// Example usage with progress report prompt:
// export function generateProgressReportPrompt(data) {
//     const { PROGRESS_REPORT_PROMPT } = await import('./progress_report_prompts.js');
//     return fillPromptTemplate(PROGRESS_REPORT_PROMPT, data);
// }

// Usage example:
/*
const data = {
    progress_report: "Student's progress details...",
    understanding_level: "intermediate",
    topic: "Neural Networks"
};

const prompt = `
    Analyze this {{progress_report}}
    For a {{understanding_level}} student
    Focusing on {{topic}}
`;

const filledPrompt = fillPromptTemplate(prompt, data);
*/