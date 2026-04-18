export const QUERYAGENTPROCESS = ["Vectorizing query", "Cosine searching vector database", "Retrieving relevant sentences", "Calling LLM model", "Returning results"]
export const EXPLAINAGENTPROCESS = ["Retrieving relevant context from website", "Calling LLM model", "Returning results"]
export const GENERALAGENTPROCESS = ["Retrieving relevant sentences", "Checking conversation history", "Calling LLM model", "Returning results"]
export const QUIZAGENTPROCESS = [ "Retrieving relevant sentences from website", "Generating possible questions", "Calling LLM model", "Returning results"]
export const RESEARCHAGENTPROCESS = [ "Retrieving relevant papers from Arxiv", "Generating possible questions", "Calling LLM model", "Returning results"]
export const PROGRESSREPORTAGENTPROCESS = [ "Retrieving relevant papers from Arxiv", "Generating possible questions", "Calling LLM model", "Returning results"]
export const OFFLINE_PROCESS = [
  "Checking network status",
  "Analyzing your query",
  "Searching local content",
  "Computing text similarities",
  "Preparing offline response"
];

const MAIN_TOPIC = 'MLSysBook.AI: Principles and Practices of Machine Learning Systems Engineering'
export let SYSTEM_PROMPT_ORIG = `Please try to provide useful, helpful and actionable answers. Stick to topics related to ${MAIN_TOPIC}.`

export const DIFFICULTY_LEVELS = [
    "You are convercing with a Beginner learner: Focus on foundational concepts, definitions, and straightforward applications in machine learning systems, suitable for learners with little to no prior knowledge.",
    "You are convercing with a Intermediate learner: Emphasize problem-solving, system design, and practical implementations, targeting learners with a basic understanding of machine learning principles.",
    "You are convercing with a Advanced learner: Challenge learners to analyze, innovate, and optimize complex machine learning systems, requiring deep expertise and a holistic grasp of advanced techniques.",
    "You are an expert ML teacher using Bloom's Taxonomy: Create responses that progress through Bloom's levels: remember, understand, apply, analyze, evaluate, and create. Guide my learning."
  ];

  export const PROGRESS_REPORT_PROMPT = `
As an ML Systems Engineering professor, analyze this student's learning patterns:

Your report should in bullet points, easy to read, concise and point to specific details.

CURRENT PROGRESS REPORT:
{{progress_report}}

${
    // Conditional section for previous evaluation
    '{{has_previous_evaluation}}' ? 
    `
PREVIOUS EVALUATION:
{{previous_evaluation}}

!! End of previous evaluation !!

PROGRESS COMPARISON:
- Compare performance changes since last evaluation
- Highlight improvements and new challenges
- Note any shifting patterns in question types
- Identify topics that have shown most growth
    ` : ''
}

1. QUESTION PATTERN ANALYSIS
- What types of questions do they consistently get right vs wrong?
- Are they stronger in:
  * Theory or practical application?
  * System design or implementation?
  * High-level concepts or technical details?
- Which topics show repeated mistakes?

2. LEARNING PATTERNS
- Identify topics with high success (>80%)
- Note areas of struggle (<70%)
- Spot any recurring misconceptions
- Highlight improvement trends

3. RECOMMENDATIONS
- List 3 specific topics to focus on next
- Suggest study methods based on their success patterns
- Provide targeted practice for weak areas

4. ACTION PLAN
- 3 concrete steps to improve understanding
- Specific practice exercises for problem areas
- Next chapters/sections to tackle

Remember to surround key ML terms with double slashes (e.g., \\machine learning\\).
Keep feedback encouraging but direct.
`;

export const MERMAID_DIAGRAM =  "Create a mermaid diagram based on the following description. Your response should follow this format:\n\n```mermaid\n[Your mermaid diagram code here]\n```\n\n$$ [Your markdown caption explaining the diagram here] $$\n\nMake the diagram clear, well-structured, and ensure it follows mermaid syntax.\nAdditional requirements:\n- Use appropriate diagram type (flowchart, sequence, class, etc.)\n- Include clear labels and relationships\n- Keep the diagram focused and readable\n- Beneath the diagram code, between two $$<description>$$ add a markdown caption to explain the image. Ensure the diagram grows equally more in height, and is oriented vertically to avoid it being overly wide.\nDescription: "


export const quiz_prompt = 
`
Create a quiz from a CHAPTER SECTION. The quiz should have 3 questions in JSON format:
- Q1 & Q2: Directly related to the quote's content.
- Q3: Requires deeper inference.

IMPORTANT: For each question, include a "sourceReference" field that contains the exact text from the source content that the question is based on. This helps users understand where the question content originated.

Use this JSON template, modifying it as needed:

{
  "questions": [
    {
      "question": "Q1 here?",
      "sourceReference": "Exact text from source content that this question is based on",
      "answers": [
        {"text": "A1", "correct": true/false, "explanation": "short explanation"},
        {"text": "A2", "correct": false, "explanation": "short explanation"},
        {"text": "A3", "correct": false, "explanation": "short explanation"},
      ]
    },
    {"question": "Q2 here?", "sourceReference": "Source text for Q2", "answers": [/* options */]},
    {"question": "Q3 here?", "sourceReference": "Source text for Q3", "answers": [/* options */]}
  ]
}


Adjust content for quote, understanding level, questions, and answers.
CHAPTER SECTION: `



export const QUIZ_SUMMATIVE_PROMPT = 
`
Create a summative test. The test should have 10 questions in JSON format:
- Q1 - Q5: Directly related to the quote's content.
- Q5-Q10: Requires deeper inference.

IMPORTANT: For each question, include a "sourceReference" field that contains the exact text from the source content that the question is based on. This helps users understand where the question content originated.

Use this JSON template, modifying it as needed:

{
  "questions": [
    {
      "question": "Q1 here?",
      "sourceReference": "Exact text from source content that this question is based on",
      "answers": [
        {"text": "A1", "correct": true/false, "explanation": "short explanation"},
        {"text": "A2", "correct": false, "explanation": "short explanation"},
        {"text": "A3", "correct": false, "explanation": "short explanation"},
      ]
    },
    {"question": "Q2 here?", "sourceReference": "Source text for Q2", "answers": [/* options */]},
    {"question": "Q3 here?", "sourceReference": "Source text for Q3", "answers": [/* options */]}
  ]
}


Adjust content for quote, understanding level, questions, and answers.
CHAPTER SECTION: `

const explain_prompt = 
"Explain the following quote to a beginner, including any necessary background knowledge, in markdown format. Do not repeat the quote or add commentary. At the end of your response, place '%%%' 3 percent signs, and then write 2 to 3 followup questions that the user might ask to expand their knowledge."


export const configs_explain = {
    prompt: explain_prompt,// "Given the following quote: {quote}+{background_knowledge}, and the users understanding level of the material which is a scale from 1 to 10, where 1 is complete beginner and 10 is mastery, understanding: {understanding}, explain the quote as you might to someone with this understanding. Give concise inuition and example. No commentary. Do not mention in your answer anything about the user's understanding level. If quote is empty, remind user to first highlight text. Return as markdown.", // You will also create 5 multiple choice quiz questions, each question has 3 choices, in which one of them is correct.",
    quote: "",
    comments: "",
    background_knowledge: "",
    power_up: 20,
    understanding: 3,
}

export const configs_question = {

    prompt: quiz_prompt,
    quote: "",
    comments: "",
    power_up: 20,
    understanding: 3,
}

export const config_mermaid_diagram = {
  prompt: MERMAID_DIAGRAM,
}
const query_prompt = `Given the following BACKGROUND_KNOWLEDGE from the current webpage, and a student's QUESTION, answer the QUESTION. 

BACKGROUND_KNOWLEDGE:
{{background_context}}

INSTRUCTIONS:
- Use the background knowledge above to inform your response when relevant
- Only reference information that directly relates to the user's question  
- When referencing background content, use the new custom reference syntax: {{ref:target-id:display-text:source-url}}
- The target-id should be the paragraph ID from the background knowledge (e.g., p-1757100723082-nx9p0zxmp)
- The display-text should be a clean, readable description of what you're referencing
- The source-url should be the URL of the page where the referenced content is located
- Examples: {{ref:p-1757100723082-nx9p0zxmp:See paragraph 2:https://example.com/page1}}, {{ref:my-section:Go to section:https://example.com/page2}}, {{ref:conclusion:Jump to conclusion:https://example.com/page3}}
- If the background knowledge doesn't contain relevant information, rely on your general knowledge
- Ensure all references are accurate and properly attributed
- You may use the BACKGROUND_KNOWLEDGE as reference. Otherwise, you can be creative and respond in a way that always ensure you MUST answer the question
- At the end of your response, place '%%%' 3 percent signs, and then write 2 to 3 followup questions that the user might ask to expand their knowledge
- Return as markdown`

export const configs_query = {
    prompt: query_prompt, // "Given the follow conversation_history (you you can ignore if it's empty): {conversation_history}, and this information from this website: {background_knowledge}, a student's question: {question}, and an 'understanding' level from 1 to 10, {understanding}, explain the query in relation to the quote: {quote}. If quote is empty then explain based on your own knowledge base. Do not mention understanding level in your response or anything in case conversation history is empty. Return as markdown", // You will also create 5 multiple choice quiz questions, each question has 3 choices, in which one of them is correct.",
    conversation_history: "",
    question: "",
    query: "",
    quote: "",
    background_knowledge: "",
    comments: "",
    power_up: 20,
    understanding: 3,
}

// Create a config object for progress reports
export const configs_progress_report = {
    prompt: PROGRESS_REPORT_PROMPT,
    understanding: 3,
    power_up: 20
};

export const configs_quiz_summative = {
  prompt: QUIZ_SUMMATIVE_PROMPT,
  quote: "",
  comments: "",
  understanding: 3,
  power_up: 20
};

export function getConfigs(type = "explain") {
  let originalConfig;  // Store the original configuration
  let config;          // Current configuration to be modified

  // Define configurations based on type
  if (type === "explain") {
      originalConfig = {...configs_explain};
  } else if (type === "quiz") {
      originalConfig = {...configs_question};
  } else if (type === "summative") {
      originalConfig = {...configs_quiz_summative};
  } else if (type === "query") {
      originalConfig = {...configs_query};
  } else if (type === "progress_report") {
      originalConfig = {...configs_progress_report};  // Use the new config object
  } else if (type === "mermaid_diagram") {
      originalConfig = {...config_mermaid_diagram};
  } else {
      throw new Error(`Unsupported config type: ${type}`);
  }

  config = {...originalConfig};  // Copy the original config to be manipulated

  function set_field(field, value) {
      if (config.hasOwnProperty(field)) {
          config[field] = value;
      } else {
          throw new Error(`Field ${field} does not exist in config.`);
      }
  }

  function get_field(field) {
      if (config.hasOwnProperty(field)) {
          return config[field];
      } else {
          throw new Error(`Field ${field} does not exist in config.`);
      }
  }

  function make_field(field, value) {
    if (!config.hasOwnProperty(field)) {
      config[field] = value;
    } else {
        this.set_field(field, value);
    }
  }

  function return_all_fields() {
      return config;
  }

  // Reset the configuration to its original state
  function reset_fields() {
      config = {...originalConfig};
  }

  // Return the config object with the getter, setter, and reset functions
  return {
      set_field,
      get_field,
      make_field,
      return_all_fields,
      reset_fields
  };
}

export const NO_QUIZZES_MESSAGE = `
# Getting Started with Your Learning Journey 🚀

## Welcome to Interactive Learning!

It looks like you haven't taken any quizzes yet. Here's how to get the most out of your learning experience:

### How to Find Quizzes 📚
- Look for the **Quiz** button at the end of each section
- Every chapter contains multiple interactive quizzes
- Start with earlier chapters and progress forward

### Benefits of Taking Quizzes 🎯
- **Reinforce Your Learning**: Test your understanding right after studying
- **Track Progress**: Watch your scores improve over time
- **Identify Gaps**: Discover which topics need more attention
- **Build Confidence**: Master concepts through practice

### Tips for Success 💡
1. Take quizzes right after reading a section
2. Retry quizzes after a few days to test retention
3. Review explanations for incorrect answers
4. Use quiz results to guide your study focus

### Ready to Start? 
👉 Head to any chapter section and look for the quiz button at the bottom!

---
*Remember: Learning is a journey, not a race. Take your time to understand each concept thoroughly.*
`;

export const NO_NEW_QUIZZES_MESSAGE = `
# Progress Report Update 📊

## No New Quiz Activity

It looks like you haven't taken any new quizzes since your last progress report. Here are some suggestions to keep your learning momentum:

### Ways to Stay Active 🎯
1. **Review Previous Material**
   - Retake quizzes from earlier sections
   - Try to improve your previous scores
   - Focus on questions you found challenging before

2. **Move Forward**
   - Explore new chapters and sections
   - Take quizzes in sections you haven't attempted yet
   - Challenge yourself with more advanced material

### Quick Tips 💡
- Set a goal to complete at least one new quiz daily
- Use the explanation feature to deepen your understanding
- Track your progress regularly to stay motivated

Ready to jump back in? Pick a section and start a new quiz! 

---
*Your learning journey continues - keep pushing forward!*
`;

export const PROGRESS_XY_CHART = `Create a mermaid XY chart showing the user's quiz performance over time. Your response should follow this format:

\`\`\`mermaid
xychart-beta
    title "Quiz Performance Over Time"
    x-axis [dates in chronological order]
    y-axis "Score %" 0 --> 100
    bar [percentage scores]
    line [percentage scores]
\`\`\`

$$ [Your markdown caption explaining the performance trends] $$

Requirements:
- Use the quiz history data to plot performance over time
- X-axis should show dates in MM/DD format
- Y-axis should show percentage scores from 0-100
- Include both bar and line representations
- Keep the chart focused on the last 10 attempts
- Caption should highlight key trends and improvements

Quiz History Data: {{quiz_history}}`;

export const PROGRESS_QUADRANT = `Analyze the student's quiz performance across different ML Systems Engineering topics and create a quadrant chart. Your response should follow this format:

\`\`\`mermaid
quadrantChart
    title ML Systems Engineering Competency Matrix
    x-axis Basic Understanding --> Advanced Application
    y-axis Theory --> Implementation
    quadrant-1 Advanced Theory
    quadrant-2 Complete Mastery
    quadrant-3 Needs Improvement
    quadrant-4 Strong Implementation
    {{topic_points}}
\`\`\`

$$ [Your markdown caption analyzing the student's strengths and areas for improvement] $$

Topic Categories to Analyze:
1. System Architecture & Design
   - System components
   - Scalability patterns
   - Integration approaches

2. ML Pipeline Development
   - Data processing
   - Model training
   - Deployment workflows

3. Model Operations
   - Monitoring
   - Performance optimization
   - Resource management

4. ML Infrastructure
   - Computing resources
   - Storage solutions
   - Serving infrastructure

Requirements:
- Plot each topic category based on theoretical understanding vs practical implementation
- Use quiz performance data to determine positions
- Position values should be between 0 and 1
- Provide specific recommendations in the caption

Progress Report Data: {{progress_report}}`;

// Text extraction configuration
export const TEXT_EXTRACTION_CONFIG = {
  MAX_TOKENS: 30000, // Maximum tokens for text extraction
  NAV_SELECTORS: [
    'nav', 'header', 'footer', 'aside',
    '.navigation', '.nav', '.navbar', '.menu',
    '.sidebar', '.header', '.footer',
    '[role="navigation"]', '[role="banner"]', '[role="contentinfo"]',
    '.breadcrumb', '.pagination', '.toc'
  ],
  CONTENT_SELECTORS: [
    'main', 'article', '.content', '.main-content',
    '.post-content', '.entry-content', '.chapter-content'
  ]
};
