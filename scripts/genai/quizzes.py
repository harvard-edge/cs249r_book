import argparse
import os
import re
import json
from pathlib import Path
from openai import OpenAI, APIError
from datetime import datetime
import yaml

# Gradio imports
import gradio as gr

# JSON Schema validation
from jsonschema import validate, ValidationError

# Callout class names for quiz insertion
QUIZ_QUESTION_CALLOUT_CLASS = "callout-quiz-question"
QUIZ_ANSWER_CALLOUT_CLASS = "callout-quiz-answer"

# Quiz section headers and patterns for easy maintenance
QUIZ_ANSWERS_SECTION_HEADER = "## Quiz Answers"
QUIZ_ANSWERS_SECTION_PATTERN = rf"^{re.escape(QUIZ_ANSWERS_SECTION_HEADER)}\s*{{#[^}}]*}}\s*\n.*?(?=\n##\s+|$)"
QUIZ_QUESTION_CALLOUT_PATTERN = rf"::: \{{\.{re.escape(QUIZ_QUESTION_CALLOUT_CLASS)}[^}}]*\}}\n.*?\n:::\n"
QUIZ_ANSWER_CALLOUT_PATTERN = rf"::: \{{\.{re.escape(QUIZ_ANSWER_CALLOUT_CLASS)}[^}}]*\}}\n.*?\n:::\n"

# Frontmatter patterns
FRONTMATTER_PATTERN = r'^(---\s*\n.*?\n---\s*\n)'
YAML_FRONTMATTER_PATTERN = r'^(---\s*\n.*?\n---\s*\n)'

# Section patterns
SECTION_PATTERN = r"^##\s+(.+?)(\s*\{#([\w\-]+)\})?\s*$"

# Configuration for question types, making it easy to modify or extend.
QUESTION_TYPE_CONFIG = [
    {
        "type": "MCQ",
        "description": "Best for checking definitions, comparisons, and system behaviors. Provide 3–5 options with plausible distractors. The `answer` field must explain why the correct choice is correct."
    },
    {
        "type": "TF",
        "description": "Good for testing basic understanding or challenging misconceptions. The `answer` must include a justification."
    },
    {
        "type": "SHORT",
        "description": "Encourages deeper reflection. Works well for \"Why is X necessary?\" or \"What would happen if...?\" questions."
    },
    {
        "type": "FILL",
        "description": "Useful for testing specific terminology or concepts that require precise recall. Use `____` (four underscores) for the blank. The `answer` MUST provide the missing word(s) first, followed by a period and then a brief explanation. For example: `performance gap. This gap occurs...` AVOID: Questions where the answer is obvious from context, where the blank can be filled with multiple reasonable answers, or where the answer appears in the same sentence as the blank."
    },
    {
        "type": "ORDER",
        "description": "Excellent for reinforcing processes or workflows. The `question` should list the items to be ordered, and the `answer` should present them in the correct sequence with explanations if necessary."
    },
    {
        "type": "CALC",
        "description": "For questions requiring mathematical calculation (e.g., performance estimates, metric calculations). The `answer` should show the steps of the calculation."
    }
]

# Dynamically generate the list of non-MCQ question types for the schema enum
NON_MCQ_TYPES = [q["type"] for q in QUESTION_TYPE_CONFIG if q["type"] != "MCQ"]

# Dynamically generate question guidelines for the system prompt
QUESTION_GUIDELINES = "\n".join(
    f"-   **{q['type']}**: {q['description']}" for q in QUESTION_TYPE_CONFIG
)

# Schema for individual quiz responses (used by AI generation)
JSON_SCHEMA = {
    "type": "object",
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "quiz_needed": {"type": "boolean", "const": False},
                "rationale": {"type": "string"}
            },
            "required": ["quiz_needed", "rationale"],
            "additionalProperties": False
        },
        {
            "type": "object",
            "properties": {
                "quiz_needed": {"type": "boolean", "const": True},
                "rationale": {
                    "type": "object",
                    "properties": {
                        "focus_areas": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 3
                        },
                        "question_strategy": {"type": "string"},
                        "difficulty_progression": {"type": "string"},
                        "integration": {"type": "string"},
                        "ranking_explanation": {"type": "string"}
                    },
                    "required": ["focus_areas", "question_strategy", "difficulty_progression", "integration", "ranking_explanation"],
                    "additionalProperties": False
                },
                "questions": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "question_type": {"type": "string", "const": "MCQ"},
                                    "question": {"type": "string"},
                                    "choices": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "minItems": 3,
                                        "maxItems": 5
                                    },
                                    "answer": {"type": "string"},
                                    "learning_objective": {"type": "string"}
                                },
                                "required": ["question_type", "question", "choices", "answer", "learning_objective"],
                                "additionalProperties": False
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "question_type": {
                                        "type": "string",
                                        "enum": NON_MCQ_TYPES
                                    },
                                    "question": {"type": "string"},
                                    "answer": {"type": "string"},
                                    "learning_objective": {"type": "string"}
                                },
                                "required": ["question_type", "question", "answer", "learning_objective"],
                                "additionalProperties": False
                            }
                        ]
                    },
                    "minItems": 1,
                    "maxItems": 5
                }
            },
            "required": ["quiz_needed", "rationale", "questions"],
            "additionalProperties": False
        }
    ]
}

# Schema for complete quiz files
QUIZ_FILE_SCHEMA = {
    "type": "object",
    "properties": {
        "metadata": {
            "type": "object",
            "properties": {
                "source_file": {"type": "string"},
                "total_sections": {"type": "integer"},
                "sections_with_quizzes": {"type": "integer"},
                "sections_without_quizzes": {"type": "integer"}
            },
            "required": ["source_file", "total_sections", "sections_with_quizzes", "sections_without_quizzes"],
            "additionalProperties": False
        },
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "section_id": {"type": "string"},
                    "section_title": {"type": "string"},
                    "quiz_data": {
                        "oneOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "quiz_needed": {"type": "boolean", "const": False},
                                    "rationale": {"type": "string"}
                                },
                                "required": ["quiz_needed", "rationale"],
                                "additionalProperties": False
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "quiz_needed": {"type": "boolean", "const": True},
                                    "rationale": {
                                        "type": "object",
                                        "properties": {
                                            "focus_areas": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                                "minItems": 2,
                                                "maxItems": 3
                                            },
                                            "question_strategy": {"type": "string"},
                                            "difficulty_progression": {"type": "string"},
                                            "integration": {"type": "string"},
                                            "ranking_explanation": {"type": "string"}
                                        },
                                        "required": ["focus_areas", "question_strategy", "difficulty_progression", "integration", "ranking_explanation"],
                                        "additionalProperties": False
                                    },
                                    "questions": {
                                        "type": "array",
                                        "items": {
                                            "oneOf": [
                                                {
                                                    "type": "object",
                                                    "properties": {
                                                        "question_type": {"type": "string", "const": "MCQ"},
                                                        "question": {"type": "string"},
                                                        "choices": {
                                                            "type": "array",
                                                            "items": {"type": "string"},
                                                            "minItems": 3,
                                                            "maxItems": 5
                                                        },
                                                        "answer": {"type": "string"},
                                                        "learning_objective": {"type": "string"}
                                                    },
                                                    "required": ["question_type", "question", "choices", "answer", "learning_objective"],
                                                    "additionalProperties": False
                                                },
                                                {
                                                    "type": "object",
                                                    "properties": {
                                                        "question_type": {
                                                            "type": "string",
                                                            "enum": NON_MCQ_TYPES
                                                        },
                                                        "question": {"type": "string"},
                                                        "answer": {"type": "string"},
                                                        "learning_objective": {"type": "string"}
                                                    },
                                                    "required": ["question_type", "question", "answer", "learning_objective"],
                                                    "additionalProperties": False
                                                }
                                            ]
                                        },
                                        "minItems": 1,
                                        "maxItems": 5
                                    }
                                },
                                "required": ["quiz_needed", "rationale", "questions"],
                                "additionalProperties": False
                            }
                        ]
                    }
                },
                "required": ["section_id", "section_title", "quiz_data"],
                "additionalProperties": False
            },
            "minItems": 1
        }
    },
    "required": ["metadata", "sections"],
    "additionalProperties": False
}

SYSTEM_PROMPT = f"""
You are an educational content specialist with expertise in machine learning systems, tasked with creating pedagogically sound quiz questions for a university-level textbook.
Your task is to first evaluate whether a quiz would be pedagogically valuable for the given section, and if so, generate 1 to 5 self-check questions and answers. Decide the number of questions based on the section's length and complexity.

## ML Systems Focus
Machine learning systems encompasses the full lifecycle: data pipelines, model training infrastructure, deployment, monitoring, serving, scaling, reliability, and operational concerns. Focus on system-level reasoning rather than algorithmic theory.

## Quiz Evaluation Criteria

First, evaluate if this section warrants a quiz by considering:
1. Does it contain concepts that students need to actively understand and apply?
2. Are there potential misconceptions that need to be addressed?
3. Does it present system design tradeoffs or operational implications?
4. Does it build on previous knowledge in ways that should be reinforced?

**Sections that typically DO NOT need quizzes:**
- Pure introductions or context-setting sections
- Sections that primarily provide historical context or motivation
- Sections that are purely descriptive without actionable concepts
- Overview sections without technical depth

**Sections that typically DO need quizzes:**
- Sections introducing new technical concepts or system components
- Sections presenting design decisions, tradeoffs, or operational considerations
- Sections addressing common pitfalls or misconceptions
- Sections requiring application of concepts to real scenarios
- Sections building on previous knowledge in critical ways

## Required JSON Schema

You MUST return a valid JSON object that strictly follows this schema:

```json
{json.dumps(JSON_SCHEMA, indent=2)}
```

## Output Format

If you determine a quiz is NOT needed, return the standard `quiz_needed: false` object.

If a quiz IS needed, follow the structure below. For "MCQ" questions, provide the question stem in the `question` field and the options in the `choices` array. For all other types, the `choices` field should be omitted.

```json
{{
    "quiz_needed": true,
    "rationale": {{
        "focus_areas": ["..."],
        "question_strategy": "...",
        "difficulty_progression": "...",
        "integration": "...",
        "ranking_explanation": "..."
    }},
    "questions": [
        {{
            "question_type": "MCQ",
            "question": "This is the question stem for the multiple choice question.",
            "choices": [
                "Option A",
                "Option B",
                "Option C",
                "Option D"
            ],
            "answer": "The correct answer is B. This is the explanation for why B is correct.",
            "learning_objective": "..."
        }},
        {{
            "question_type": "Short",
            "question": "This is a short answer question.",
            "answer": "This is the answer to the short answer question.",
            "learning_objective": "..."
        }}
    ]
}}
```

## Question Guidelines

**Content Focus:**
- Emphasize conceptual understanding and system-level reasoning
- Include at least one question about design tradeoffs or operational implications
- Address common misconceptions when applicable
- Avoid surface-level recall or trivia
- Connect to practical ML systems scenarios

**Question Types (use a variety based on content):**
{QUESTION_GUIDELINES}
-   **Do not** embed options (e.g., A, B, C) in the `question` string for MCQ questions; use the `choices` array instead.

**Quality Standards:**
- For `MCQ` questions, the `answer` string MUST start with `The correct answer is [LETTER].` followed by an explanation of why this choice is correct. Do NOT repeat the answer text in the explanation.
- Use clear, academically appropriate language
- Avoid repeating exact phrasing from source text
- Keep answers concise and informative (~75-150 words total per Q&A pair)
- Ensure questions collectively cover the section's main learning objectives
- Progress from basic understanding to application/analysis when multiple questions are used
- Note: Questions will appear at the end of each major section, with answers provided separately at the chapter's end. Design questions that serve as immediate comprehension checks for the section just read.

**CRITICAL ANTI-PATTERNS TO AVOID:**
- **Questions where the answer is obvious from the question itself** (e.g., fill-in-the-blank where the answer appears in the same sentence)
- **Questions that test surface-level recall without deeper understanding**
- **Questions where multiple reasonable answers could be correct**
- **Questions that simply repeat information from the text without requiring analysis**
- **Questions that test trivial facts rather than conceptual understanding**
- **Questions where the explanation just restates what's already obvious from the question**

**EXAMPLE OF A POOR FILL QUESTION:**
❌ "In ML systems, the choice between cloud and edge architectures can be influenced by ________ requirements, such as data sovereignty and privacy."
Answer: "privacy. Privacy requirements can dictate the need for data to be processed closer to its source, influencing the choice of architecture towards edge solutions to comply with regulations."

**WHY THIS IS POOR:**
- The answer "privacy" is obvious from the context (mentioned right after the blank)
- The explanation just repeats what's already stated in the question
- Tests surface-level recall, not understanding
- No deeper analysis or application required

**BETTER ALTERNATIVE:**
✅ "When designing an ML system for a healthcare application, what specific architectural consideration becomes critical due to regulatory requirements?"
Answer: "Data residency and processing location. Healthcare data must often be processed within specific geographic boundaries due to regulations like HIPAA, requiring careful consideration of whether to use cloud providers with local data centers or edge processing to keep data within jurisdictional boundaries."

**For FILL questions specifically:**
- Only use for testing specific terminology or concepts that require precise recall
- Ensure the blank tests a concept that isn't immediately obvious from the surrounding context
- The answer should be a specific term or concept, not a general word that could be inferred
- Avoid questions where the answer appears in the same sentence or immediately before/after the blank
- Test understanding of relationships, not just vocabulary recall

**Bloom's Taxonomy Mix:**
- Remember: Key terms and concepts
- Understand: Explain implications and relationships  
- Apply: Use concepts in new scenarios
- Analyze: Compare approaches and identify tradeoffs
- Evaluate: Justify design decisions
- Create: Propose solutions to system challenges

**Quality Check**
Before finalizing, ensure:
- Questions test different aspects of the content (avoid redundancy)
- At least one question addresses system-level implications
- Questions are appropriate for the textbook's target audience
- Answer explanations help reinforce learning, not just state correctness
- The response strictly follows the JSON schema provided above
- **No questions fall into the anti-patterns listed above**
"""

def update_qmd_frontmatter(qmd_file_path, quiz_file_name):
    """Adds or updates the 'quiz' key in the QMD file's YAML frontmatter using a YAML parser."""
    print(f"  Updating frontmatter in: {os.path.basename(qmd_file_path)}")
    try:
        # We use a proper YAML parser to safely handle the frontmatter.
        with open(qmd_file_path, 'r+', encoding='utf-8') as f:
            content = f.read()
            
            frontmatter_pattern = re.compile(FRONTMATTER_PATTERN, re.DOTALL)
            match = frontmatter_pattern.match(content)
            
            if match:
                # Frontmatter exists
                frontmatter_str = match.group(1)
                yaml_content_str = frontmatter_str.strip().strip('---').strip()
                
                try:
                    frontmatter_data = yaml.safe_load(yaml_content_str)
                    if not isinstance(frontmatter_data, dict):
                        frontmatter_data = {}
                except yaml.YAMLError:
                    frontmatter_data = {} # On error, start fresh to avoid corruption

                # Update the quiz key
                frontmatter_data['quiz'] = quiz_file_name
                
                # Dump back to YAML string, keeping order and format
                new_yaml_content = yaml.dump(frontmatter_data, default_flow_style=False, sort_keys=False, indent=2)
                
                new_frontmatter = f"---\n{new_yaml_content.strip()}\n---\n"
                
                # Replace old frontmatter block
                new_content = content.replace(frontmatter_str, new_frontmatter, 1)
            else:
                # No frontmatter, create it
                frontmatter_data = {'quiz': quiz_file_name}
                new_yaml_content = yaml.dump(frontmatter_data, default_flow_style=False, sort_keys=False)
                new_frontmatter = f"---\n{new_yaml_content}---\n\n"
                new_content = new_frontmatter + content
            
            f.seek(0)
            f.write(new_content)
            f.truncate()
        
        print(f"  ✓ Updated frontmatter in {os.path.basename(qmd_file_path)} with 'quiz: {quiz_file_name}'")
    except Exception as e:
        print(f"  ❌ Error updating frontmatter in {qmd_file_path}: {e}")

def extract_sections_with_ids(markdown_text):
    """
    Extracts all level-2 sections (##) with their content and section reference (e.g., {#sec-...}).
    Returns a list of dicts: {section_id, section_title, section_text}
    If any section is missing a reference, prints an error and returns None.
    """
    section_pattern = re.compile(SECTION_PATTERN, re.MULTILINE)
    all_matches = list(section_pattern.finditer(markdown_text))
    
    # Filter out "Quiz Answers" sections
    content_matches = [m for m in all_matches if m.group(1).strip().lower() != 'quiz answers']
    
    # First, validate all content sections have IDs
    missing_refs = []
    for match in content_matches:
        title = match.group(1).strip()
        ref = match.group(3)
        if not ref:
            missing_refs.append(title)
    
    if missing_refs:
        print("ERROR: The following sections are missing section reference labels (e.g., {#sec-...}):")
        for title in missing_refs:
            print(f"  - {title}")
        print("\nPlease add section references to all sections and re-run the script.")
        return None
    
    # If all sections have IDs, proceed with extraction
    sections = []
    for i, match in enumerate(content_matches):
        title = match.group(1).strip()
        ref = match.group(3)
        start = match.end()
        
        # Find the correct end position from the original list of all matches
        original_index = all_matches.index(match)
        end = all_matches[original_index + 1].start() if original_index + 1 < len(all_matches) else len(markdown_text)
        
        content = markdown_text[start:end].strip()
        # Store the full section reference including the # symbol
        full_ref = f"#{ref}"
        sections.append({
            "section_id": full_ref,
            "section_title": title,
            "section_text": content
        })
    return sections

def call_openai(client, system_prompt, user_prompt, model="gpt-4o"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.4
        )
        content = response.choices[0].message.content
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                return {"quiz_needed": False, "rationale": "No JSON found in response"}
        
        # Validate the response against JSON_SCHEMA
        try:
            validate(instance=data, schema=JSON_SCHEMA)
        except ValidationError as e:
            print(f"⚠️  Warning: AI response doesn't match schema: {e.message}")
            # Return a fallback response
            return {"quiz_needed": False, "rationale": f"Schema validation failed: {e.message}"}
        
        return data
    except APIError as e:
        return {"quiz_needed": False, "rationale": f"API error: {str(e)}"}
    except Exception as e:
        return {"quiz_needed": False, "rationale": f"Unexpected error: {str(e)}"}

def validate_individual_quiz_response(data):
    """Validate individual quiz response manually"""
    if not isinstance(data, dict):
        return False
    
    if 'quiz_needed' not in data:
        return False
    
    if not isinstance(data['quiz_needed'], bool):
        return False
    
    if 'rationale' not in data:
        return False
    
    if data['quiz_needed']:
        # Check for required fields when quiz is needed
        if 'questions' not in data:
            return False
        
        if not isinstance(data['questions'], list):
            return False
        
        if len(data['questions']) < 1 or len(data['questions']) > 5:
            return False
        
        # Validate each question
        for question in data['questions']:
            if not isinstance(question, dict):
                return False
            
            required_fields = ['question', 'answer', 'learning_objective']
            if not all(field in question for field in required_fields):
                return False
        
        # Validate rationale structure
        rationale = data['rationale']
        if not isinstance(rationale, dict):
            return False
        
        required_rationale_fields = ['focus_areas', 'question_strategy', 'difficulty_progression', 'integration', 'ranking_explanation']
        if not all(field in rationale for field in required_rationale_fields):
            return False
        
        if not isinstance(rationale['focus_areas'], list) or len(rationale['focus_areas']) < 2 or len(rationale['focus_areas']) > 3:
            return False
    
    return True

def build_user_prompt(section_title, section_text, chapter_number=None, chapter_title=None):
    """
    Build user prompt with chapter context for appropriate difficulty progression.
    
    Args:
        section_title: Title of the section
        section_text: Content of the section
        chapter_number: Chapter number (1-20)
        chapter_title: Chapter title
    """
    # Define chapter progression context
    chapter_context = ""
    difficulty_guidelines = ""
    
    if chapter_number is not None:
        chapter_context = f"""
**Chapter Context:**
This is Chapter {chapter_number}: {chapter_title or 'Unknown'} in a 20-chapter textbook on Machine Learning Systems.
The book progresses from foundational concepts to advanced topics and operational concerns.
"""
        
        # Progressive difficulty guidelines based on chapter position
        if chapter_number <= 5:
            difficulty_guidelines = """
**Chapter Difficulty Guidelines (Chapters 1-5):**
- Focus on foundational concepts and basic understanding
- Emphasize core definitions and fundamental principles
- Questions should test comprehension of basic ML systems concepts
- Avoid overly complex scenarios or advanced technical details
"""
        elif chapter_number <= 10:
            difficulty_guidelines = """
**Chapter Difficulty Guidelines (Chapters 6-10):**
- Intermediate complexity with practical applications
- Questions should test understanding of technical implementation
- Include questions about tradeoffs and design decisions
- Connect concepts to real-world ML system scenarios
"""
        elif chapter_number <= 15:
            difficulty_guidelines = """
**Chapter Difficulty Guidelines (Chapters 11-15):**
- Advanced topics requiring system-level reasoning
- Questions should test deep understanding of optimization and operational concerns
- Focus on integration of multiple concepts
- Emphasize practical implications and real-world challenges
"""
        else:
            difficulty_guidelines = """
**Chapter Difficulty Guidelines (Chapters 16-20):**
- Specialized topics requiring integration across multiple concepts
- Questions should test synthesis of knowledge from throughout the book
- Focus on ethical, societal, and advanced operational considerations
- Emphasize critical thinking about ML systems in broader contexts
"""
    
    return f"""
This section is titled "{section_title}".

{chapter_context}
{difficulty_guidelines}

Section content:
{section_text}

Generate a self-check quiz with 1 to 5 well-structured questions and answers based on this section. Include a rationale explaining your question generation strategy and focus areas. Return your response in the specified JSON format.
""".strip()

def regenerate_section_quiz(client, section_title, section_text, current_quiz_data, user_prompt, chapter_number=None, chapter_title=None):
    """
    Regenerate quiz questions for a section with custom instructions.
    
    Args:
        client: OpenAI client
        section_title: Title of the section
        section_text: Content of the section
        current_quiz_data: Current quiz data for the section
        user_prompt: User's regeneration instructions
        chapter_number: Chapter number (1-20)
        chapter_title: Chapter title
    """
    # Create a custom system prompt that includes the user's instructions
    custom_system_prompt = f"""
{SYSTEM_PROMPT}

## REGENERATION INSTRUCTIONS
The user has requested the following changes to the quiz questions:
{user_prompt}

Please regenerate the quiz questions following these specific instructions while maintaining all the quality standards and anti-patterns listed above.

## ADDITIONAL REGENERATION GUIDELINES
- Ensure the new questions address the user's specific requirements
- Maintain the same high quality standards as the original prompt
- Pay special attention to avoiding the anti-patterns listed above
- If the user requests changes to question types, ensure the new types are appropriate for the content
- Provide a brief comment explaining how you addressed the user's instructions
"""
    
    # Build the user prompt with chapter context
    user_prompt_text = build_user_prompt(section_title, section_text, chapter_number, chapter_title)
    user_prompt_text += f"\n\nPlease regenerate the quiz questions following the user's specific instructions: {user_prompt}"
    
    # Call the AI
    response = call_openai(client, custom_system_prompt, user_prompt_text)
    
    # Add a regeneration comment if the response includes one
    if isinstance(response, dict) and 'quiz_needed' in response:
        # Extract any comment from the response (this would be in the AI's response text)
        # For now, we'll add a simple comment
        response['_regeneration_comment'] = f"Regenerated based on user instructions: {user_prompt}"
    
    return response

# Gradio Application
class QuizEditorGradio:
    def __init__(self, initial_file_path=None):
        self.quiz_data = None
        self.sections = []
        self.current_section_index = 0
        self.initial_file_path = initial_file_path
        self.original_qmd_content = None
        self.qmd_file_path = None
        self.question_states = {}  # Track checked/unchecked state for each question
        
    def load_quiz_file(self, file_path=None):
        """Load a quiz JSON file"""
        # Use provided file path or initial file path
        path_to_load = file_path or self.initial_file_path
        
        if not path_to_load:
            return "No file path provided", "No file loaded", "No sections", "", ""
            
        try:
            # Check if file exists
            if not os.path.exists(path_to_load):
                return f"File not found: {path_to_load}", "File not found", "No sections", "", ""
            
            with open(path_to_load, 'r', encoding='utf-8') as f:
                self.quiz_data = json.load(f)
            
            # Validate JSON structure
            if not isinstance(self.quiz_data, dict):
                return f"Invalid JSON structure in {path_to_load}", "Invalid file format", "No sections", "", ""
            
            self.sections = self.quiz_data.get('sections', [])
            if not self.sections:
                return f"No sections found in {path_to_load}", "No sections found", "No sections", "", ""
            
            # Try to load the original .qmd file
            self.load_original_qmd_file(path_to_load)
            
            # Initialize question states for all sections
            self.initialize_question_states()
            
            self.current_section_index = 0
            
            # Load first section
            section = self.sections[0]
            title = f"{section['section_title']} ({section['section_id']})"
            section_text = self.get_full_section_content(section)
            questions_text = self.format_questions_with_buttons(section)
            nav_text = f"Section 1 of {len(self.sections)}"
            
            return title, nav_text, section_text, questions_text, ""
            
        except json.JSONDecodeError as e:
            return f"Invalid JSON in {path_to_load}: {str(e)}", "JSON Error", "No sections", "", ""
        except Exception as e:
            return f"Error loading {path_to_load}: {str(e)}", "Error loading file", "No sections", "", ""
    
    def initialize_question_states(self):
        """Initialize checked state for all questions (all checked by default)"""
        self.question_states = {}
        for i, section in enumerate(self.sections):
            section_id = section['section_id']
            quiz_data = section.get('quiz_data', {})
            if quiz_data.get('quiz_needed', False):
                questions = quiz_data.get('questions', [])
                self.question_states[section_id] = [True] * len(questions)  # All checked by default
    
    def update_question_state(self, section_id, question_index, checked):
        """Update the checked state of a question"""
        if section_id not in self.question_states:
            self.question_states[section_id] = []
        
        # Ensure the list is long enough
        while len(self.question_states[section_id]) <= question_index:
            self.question_states[section_id].append(True)
        
        self.question_states[section_id][question_index] = checked
    
    def load_original_qmd_file(self, quiz_file_path):
        """Try to load the original .qmd file based on the quiz file path"""
        try:
            # Get metadata from quiz file
            metadata = self.quiz_data.get('metadata', {})
            source_file = metadata.get('source_file')
            
            if source_file and os.path.exists(source_file):
                self.qmd_file_path = source_file
                with open(source_file, 'r', encoding='utf-8') as f:
                    self.original_qmd_content = f.read()
            else:
                # Try to find .qmd file in the same directory
                quiz_dir = os.path.dirname(quiz_file_path)
                quiz_name = os.path.splitext(os.path.basename(quiz_file_path))[0]
                
                # Look for .qmd files in the directory
                for file in os.listdir(quiz_dir):
                    if file.endswith('.qmd') and not file.startswith('.'):
                        self.qmd_file_path = os.path.join(quiz_dir, file)
                        with open(self.qmd_file_path, 'r', encoding='utf-8') as f:
                            self.original_qmd_content = f.read()
                        break
                        
        except Exception as e:
            print(f"Warning: Could not load original .qmd file: {str(e)}")
            self.original_qmd_content = None
            self.qmd_file_path = None
    
    def get_full_section_content(self, section):
        """Get the full section content from the original .qmd file"""
        if not self.original_qmd_content:
            return section.get('section_text', 'No section text available')
        
        section_id = section['section_id']
        # Remove the # prefix if present
        if section_id.startswith('#'):
            section_id = section_id[1:]
        
        # Find the section in the original content
        section_pattern = re.compile(rf'^##\s+.*?\{{\#{re.escape(section_id)}\}}.*?$', re.MULTILINE)
        match = section_pattern.search(self.original_qmd_content)
        
        if match:
            # Find the start and end of this section
            start_pos = match.start()
            
            # Find the next section or end of file
            next_section_pattern = re.compile(r'^##\s+', re.MULTILINE)
            next_match = next_section_pattern.search(self.original_qmd_content, start_pos + 1)
            
            if next_match:
                end_pos = next_match.start()
            else:
                end_pos = len(self.original_qmd_content)
            
            # Extract the full section content
            section_content = self.original_qmd_content[start_pos:end_pos].strip()
            return section_content
        else:
            # Fallback to the stored section text
            return section.get('section_text', 'No section text available')
    
    def load_from_path(self, file_path):
        """Load quiz from a file path string"""
        return self.load_quiz_file(file_path)
    
    def format_questions_with_buttons(self, section):
        """Format questions for display with status indicators"""
        quiz_data = section.get('quiz_data', {})
        
        if not quiz_data.get('quiz_needed', False):
            return "No quiz needed for this section"
        
        questions = quiz_data.get('questions', [])
        if not questions:
            return "No questions available"
        
        section_id = section['section_id']
        question_states = self.question_states.get(section_id, [True] * len(questions))
        
        formatted = []
        formatted.append("**Question Status (all questions will be kept by default):**\n\n")
        formatted.append("*Note: Currently showing question status. Use Save & Exit to keep all questions, or edit the JSON file manually to remove unwanted questions.*\n\n")
        
        for i, question in enumerate(questions):
            checked = question_states[i] if i < len(question_states) else True
            status = "✅ WILL KEEP" if checked else "❌ WILL REMOVE"
            
            q_text = format_question_for_display(question, i+1) + "\n"
            a_text = f"**A:** {question['answer']}\n\n"
            
            if 'learning_objective' in question:
                obj_text = f"**Learning Objective:** {question['learning_objective']}\n\n"
            else:
                obj_text = ""
            
            formatted.append(f"{q_text}{a_text}{obj_text}**Status:** {status}\n\n---\n\n")
        
        return "".join(formatted)
    
    def save_changes(self):
        """Save the current quiz data with removed questions"""
        if not self.quiz_data:
            return "No data to save"
        
        # Create a copy of the data to modify
        modified_data = json.loads(json.dumps(self.quiz_data))
        
        # Remove unchecked questions from each section
        for section in modified_data['sections']:
            section_id = section['section_id']
            quiz_data = section.get('quiz_data', {})
            
            if quiz_data.get('quiz_needed', False):
                questions = quiz_data.get('questions', [])
                question_states = self.question_states.get(section_id, [True] * len(questions))
                
                # Keep only checked questions
                kept_questions = []
                for i, question in enumerate(questions):
                    if i < len(question_states) and question_states[i]:
                        kept_questions.append(question)
                
                # Update the questions list
                quiz_data['questions'] = kept_questions
                
                # Update rationale if no questions remain
                if not kept_questions:
                    quiz_data['quiz_needed'] = False
                    quiz_data['rationale'] = "All questions were removed by user"
        
        # Save to the original file
        try:
            with open(self.initial_file_path, 'w', encoding='utf-8') as f:
                json.dump(modified_data, f, indent=2, ensure_ascii=False)
            return f"Saved changes to {os.path.basename(self.initial_file_path)}"
        except Exception as e:
            return f"Error saving file: {str(e)}"
    
    def toggle_question(self, section_id, question_index):
        """Toggle the checked state of a question"""
        if section_id not in self.question_states:
            self.question_states[section_id] = []
        
        # Ensure the list is long enough
        while len(self.question_states[section_id]) <= question_index:
            self.question_states[section_id].append(True)
        
        # Toggle the state
        self.question_states[section_id][question_index] = not self.question_states[section_id][question_index]
        
        # Return updated questions display
        section = next((s for s in self.sections if s['section_id'] == section_id), None)
        if section:
            return self.format_questions_with_buttons(section)
        return "Error updating question"
    
    def navigate_section(self, direction):
        """Navigate to previous or next section"""
        if not self.sections:
            return ["No file loaded", "No sections", "No content loaded"] + [False] * 5 + [""] * 5
        
        if direction == "prev" and self.current_section_index > 0:
            self.current_section_index -= 1
        elif direction == "next" and self.current_section_index < len(self.sections) - 1:
            self.current_section_index += 1
        
        section = self.sections[self.current_section_index]
        title = f"{section['section_title']} ({section['section_id']})"
        section_text = self.get_full_section_content(section)
        nav_text = f"Section {self.current_section_index + 1} of {len(self.sections)}"
        
        # Prepare checkbox states and question texts
        quiz_data = section.get('quiz_data', {})
        checkbox_states = [False] * 5
        question_texts = [""] * 5
        
        if quiz_data.get('quiz_needed', False):
            questions = quiz_data.get('questions', [])
            section_id = section['section_id']
            question_states = self.question_states.get(section_id, [True] * len(questions))
            
            for i in range(min(5, len(questions))):
                checkbox_states[i] = question_states[i] if i < len(question_states) else True
                question = questions[i]
                q_text = format_question_for_display(question, i+1) + "\n"
                a_text = f"**A:** {question['answer']}\n\n"
                if 'learning_objective' in question:
                    obj_text = f"*Learning Objective:* {question['learning_objective']}\n\n"
                else:
                    obj_text = ""
                question_texts[i] = f"{q_text}{a_text}{obj_text}"
        else:
            # No quiz needed for this section
            question_texts[0] = "**No quiz needed for this section**\n\n*This section was determined to not require a quiz based on its content.*"
        
        return [title, nav_text, section_text] + checkbox_states + question_texts
    
    def load_quiz_file_with_checkboxes(self, file_path=None):
        """Load a quiz JSON file and return data for checkboxes"""
        # Use provided file path or initial file path
        path_to_load = file_path or self.initial_file_path
        
        if not path_to_load:
            return ["No file path provided", "No file loaded", "No content loaded"] + [False] * 5 + [""] * 5
            
        try:
            # Check if file exists
            if not os.path.exists(path_to_load):
                return [f"File not found: {path_to_load}", "File not found", "No content loaded"] + [False] * 5 + [""] * 5
            
            with open(path_to_load, 'r', encoding='utf-8') as f:
                self.quiz_data = json.load(f)
            
            # Validate JSON structure
            if not isinstance(self.quiz_data, dict):
                return [f"Invalid JSON structure in {path_to_load}", "Invalid file format", "No content loaded"] + [False] * 5 + [""] * 5
            
            self.sections = self.quiz_data.get('sections', [])
            if not self.sections:
                return [f"No sections found in {path_to_load}", "No sections found", "No content loaded"] + [False] * 5 + [""] * 5
            
            # Try to load the original .qmd file
            self.load_original_qmd_file(path_to_load)
            
            # Initialize question states for all sections
            self.initialize_question_states()
            
            self.current_section_index = 0
            
            # Load first section
            section = self.sections[0]
            title = f"{section['section_title']} ({section['section_id']})"
            section_text = self.get_full_section_content(section)
            nav_text = f"Section 1 of {len(self.sections)}"
            
            # Prepare checkbox states and question texts
            quiz_data = section.get('quiz_data', {})
            checkbox_states = [False] * 5
            question_texts = [""] * 5
            
            if quiz_data.get('quiz_needed', False):
                questions = quiz_data.get('questions', [])
                section_id = section['section_id']
                question_states = self.question_states.get(section_id, [True] * len(questions))
                
                for i in range(min(5, len(questions))):
                    checkbox_states[i] = question_states[i] if i < len(question_states) else True
                    question = questions[i]
                    q_text = format_question_for_display(question, i+1) + "\n"
                    a_text = f"**A:** {question['answer']}\n\n"
                    if 'learning_objective' in question:
                        obj_text = f"*Learning Objective:* {question['learning_objective']}\n\n"
                    else:
                        obj_text = ""
                    question_texts[i] = f"{q_text}{a_text}{obj_text}"
            else:
                # No quiz needed for this section
                question_texts[0] = "**No quiz needed for this section**\n\n*This section was determined to not require a quiz based on its content.*"
            
            return [title, nav_text, section_text] + checkbox_states + question_texts
            
        except json.JSONDecodeError as e:
            return [f"Invalid JSON in {path_to_load}: {str(e)}", "JSON Error", "No content loaded"] + [False] * 5 + [""] * 5
        except Exception as e:
            return [f"Error loading {path_to_load}: {str(e)}", "Error loading file", "No content loaded"] + [False] * 5 + [""] * 5
    
    def save_changes_with_checkboxes(self, checkbox_states):
        """Save the current quiz data with removed questions based on checkbox states"""
        if not self.quiz_data or not self.sections:
            return "No data to save"
        
        # Update question states for current section
        current_section = self.sections[self.current_section_index]
        section_id = current_section['section_id']
        self.question_states[section_id] = checkbox_states[:5]  # Take first 5 values
        
        # Create a copy of the data to modify
        modified_data = json.loads(json.dumps(self.quiz_data))
        
        # Remove unchecked questions from each section
        for section in modified_data['sections']:
            section_id = section['section_id']
            quiz_data = section.get('quiz_data', {})
            
            if quiz_data.get('quiz_needed', False):
                questions = quiz_data.get('questions', [])
                question_states = self.question_states.get(section_id, [True] * len(questions))
                
                # Keep only checked questions
                kept_questions = []
                for i, question in enumerate(questions):
                    if i < len(question_states) and question_states[i]:
                        kept_questions.append(question)
                
                # Update the questions list
                quiz_data['questions'] = kept_questions
                
                # Update rationale if no questions remain
                if not kept_questions:
                    quiz_data['quiz_needed'] = False
                    quiz_data['rationale'] = "All questions were removed by user"
        
        # Save to the original file
        try:
            with open(self.initial_file_path, 'w', encoding='utf-8') as f:
                json.dump(modified_data, f, indent=2, ensure_ascii=False)
            return f"Saved changes to {os.path.basename(self.initial_file_path)}"
        except Exception as e:
            return f"Error saving file: {str(e)}"

def format_quiz_information(section, quiz_data):
    """Format quiz information for display including rationale and metadata"""
    if not quiz_data.get('quiz_needed', False):
        return "**No quiz needed for this section**\n\n" + quiz_data.get('rationale', 'No rationale provided')
    
    rationale = quiz_data.get('rationale', {})
    questions = quiz_data.get('questions', [])
    
    # Format the information
    info = f"**Quiz Information for: {section['section_title']}**\n\n"
    
    if isinstance(rationale, dict):
        # Detailed rationale with focus areas
        focus_areas = rationale.get('focus_areas', [])
        if focus_areas:
            info += "**Focus Areas:**\n"
            for i, area in enumerate(focus_areas, 1):
                info += f"{i}. {area}\n"
            info += "\n"
        
        question_strategy = rationale.get('question_strategy', '')
        if question_strategy:
            info += f"**Question Strategy:** {question_strategy}\n\n"
        
        difficulty_progression = rationale.get('difficulty_progression', '')
        if difficulty_progression:
            info += f"**Difficulty Progression:** {difficulty_progression}\n\n"
        
        integration = rationale.get('integration', '')
        if integration:
            info += f"**Integration:** {integration}\n\n"
        
        ranking_explanation = rationale.get('ranking_explanation', '')
        if ranking_explanation:
            info += f"**Question Order:** {ranking_explanation}\n\n"
    else:
        # Simple rationale string
        info += f"**Rationale:** {rationale}\n\n"
    
    # Add question count
    info += f"**Questions:** {len(questions)} question{'s' if len(questions) != 1 else ''}\n\n"
    
    # Add learning objectives summary
    learning_objectives = []
    for question in questions:
        if 'learning_objective' in question:
            learning_objectives.append(question['learning_objective'])
    
    if learning_objectives:
        info += "**Learning Objectives:**\n"
        for i, obj in enumerate(learning_objectives, 1):
            info += f"{i}. {obj}\n"
    
    return info

def format_question_for_display(question, question_number):
    """Format a question for display in the Gradio interface based on its type"""
    question_type = question.get('question_type', 'SHORT')
    question_text = question.get('question', '')
    
    if question_type == "MCQ":
        # Format MCQ with bold stem and indented choices
        formatted = f"**Q{question_number}:** {question_text}\n\n"
        
        choices = question.get('choices', [])
        for j, choice in enumerate(choices):
            letter = chr(ord('A') + j)
            formatted += f"   {letter}) {choice}\n"
        
        return formatted
    else:
        # Standard formatting for other types
        return f"**Q{question_number}:** {question_text}"

def create_gradio_interface(initial_file_path=None):
    """Create the Gradio interface"""
    editor = QuizEditorGradio(initial_file_path)
    
    # Load the quiz file immediately
    if initial_file_path:
        load_result = editor.load_quiz_file(initial_file_path)
        if isinstance(load_result, tuple):
            section_title, nav_info, section_text, questions_text, _ = load_result
        else:
            section_title, nav_info, section_text, questions_text = "Error", "Error", "Error loading file", ""
    else:
        section_title, nav_info, section_text, questions_text = "No file loaded", "No sections", "No content loaded", ""
    
    # Get initial quiz info
    if editor.sections:
        initial_section = editor.sections[0]
        quiz_info = format_quiz_information(initial_section, initial_section.get('quiz_data', {}))
    else:
        quiz_info = "No quiz information available"
    
    with gr.Blocks(title="Quiz Editor", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# Quiz Editor")
        
        # Top row with section title and navigation (50-50)
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Section Title")
                section_title_box = gr.Textbox(label="", interactive=False, value=section_title)
            
            with gr.Column(scale=1):
                gr.Markdown("### Navigation")
                nav_info_box = gr.Textbox(label="", interactive=False, value=nav_info)
        
        # Section content
        gr.Markdown("### Section Content (from .qmd file)")
        section_text_box = gr.Textbox(label="", lines=15, interactive=False, max_lines=20, value=section_text)
        
        gr.Markdown("### Generated Questions")
        
        # Dynamically create question rows based on the number of questions
        question_checkboxes = []
        question_markdowns = []
        answer_markdowns = []
        learning_obj_markdowns = []
        
        def create_question_rows(num_questions, questions):
            for i in range(num_questions):
                with gr.Row(visible=True) as row_group:
                    with gr.Column(scale=1):  # Checkboxake the checkbox sCheckbox without text
                        checkbox = gr.Checkbox(label="Select", value=True, visible=False)
                        question_checkboxes.append(checkbox)
                    with gr.Column(scale=3):
                        # Display the question text
                        question_text = f"Q: {questions[i]['question']}"
                        question_md = gr.Markdown(question_text, visible=False)
                        question_markdowns.append(question_md)  # ✅ correct list
                    with gr.Column(scale=3):
                        # Display the answer in the middle column
                        answer_text = f"**Answer:** {questions[i]['answer']}"
                        answer_md = gr.Markdown(answer_text, visible=False)
                        answer_markdowns.append(answer_md)
                    with gr.Column(scale=2):
                        # Display the learning objective in the last column
                        learning_text = f"**Learning Objective:** {questions[i].get('learning_objective', 'N/A')}"
                        learning_md = gr.Markdown(learning_text, visible=False)
                        learning_obj_markdowns.append(learning_md)
        
        # Create maximum possible question rows (5 as per schema)
        max_questions = 5
        dummy_questions = [{"question": "", "answer": "", "learning_objective": ""}] * max_questions
        create_question_rows(max_questions, dummy_questions)
        
        # Bottom row with navigation and save buttons
        with gr.Row():
            with gr.Column(scale=1):
                prev_btn = gr.Button("← Previous", size="lg")
            with gr.Column(scale=1):
                save_btn = gr.Button("💾 Save & Exit", size="lg", variant="primary")
            with gr.Column(scale=1):
                next_btn = gr.Button("Next →", size="lg")
        
        # Regeneration section
        gr.Markdown("### Regenerate Questions")
        with gr.Row():
            with gr.Column(scale=3):
                prompt_input = gr.Textbox(
                    label="Regeneration Instructions", 
                    placeholder="Enter instructions for regenerating questions (e.g., 'Make questions more challenging', 'Focus on practical applications', 'Add more multiple choice questions')",
                    lines=3,
                    max_lines=5
                )
            with gr.Column(scale=1):
                regenerate_btn = gr.Button("🔄 Regenerate", size="lg", variant="secondary")
        
        # Status display for operations (moved here, below regeneration)
        status_display = gr.Textbox(label="Status", interactive=False, visible=True, lines=3)
        
        # Quiz Information section (moved to bottom)
        gr.Markdown("### Quiz Information")
        quiz_info_display = gr.Markdown(quiz_info, visible=True)
        
        # Status display for operations (smaller, at bottom)
        status_display = gr.Textbox(label="Status", interactive=False, visible=True, lines=2)
        

        def get_section_data(section_idx):
            # Returns: section_title, nav_info, section_text, quiz_info, [checkbox_states], [question_markdowns], [answer_md], [learning_md]
            # Always returns fixed number of outputs to match interface components (max 5 questions)
            max_components = 5
            
            if not editor.sections:
                return [
                    "No file loaded", "No sections", "No content loaded", "No quiz information available",
                    *(gr.update(value=False, visible=False) for i in range(max_components)),
                    *(gr.update(value="", visible=False) for i in range(max_components)),
                    *(gr.update(value="", visible=False) for i in range(max_components)),
                    *(gr.update(value="", visible=False) for i in range(max_components))
                ]
            
            section = editor.sections[section_idx]
            title = f"{section['section_title']} ({section['section_id']})"
            section_text_val = editor.get_full_section_content(section)
            nav_text = f"Section {section_idx+1} of {len(editor.sections)}"
            
            # Add question count to navigation
            quiz_data = section.get('quiz_data', {})
            questions = quiz_data.get('questions', []) if quiz_data.get('quiz_needed', False) else []
            num_questions = len(questions)
            if num_questions > 0:
                nav_text += f" ({num_questions} question{'s' if num_questions != 1 else ''})"
            else:
                nav_text += " (no quiz)"
            
            # Format quiz information
            quiz_info = format_quiz_information(section, quiz_data)
            
            # Initialize with False/empty for all component slots
            checkbox_states = [False] * max_components
            question_markdowns = [""] * max_components
            answer_md = [""] * max_components
            learning_md = [""] * max_components
            
            if num_questions == 0:
                # No quiz needed for this section - show only first row with message
                question_markdowns[0] = "**No quiz needed for this section**"
                answer_md[0] = "*This section was determined to not require a quiz based on its content.*"
                learning_md[0] = ""
                # Only show the first row, hide the rest
                return [
                    title, nav_text, section_text_val, quiz_info,
                    *(gr.update(value=checkbox_states[i], visible=(i == 0)) for i in range(max_components)),
                    *(gr.update(value=question_markdowns[i], visible=(i == 0)) for i in range(max_components)),
                    *(gr.update(value=answer_md[i], visible=(i == 0)) for i in range(max_components)),
                    *(gr.update(value=learning_md[i], visible=(i == 0)) for i in range(max_components))
                ]
            else:
                # Get saved checkbox states for this section
                section_id = section['section_id']
                saved_states = editor.question_states.get(section_id, [True] * num_questions)
                
                # Fill in data for actual questions (up to max_components)
                for i in range(min(num_questions, max_components)):
                    # Use saved checkbox state if available, otherwise default to True
                    checkbox_states[i] = saved_states[i] if i < len(saved_states) else True
                    question_markdowns[i] = format_question_for_display(questions[i], i+1)
                    answer_md[i] = f"**Answer:** {questions[i]['answer']}"
                    learning_md[i] = f"**Learning Objective:** {questions[i].get('learning_objective', 'N/A')}"
                
                # Return gr.update() objects with proper visibility control
                return [
                    title, nav_text, section_text_val, quiz_info,
                    *(gr.update(value=checkbox_states[i], visible=(i < num_questions)) for i in range(max_components)),
                    *(gr.update(value=question_markdowns[i], visible=(i < num_questions)) for i in range(max_components)),
                    *(gr.update(value=answer_md[i], visible=(i < num_questions)) for i in range(max_components)),
                    *(gr.update(value=learning_md[i], visible=(i < num_questions)) for i in range(max_components))
                ]
        
        # Navigation handlers
        def nav_prev():
            if editor.current_section_index > 0:
                editor.current_section_index -= 1
            return get_section_data(editor.current_section_index)
        def nav_next():
            if editor.current_section_index < len(editor.sections)-1:
                editor.current_section_index += 1
            return get_section_data(editor.current_section_index)
        
        # Checkbox change handler
        def checkbox_change(*checkbox_values):
            if editor.sections and editor.current_section_index < len(editor.sections):
                current_section = editor.sections[editor.current_section_index]
                section_id = current_section['section_id']
                # Only save the checkbox states for questions that actually exist
                quiz_data = current_section.get('quiz_data', {})
                questions = quiz_data.get('questions', []) if quiz_data.get('quiz_needed', False) else []
                num_questions = len(questions)
                editor.question_states[section_id] = list(checkbox_values[:num_questions])
            return  # No output needed
        
        # Save handler - directly save changes
        def save_changes(*checkbox_values):
            if editor.sections and editor.current_section_index < len(editor.sections):
                current_section = editor.sections[editor.current_section_index]
                section_id = current_section['section_id']
                # Only save the checkbox states for questions that actually exist
                quiz_data = current_section.get('quiz_data', {})
                questions = quiz_data.get('questions', []) if quiz_data.get('quiz_needed', False) else []
                num_questions = len(questions)
                editor.question_states[section_id] = list(checkbox_values[:num_questions])
            
            # Count total changes across all sections
            total_questions_removed = 0
            for section in editor.sections:
                section_id = section['section_id']
                quiz_data = section.get('quiz_data', {})
                questions = quiz_data.get('questions', []) if quiz_data.get('quiz_needed', False) else []
                question_states = editor.question_states.get(section_id, [True] * len(questions))
                removed_count = sum(1 for state in question_states if not state)
                total_questions_removed += removed_count
            
            # Save the changes
            result = editor.save_changes_with_checkboxes(list(checkbox_values))
            
            if total_questions_removed > 0:
                return f"{result}\n\nRemoved {total_questions_removed} question(s) total."
            else:
                return f"{result}\n\nAll questions kept."
        
        # Confirmed save handler
        def confirmed_save(confirm_save, *checkbox_values):
            if not confirm_save:
                return "❌ Save cancelled"
            
            if editor.sections and editor.current_section_index < len(editor.sections):
                current_section = editor.sections[editor.current_section_index]
                section_id = current_section['section_id']
                # Only save the checkbox states for questions that actually exist
                quiz_data = current_section.get('quiz_data', {})
                questions = quiz_data.get('questions', []) if quiz_data.get('quiz_needed', False) else []
                num_questions = len(questions)
                editor.question_states[section_id] = list(checkbox_values[:num_questions])
            
            return editor.save_changes_with_checkboxes(list(checkbox_values))
        
        # Regeneration handler
        def regenerate_questions(user_prompt):
            if not user_prompt.strip():
                return "Please enter regeneration instructions"
            
            if not editor.sections or editor.current_section_index >= len(editor.sections):
                return "No section loaded"
            
            current_section = editor.sections[editor.current_section_index]
            section_title = current_section['section_title']
            section_id = current_section['section_id']
            current_quiz_data = current_section.get('quiz_data', {})
            
            # Get section text
            section_text = editor.get_full_section_content(current_section)
            
            try:
                # Initialize OpenAI client
                client = OpenAI()
                
                # Call regeneration function
                new_quiz_data = regenerate_section_quiz(
                    client, section_title, section_text, current_quiz_data, user_prompt
                )
                
                # Get the regeneration comment
                comment = new_quiz_data.pop('_regeneration_comment', '')  # Remove from data so it doesn't get saved
                
                # Update the current section with new quiz data
                current_section['quiz_data'] = new_quiz_data
                
                
                # Reset question states for this section
                if new_quiz_data.get('quiz_needed', False):
                    questions = new_quiz_data.get('questions', [])
                    editor.question_states[section_id] = [True] * len(questions)
                else:
                    editor.question_states[section_id] = []
                
                # Create status message with the model's comment
                if comment:
                    status_msg = f"✅ **Regenerated questions for '{section_title}'**\n\n{comment}"
                else:
                    status_msg = f"✅ **Regenerated questions for '{section_title}'**"
                
                return status_msg
                
            except Exception as e:
                return f"❌ Error regenerating questions: {str(e)}"
        
        # Initial load
        def initial_load():
            return get_section_data(editor.current_section_index)
        
        # Wire up components
        prev_btn.click(nav_prev, outputs=[section_title_box, nav_info_box, section_text_box, quiz_info_display] + question_checkboxes + question_markdowns + answer_markdowns + learning_obj_markdowns)
        next_btn.click(nav_next, outputs=[section_title_box, nav_info_box, section_text_box, quiz_info_display] + question_checkboxes + question_markdowns + answer_markdowns + learning_obj_markdowns)
        for cb in question_checkboxes:
            cb.change(checkbox_change, inputs=question_checkboxes, outputs=[])
        
        # Save button - directly save changes
        save_btn.click(save_changes, inputs=question_checkboxes, outputs=[status_display])
        
        # Regenerate button - updates status and refreshes the current section
        def regenerate_and_refresh(user_prompt):
            status = regenerate_questions(user_prompt)
            if status.startswith("✅"):
                # If successful, refresh the current section display and clear the prompt
                section_data = get_section_data(editor.current_section_index)
                return [status, ""] + section_data  # Clear prompt_input
            else:
                # If error, just return status and keep prompt
                return [status, user_prompt] + [""] * (1 + 2 + 1 + 1 + 5 + 5 + 5 + 5)  # status + prompt + title + nav + text + quiz_info + checkboxes + questions + answers + learning
        
        regenerate_btn.click(
            regenerate_and_refresh, 
            inputs=[prompt_input], 
            outputs=[status_display, prompt_input, section_title_box, nav_info_box, section_text_box, quiz_info_display] + question_checkboxes + question_markdowns + answer_markdowns + learning_obj_markdowns
        )
        
        interface.load(initial_load, outputs=[section_title_box, nav_info_box, section_text_box, quiz_info_display] + question_checkboxes + question_markdowns + answer_markdowns + learning_obj_markdowns)
    return interface

def run_gui(quiz_file_path=None):
    """Run the Gradio application for quiz review"""
    if not quiz_file_path:
        print("Error: Quiz file path is required for GUI mode")
        print("Usage: python quizzes.py --mode review <file_path>")
        return
    
    print(f"Launching quiz review GUI for: {quiz_file_path}")
    interface = create_gradio_interface(quiz_file_path)
    interface.launch(share=False, server_name="0.0.0.0", server_port=7860)

def show_usage_examples():
    """Show usage examples for different modes"""
    print("\n=== Usage Examples ===")
    print("\n1. Generate quizzes from a markdown file:")
    print("   python quizzes.py --mode generate -f chapter1.qmd")
    print("   python quizzes.py --mode generate -f chapter1.qmd -o my_quizzes.json")
    print("   python quizzes.py --mode generate -d ./chapters/")
    
    print("\n2. Review existing quizzes with GUI:")
    print("   python quizzes.py --mode review -f quizzes.json")
    print("   python quizzes.py --mode review -f chapter1.qmd")
    print("   # In the GUI, you can regenerate questions with custom instructions")
    
    print("\n3. Verify quiz file structure and correspondence:")
    print("   python quizzes.py --mode verify -f quizzes.json")
    print("   python quizzes.py --mode verify -f chapter1.qmd")
    print("   python quizzes.py --mode verify -d ./quiz_files/")
    
    print("\n4. Insert quizzes into markdown:")
    print("   python quizzes.py --mode insert -f chapter1.qmd")
    print("   python quizzes.py --mode insert -f quizzes.json")
    
    print("\n5. Clean quizzes from markdown files:")
    print("   python quizzes.py --mode clean -f chapter1.qmd")
    print("   python quizzes.py --mode clean -d ./chapters/")
    print("   python quizzes.py --mode clean --backup -f chapter1.qmd")
    print("   python quizzes.py --mode clean --dry-run -d ./chapters/")
    
    print("\n⚠️  IMPORTANT: You must specify either -f (file) or -d (directory) for all modes.")
    print("   The tool automatically detects file types (JSON vs QMD) and performs the appropriate action.")

def main():
    parser = argparse.ArgumentParser(
        description="Quiz generation and management tool for markdown files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode generate -f chapter1.qmd
  %(prog)s --mode generate -d ./some_dir/
  %(prog)s --mode review -f chapter1.qmd
  %(prog)s --mode review -f quizzes.json
  %(prog)s --mode insert -f chapter1.qmd
  %(prog)s --mode insert -f quizzes.json
  %(prog)s --mode clean -f chapter1.qmd
  %(prog)s --mode clean -d ./chapters/
  %(prog)s --mode verify -f chapter1.qmd
  %(prog)s --mode verify -f quizzes.json
  %(prog)s --mode verify -d ./quiz_files/

Note: You must specify either -f (file) or -d (directory) for all modes.
The tool will automatically detect file types (JSON vs QMD) and perform the appropriate action.
        """
    )
    parser.add_argument("--mode", choices=["generate", "review", "insert", "verify", "clean"], 
                       required=True, help="Mode of operation")
    parser.add_argument("-f", "--file", help="Path to a file (.qmd, .md, or .json)")
    parser.add_argument("-d", "--directory", help="Path to directory")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use (generate mode only)")
    parser.add_argument("-o", "--output", default="quizzes.json", help="Path to output JSON file (generate mode only)")
    parser.add_argument("--backup", action="store_true", help="Create backup files before cleaning")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--examples", action="store_true", help="Show usage examples")
    args = parser.parse_args()

    if args.examples:
        show_usage_examples()
        return

    # Validate that either -f or -d is provided
    if not args.file and not args.directory:
        print("Error: You must specify either -f (file) or -d (directory)")
        parser.print_help()
        return

    # Determine file type and route to appropriate function
    if args.file:
        file_ext = os.path.splitext(args.file)[1].lower()
        if file_ext in ['.json']:
            # JSON file - treat as quiz file
            if args.mode == "generate":
                print("Error: Generate mode requires a .qmd file, not a .json file")
                return
            elif args.mode == "review":
                run_review_mode_simple(args.file)
            elif args.mode == "insert":
                run_insert_mode_simple(args.file)
            elif args.mode == "verify":
                run_verify_mode_simple(args.file)
            elif args.mode == "clean":
                print("Error: Clean mode requires a .qmd file, not a .json file")
                return
        elif file_ext in ['.qmd', '.md']:
            # QMD/MD file - treat as markdown file
            if args.mode == "generate":
                run_generate_mode_simple(args.file, args)
            elif args.mode == "review":
                run_review_mode_simple(args.file)
            elif args.mode == "insert":
                run_insert_mode_simple(args.file)
            elif args.mode == "verify":
                run_verify_mode_simple(args.file)
            elif args.mode == "clean":
                run_clean_mode_simple(args.file, args)
        else:
            print(f"Error: Unsupported file extension: {file_ext}")
            print("Supported extensions: .qmd, .md, .json")
            return
    elif args.directory:
        # Directory mode
        if args.mode == "generate":
            run_generate_mode_directory(args.directory, args)
        elif args.mode == "review":
            print("Error: Review mode requires a specific file, not a directory")
            return
        elif args.mode == "insert":
            print("Error: Insert mode requires a specific file, not a directory")
            return
        elif args.mode == "verify":
            run_verify_mode_directory(args.directory)
        elif args.mode == "clean":
            run_clean_mode_directory(args.directory, args)

def run_generate_mode_simple(qmd_file, args):
    """Generate new quizzes from a markdown file"""
    print(f"=== Quiz Generation Mode (Single File) ===")
    generate_for_file(qmd_file, args)

def run_generate_mode_directory(directory, args):
    """Generate new quizzes from a directory of .qmd files"""
    print(f"=== Quiz Generation Mode (Directory) ===")
    generate_for_directory(directory, args)

def run_review_mode_simple(file_path):
    """Review and edit existing quizzes from a file (JSON or QMD)"""
    print("=== Quiz Review Mode ===")
    
    # Determine file type
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext in ['.json']:
        # JSON file - run GUI directly
        run_gui(file_path)
    elif file_ext in ['.qmd', '.md']:
        # QMD file - find corresponding quiz file first
        quiz_file_path = find_quiz_file_from_qmd(file_path)
        if quiz_file_path:
            run_gui(quiz_file_path)
        else:
            print(f"❌ No corresponding quiz file found for {file_path}")
            print("   Make sure the markdown file has 'quiz: filename.json' in its frontmatter")
    else:
        print(f"❌ Unsupported file type: {file_ext}")
        print("   Supported types: .json, .qmd, .md")

def run_insert_mode_simple(file_path):
    """Insert quizzes into markdown files"""
    print("=== Quiz Insert Mode ===")
    
    # Determine file type
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext in ['.json']:
        # JSON file - find corresponding QMD file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                quiz_data = json.load(f)
            qmd_file_path = find_qmd_file_from_quiz(file_path, quiz_data)
            if qmd_file_path:
                insert_quizzes_into_markdown(qmd_file_path, file_path)
            else:
                print(f"❌ No corresponding QMD file found for {file_path}")
                print("   Make sure the quiz file has 'source_file' in its metadata")
        except Exception as e:
            print(f"❌ Error reading quiz file: {str(e)}")
    elif file_ext in ['.qmd', '.md']:
        # QMD file - find corresponding quiz file
        quiz_file_path = find_quiz_file_from_qmd(file_path)
        if quiz_file_path:
            insert_quizzes_into_markdown(file_path, quiz_file_path)
        else:
            print(f"❌ No corresponding quiz file found for {file_path}")
            print("   Make sure the markdown file has 'quiz: filename.json' in its frontmatter")
    else:
        print(f"❌ Unsupported file type: {file_ext}")
        print("   Supported types: .json, .qmd, .md")

def run_verify_mode_simple(file_path):
    """Verify quiz files and validate their structure"""
    print("=== Quiz Verify Mode ===")
    
    # Determine file type
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext in ['.json']:
        # JSON file - verify quiz file and find corresponding QMD
        print(f"🔍 Verifying quiz file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                quiz_data = json.load(f)
            
            # Basic validation
            if not isinstance(quiz_data, dict):
                print("❌ Quiz file must be a JSON object")
                return
            
            if 'sections' not in quiz_data:
                print("❌ Quiz file missing 'sections' key")
                return
            
            sections = quiz_data.get('sections', [])
            print(f"✅ Quiz file is valid JSON")
            print(f"   - Found {len(sections)} sections")
            
            # Count questions
            total_questions = 0
            sections_with_quizzes = 0
            for section in sections:
                quiz_data_section = section.get('quiz_data', {})
                if quiz_data_section.get('quiz_needed', False):
                    sections_with_quizzes += 1
                    questions = quiz_data_section.get('questions', [])
                    total_questions += len(questions)
            
            print(f"   - Sections with quizzes: {sections_with_quizzes}")
            print(f"   - Total questions: {total_questions}")
            
            # Try to find corresponding QMD file
            metadata = quiz_data.get('metadata', {})
            source_file = metadata.get('source_file')
            if source_file:
                if os.path.exists(source_file):
                    print(f"✅ Found corresponding QMD file: {source_file}")
                else:
                    print(f"⚠️  QMD file not found: {source_file}")
            else:
                print("⚠️  No source file specified in metadata")
            
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {str(e)}")
        except Exception as e:
            print(f"❌ Error reading file: {str(e)}")
            
    elif file_ext in ['.qmd', '.md']:
        # QMD file - verify QMD file and find corresponding quiz
        print(f"🔍 Verifying QMD file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract sections
            sections = extract_sections_with_ids(content)
            if sections:
                print(f"✅ QMD file is valid")
                print(f"   - Found {len(sections)} sections")
                
                # Show section titles
                for section in sections:
                    print(f"     - {section['section_title']} ({section['section_id']})")
                
                # Try to find corresponding quiz file
                quiz_metadata = extract_quiz_metadata(content)
                if quiz_metadata:
                    quiz_path = os.path.join(os.path.dirname(file_path), quiz_metadata)
                    if os.path.exists(quiz_path):
                        print(f"✅ Found corresponding quiz file: {quiz_path}")
                    else:
                        print(f"⚠️  Quiz file not found: {quiz_path}")
                else:
                    print("⚠️  No quiz file specified in frontmatter")
            else:
                print("❌ No sections found in QMD file")
                
        except Exception as e:
            print(f"❌ Error reading file: {str(e)}")
        
    else:
        print(f"❌ Unsupported file type: {file_ext}")
        print("   Supported types: .json, .qmd, .md")

def run_clean_mode_simple(qmd_file, args):
    """Clean all quizzes from a markdown file"""
    print("=== Quiz Clean Mode (Single File) ===")
    clean_single_file(qmd_file, args)

def run_clean_mode_directory(directory, args):
    """Clean all quizzes from all QMD files in a directory"""
    print("=== Quiz Clean Mode (Directory) ===")
    clean_directory(directory, args)

def run_verify_mode_directory(directory_path):
    """Verify all quiz files in a directory"""
    print("=== Quiz Verify Mode (Directory) ===")
    run_verify_directory(directory_path)

def extract_quiz_metadata(content):
    """Extract quiz file name from YAML frontmatter"""
    frontmatter_pattern = re.compile(FRONTMATTER_PATTERN, re.DOTALL)
    match = frontmatter_pattern.match(content)
    if match:
        frontmatter = match.group(1)
        # Look for quiz: filename.json
        quiz_match = re.search(r'^quiz:\s*(.+)$', frontmatter, re.MULTILINE)
        if quiz_match:
            return quiz_match.group(1).strip()
    return None

def find_quiz_file_from_qmd(qmd_file_path):
    """Find the corresponding quiz file for a QMD file"""
    try:
        with open(qmd_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        quiz_file = extract_quiz_metadata(content)
        if quiz_file:
            quiz_path = os.path.join(os.path.dirname(qmd_file_path), quiz_file)
            if os.path.exists(quiz_path):
                return quiz_path
    except Exception:
        pass
    return None

def find_qmd_file_from_quiz(quiz_file_path, quiz_data):
    """Find the corresponding QMD file for a quiz file"""
    metadata = quiz_data.get('metadata', {})
    source_file = metadata.get('source_file')
    if source_file and os.path.exists(source_file):
        return source_file
    return None

def extract_chapter_info(file_path):
    """Extract chapter number and title from file path"""
    # Map file paths to chapter numbers based on the book outline
    chapter_mapping = {
        'introduction': (1, 'Introduction'),
        'ml_systems': (2, 'ML Systems'),
        'dl_primer': (3, 'DL Primer'),
        'dnn_architectures': (4, 'DNN Architectures'),
        'workflow': (5, 'AI Workflow'),
        'data_engineering': (6, 'Data Engineering'),
        'frameworks': (7, 'AI Frameworks'),
        'training': (8, 'AI Training'),
        'efficient_ai': (9, 'Efficient AI'),
        'optimizations': (10, 'Model Optimizations'),
        'hw_acceleration': (11, 'AI Acceleration'),
        'benchmarking': (12, 'Benchmarking AI'),
        'ops': (13, 'ML Operations'),
        'ondevice_learning': (14, 'On-Device Learning'),
        'privacy_security': (15, 'Security & Privacy'),
        'responsible_ai': (16, 'Responsible AI'),
        'sustainable_ai': (17, 'Sustainable AI'),
        'robust_ai': (18, 'Robust AI'),
        'ai_for_good': (19, 'AI for Good'),
        'conclusion': (20, 'Conclusion')
    }
    
    # Extract chapter name from file path
    file_name = os.path.basename(file_path).replace('.qmd', '').replace('.md', '')
    
    for key, (number, title) in chapter_mapping.items():
        if key in file_name:
            return number, title
    
    return None, None

def generate_for_file(qmd_file, args):
    """Generate quizzes for a single QMD file"""
    print(f"Generating quizzes for: {qmd_file}")
    
    try:
        # Read the QMD file
        with open(qmd_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract sections
        sections = extract_sections_with_ids(content)
        if not sections:
            print("❌ No sections found or sections missing IDs")
            return
        
        # Extract chapter info
        chapter_number, chapter_title = extract_chapter_info(qmd_file)
        
        print(f"Found {len(sections)} sections")
        if chapter_number:
            print(f"Chapter {chapter_number}: {chapter_title}")
        
        # Initialize OpenAI client
        client = OpenAI()
        
        # Generate quizzes for each section
        quiz_sections = []
        sections_with_quizzes = 0
        sections_without_quizzes = 0
        
        for i, section in enumerate(sections):
            print(f"\nProcessing section {i+1}/{len(sections)}: {section['section_title']}")
            
            # Build user prompt with chapter context
            user_prompt = build_user_prompt(
                section['section_title'], 
                section['section_text'],
                chapter_number,
                chapter_title
            )
            
            # Call OpenAI
            response = call_openai(client, SYSTEM_PROMPT, user_prompt, args.model)
            
            if response.get('quiz_needed', False):
                sections_with_quizzes += 1
                print(f"  ✅ Generated quiz with {len(response.get('questions', []))} questions")
            else:
                sections_without_quizzes += 1
                print(f"  ⏭️  No quiz needed: {response.get('rationale', 'No rationale provided')}")
            
            quiz_sections.append({
                'section_id': section['section_id'],
                'section_title': section['section_title'],
                'quiz_data': response
            })
        
        # Create quiz file structure
        quiz_data = {
            'metadata': {
                'source_file': os.path.abspath(qmd_file),
                'total_sections': len(sections),
                'sections_with_quizzes': sections_with_quizzes,
                'sections_without_quizzes': sections_without_quizzes
            },
            'sections': quiz_sections
        }
        
        # Save to output file
        output_file = args.output
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(quiz_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Quiz generation complete!")
        print(f"   - Output file: {output_file}")
        print(f"   - Sections with quizzes: {sections_with_quizzes}")
        print(f"   - Sections without quizzes: {sections_without_quizzes}")
        
        # Update QMD frontmatter
        update_qmd_frontmatter(qmd_file, os.path.basename(output_file))
        
    except Exception as e:
        print(f"❌ Error generating quizzes: {str(e)}")

def generate_for_directory(directory, args):
    """Generate quizzes for all QMD files in a directory"""
    print(f"Generating quizzes for directory: {directory}")
    
    qmd_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.qmd') or file.endswith('.md'):
                qmd_files.append(os.path.join(root, file))
    
    if not qmd_files:
        print("❌ No QMD files found in directory")
        return
    
    print(f"Found {len(qmd_files)} QMD files")
    
    for qmd_file in qmd_files:
        print(f"\n{'='*60}")
        # Create output file name based on input file
        base_name = os.path.splitext(os.path.basename(qmd_file))[0]
        args.output = f"{base_name}_quizzes.json"
        generate_for_file(qmd_file, args)

def insert_quizzes_into_markdown(qmd_file_path, quiz_file_path):
    """Insert quizzes into a markdown file"""
    print(f"Inserting quizzes from {quiz_file_path} into {qmd_file_path}")
    # Implementation would go here - this is a placeholder
    print("❌ Insert functionality not yet implemented")

def clean_single_file(qmd_file, args):
    """Clean quizzes from a single QMD file"""
    print(f"Cleaning quizzes from: {qmd_file}")
    
    try:
        # Read the QMD file
        with open(qmd_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Create backup if requested
        if args.backup:
            backup_file = f"{qmd_file}.backup"
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  📋 Created backup: {backup_file}")
        
        # Remove quiz callouts using the defined constants
        # Pattern to match quiz question callouts (more flexible for additional attributes)
        quiz_question_pattern = re.compile(QUIZ_QUESTION_CALLOUT_PATTERN, re.DOTALL)
        
        # Pattern to match quiz answer callouts (more flexible for additional attributes)
        quiz_answer_pattern = re.compile(QUIZ_ANSWER_CALLOUT_PATTERN, re.DOTALL)
        
        # Count how many callouts we find
        question_matches = quiz_question_pattern.findall(content)
        answer_matches = quiz_answer_pattern.findall(content)
        
        if args.dry_run:
            print(f"  🔍 DRY RUN - Would remove:")
            print(f"     - {len(question_matches)} quiz question callouts")
            print(f"     - {len(answer_matches)} quiz answer callouts")
            print(f"     - {len(quiz_answers_matches)} quiz answers section(s)")
            return
        
        # Remove the callouts
        content = quiz_question_pattern.sub('', content)
        content = quiz_answer_pattern.sub('', content)
        
        # Remove the entire "Quiz Answers" section if it exists
        quiz_answers_pattern = re.compile(QUIZ_ANSWERS_SECTION_PATTERN, re.DOTALL | re.MULTILINE)
        quiz_answers_matches = quiz_answers_pattern.findall(content)
        content = quiz_answers_pattern.sub('', content)
        
        # Remove quiz frontmatter entry using YAML processing
        frontmatter_pattern = re.compile(YAML_FRONTMATTER_PATTERN, re.DOTALL)
        match = frontmatter_pattern.match(content)
        
        if match:
            frontmatter_str = match.group(1)
            yaml_content_str = frontmatter_str.strip().strip('---').strip()
            
            try:
                frontmatter_data = yaml.safe_load(yaml_content_str)
                if isinstance(frontmatter_data, dict) and 'quiz' in frontmatter_data:
                    # Remove the quiz key
                    del frontmatter_data['quiz']
                    
                    # Reconstruct frontmatter
                    new_yaml_content = yaml.dump(frontmatter_data, default_flow_style=False, sort_keys=False, indent=2)
                    new_frontmatter = f"---\n{new_yaml_content.strip()}\n---\n"
                    
                    # Replace old frontmatter
                    content = content.replace(frontmatter_str, new_frontmatter, 1)
                    print(f"  ✅ Removed quiz frontmatter entry")
            except yaml.YAMLError:
                print(f"  ⚠️  Warning: Could not parse YAML frontmatter")
        
        # Write back to file
        with open(qmd_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  ✅ Removed {len(question_matches)} quiz question callouts")
        print(f"  ✅ Removed {len(answer_matches)} quiz answer callouts")
        if len(quiz_answers_matches) > 0:
            print(f"  ✅ Removed {len(quiz_answers_matches)} quiz answers section(s)")
        
    except Exception as e:
        print(f"❌ Error cleaning file: {str(e)}")

def clean_directory(directory, args):
    """Clean quizzes from all QMD files in a directory"""
    print(f"Cleaning quizzes from directory: {directory}")
    
    qmd_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.qmd') or file.endswith('.md'):
                qmd_files.append(os.path.join(root, file))
    
    if not qmd_files:
        print("❌ No QMD files found in directory")
        return
    
    print(f"Found {len(qmd_files)} QMD files")
    
    if args.dry_run:
        print("🔍 DRY RUN - Would clean the following files:")
        for qmd_file in qmd_files:
            print(f"  - {qmd_file}")
        return
    
    for i, qmd_file in enumerate(qmd_files, 1):
        print(f"\n[{i}/{len(qmd_files)}] Cleaning: {qmd_file}")
        try:
            # Create a temporary args object for this file
            file_args = argparse.Namespace()
            file_args.backup = args.backup
            file_args.dry_run = args.dry_run
            clean_single_file(qmd_file, file_args)
        except Exception as e:
            print(f"❌ Error cleaning {qmd_file}: {str(e)}")
    
    print(f"\n✅ Clean operation complete for {len(qmd_files)} files")

def run_verify_directory(directory_path):
    """Verify all quiz files in a directory"""
    print(f"Verifying quiz files in directory: {directory_path}")
    # Implementation would go here - this is a placeholder
    print("❌ Verify directory functionality not yet implemented")

if __name__ == "__main__":
    main()

