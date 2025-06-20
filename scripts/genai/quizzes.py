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
        "description": "Useful for terminology. Use `____` (four underscores) for the blank. The `answer` MUST provide the missing word(s) first, followed by a period and then a brief explanation. For example: `performance gap. This gap occurs...`"
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
- For `MCQ` questions, the `answer` string MUST start with `The correct answer is [LETTER].` followed by the text of the correct choice and then an explanation.
- Use clear, academically appropriate language
- Avoid repeating exact phrasing from source text
- Keep answers concise and informative (~75-150 words total per Q&A pair)
- Ensure questions collectively cover the section's main learning objectives
- Progress from basic understanding to application/analysis when multiple questions are used
- Note: Questions will appear at the end of each major section, with answers provided separately at the chapter's end. Design questions that serve as immediate comprehension checks for the section just read.

**Bloom's Taxonomy Mix:**
- Remember: Key terms and concepts
- Understand: Explain implications and relationships  
- Apply: Use concepts in new scenarios
- Analyze: Compare approaches and identify tradeoffs
- Evaluate: Justify design decisions
- Create: Propose solutions to system challenges

## Quality Check
Before finalizing, ensure:
- Questions test different aspects of the content (avoid redundancy)
- At least one question addresses system-level implications
- Questions are appropriate for the textbook's target audience
- Answer explanations help reinforce learning, not just state correctness
- The response strictly follows the JSON schema provided above
"""

def update_qmd_frontmatter(qmd_file_path, quiz_file_name):
    """Adds or updates the 'quiz' key in the QMD file's YAML frontmatter using a YAML parser."""
    print(f"  Updating frontmatter in: {os.path.basename(qmd_file_path)}")
    try:
        # We use a proper YAML parser to safely handle the frontmatter.
        with open(qmd_file_path, 'r+', encoding='utf-8') as f:
            content = f.read()
            
            frontmatter_pattern = re.compile(r'^(---\s*\n.*?\n---\s*\n)', re.DOTALL)
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
    section_pattern = re.compile(r"^##\s+(.+?)(\s*\{#([\w\-]+)\})?\s*$", re.MULTILINE)
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

def build_user_prompt(section_title, section_text):
    return f"""
This section is titled \"{section_title}\".

Section content:
{section_text}

Generate a self-check quiz with 1 to 5 well-structured questions and answers based on this section. Include a rationale explaining your question generation strategy and focus areas. Return your response in the specified JSON format.
""".strip()

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
    print("   python quizzes.py --mode generate --file chapter1.qmd")
    print("   python quizzes.py --mode generate --file chapter1.qmd --output my_quizzes.json")
    print("   python quizzes.py --mode generate --directory ./chapters/")
    
    print("\n2. Review existing quizzes with GUI:")
    print("   python quizzes.py --mode review quizzes.json  # Positional argument")
    print("   python quizzes.py --mode review chapter1.qmd  # Positional argument (finds quiz file)")
    print("   # In the GUI, you can regenerate questions with custom instructions")
    
    print("\n3. Verify quiz file structure and correspondence:")
    print("   python quizzes.py --mode verify -q quizzes.json")
    print("   python quizzes.py --mode verify -q chapter1.qmd  # Will find corresponding quiz file")
    print("   python quizzes.py --mode verify --file chapter1.qmd")
    print("   python quizzes.py --mode verify --directory ./quiz_files/")
    
    print("\n4. Insert quizzes into markdown (future feature):")
    print("   python quizzes.py --mode insert chapter1.qmd  # Finds quiz file automatically")
    print("   python quizzes.py --mode insert quizzes.json  # Finds markdown file automatically")
    
    print("\n5. Clean quizzes from markdown files:")
    print("   python quizzes.py --mode clean --file chapter1.qmd")
    print("   python quizzes.py --mode clean --directory ./chapters/")
    print("   python quizzes.py --mode clean --backup --file chapter1.qmd")
    print("   python quizzes.py --mode clean --dry-run --directory ./chapters/")
    
    print("\n6. Default mode (generate):")
    print("   python quizzes.py --file chapter1.qmd")
    
    print("\n⚠️  IMPORTANT: In generate mode, you MUST use -f/--file or -d/--directory options.")
    print("   Do NOT pass files as positional arguments.")

def main():
    parser = argparse.ArgumentParser(
        description="Quiz generation and management tool for markdown files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode generate --file chapter1.qmd
  %(prog)s --mode generate --directory ./some_dir/
  %(prog)s --mode review quizzes.json  # Positional argument
  %(prog)s --mode review chapter1.qmd  # Positional argument (finds quiz file)
  %(prog)s --mode insert chapter1.qmd  # Finds quiz file automatically
  %(prog)s --mode insert quizzes.json  # Finds markdown file automatically
  %(prog)s --mode clean --file chapter1.qmd
  %(prog)s --mode clean --directory ./chapters/
  %(prog)s --mode clean --backup --file chapter1.qmd
  %(prog)s --mode clean --dry-run --directory ./chapters/
  %(prog)s --mode verify -q quizzes.json
  %(prog)s --mode verify -q chapter1.qmd  # Will find corresponding quiz file
  %(prog)s --mode verify --directory ./quiz_files/
  %(prog)s --mode verify --file chapter1.qmd

Note: In generate mode, you MUST use -f/--file for a single .qmd file or -d/--directory for a directory of .qmd files. 
Do NOT pass the file as a positional argument.
        """
    )
    parser.add_argument("--mode", choices=["generate", "review", "insert", "verify", "clean"], default="generate",
                       help="Mode of operation: generate (create new quizzes), review (edit existing quizzes), insert (add quizzes to markdown), verify (validate quiz files), clean (remove all quizzes)")
    parser.add_argument("file_path", nargs="?", help="File path for review/insert modes (quiz JSON or markdown .qmd file).")
    parser.add_argument("-f", "--file", help="Path to a markdown (.qmd/.md) file. REQUIRED for generate mode unless --directory is used.")
    parser.add_argument("-q", "--quiz-file", help="Path to quiz JSON file (for verify mode only). For verify mode, can also accept .qmd files to find corresponding quiz files.")
    parser.add_argument("-d", "--directory", help="Path to directory containing .qmd files (for generate/clean modes) or quiz JSON files (for verify mode). REQUIRED for generate/clean modes unless --file is used.")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use (generate mode only).")
    parser.add_argument("-o", "--output", default="quizzes.json", help="Path to output JSON file (generate mode only).")
    parser.add_argument("--backup", action="store_true", help="Create backup files before cleaning (clean mode only).")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be cleaned without making changes (clean mode only).")
    parser.add_argument("--examples", action="store_true", help="Show usage examples")
    args = parser.parse_args()

    if args.examples:
        show_usage_examples()
        return

    if args.mode == "generate":
        run_generate_mode(args)
    elif args.mode == "review":
        run_review_mode(args)
    elif args.mode == "insert":
        run_insert_mode(args)
    elif args.mode == "verify":
        run_verify_mode(args)
    elif args.mode == "clean":
        run_clean_mode(args)
    else:
        print(f"Unknown mode: {args.mode}")
        parser.print_help()

def run_generate_mode(args):
    """Generate new quizzes from a markdown file or directory of .qmd files"""
    if not args.file and not args.directory:
        print("Error: In generate mode, you must specify either:")
        print("  -f/--file <path>     for a single .qmd file")
        print("  -d/--directory <path> for a directory containing .qmd files")
        print("\nExamples:")
        print("  python quizzes.py --mode generate --file chapter1.qmd")
        print("  python quizzes.py --mode generate --directory ./chapters/")
        print("  python quizzes.py --file chapter1.qmd  # generate mode is default")
        return
    
    if args.file:
        # Enforce .qmd extension
        if not args.file.lower().endswith('.qmd'):
            print("Error: The input file must have a .qmd extension for generate mode.")
            print("Please provide a file with .qmd extension.")
            return
        print("=== Quiz Generation Mode (Single File) ===")
        generate_for_file(args.file, args)
    elif args.directory:
        print("=== Quiz Generation Mode (Directory) ===")
        generate_for_directory(args.directory, args)

def generate_for_file(qmd_file, args):
    with open(qmd_file, "r", encoding="utf-8") as f:
        content = f.read()

    sections = extract_sections_with_ids(content)
    if sections is None:
        exit(1)

    print(f"Found {len(sections)} sections to process:")
    for section in sections:
        print(f"  - {section['section_title']} ({section['section_id']})")

    client = OpenAI()
    results = []
    
    for i, section in enumerate(sections, 1):
        print(f"\nProcessing section {i}/{len(sections)}: {section['section_title']} ({section['section_id']})")
        
        prompt = build_user_prompt(section["section_title"], section["section_text"])
        quiz_response = call_openai(client, SYSTEM_PROMPT, prompt, model=args.model)
        
        # Create a structured result for this section
        section_result = {
            "section_id": section["section_id"],
            "section_title": section["section_title"],
            "quiz_data": quiz_response
        }
        
        # Add status information
        if quiz_response.get("quiz_needed", False):
            num_questions = len(quiz_response.get("questions", []))
            print(f"  ✓ Quiz generated with {num_questions} questions")
        else:
            print(f"  - No quiz needed: {quiz_response.get('rationale', 'No rationale provided')}")
        
        results.append(section_result)

    # Create the final output structure
    output_data = {
        "metadata": {
            "source_file": qmd_file,
            "total_sections": len(sections),
            "sections_with_quizzes": sum(1 for r in results if r["quiz_data"].get("quiz_needed", False)),
            "sections_without_quizzes": sum(1 for r in results if not r["quiz_data"].get("quiz_needed", False))
        },
        "sections": results
    }

    # Validate the final output structure
    try:
        validate(instance=output_data, schema=QUIZ_FILE_SCHEMA)
        print("✅ Generated quiz file structure is valid")
    except ValidationError as e:
        print(f"❌ Generated quiz file structure is invalid:")
        print(f"   - Path: {' -> '.join(str(p) for p in e.path)}")
        print(f"   - Error: {e.message}")
        print("   Attempting to fix structure...")
        output_data = fix_quiz_file_structure(output_data)

    # Use the specified output path or default to quizzes.json in the same directory as input
    out_path = args.output if os.path.isabs(args.output) else os.path.join(os.path.dirname(qmd_file), args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Wrote quizzes to {out_path}")
    print(f"  - Total sections processed: {len(sections)}")
    print(f"  - Sections with quizzes: {output_data['metadata']['sections_with_quizzes']}")
    print(f"  - Sections without quizzes: {output_data['metadata']['sections_without_quizzes']}")

    # Update the frontmatter of the qmd file
    update_qmd_frontmatter(qmd_file, os.path.basename(out_path))

def generate_for_directory(directory, args):
    # Find all .qmd files in the directory (non-recursive)
    qmd_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.qmd')]
    if not qmd_files:
        print(f"No .qmd files found in directory: {directory}")
        return
    print(f"Found {len(qmd_files)} .qmd files in directory: {directory}")
    for qmd_file in qmd_files:
        print(f"\n--- Generating for: {qmd_file} ---")
        generate_for_file(qmd_file, args)

def validate_final_quiz_structure(data):
    """Validate the final quiz file structure manually"""
    if not isinstance(data, dict):
        return False
    
    if 'metadata' not in data or 'sections' not in data:
        return False
    
    metadata = data['metadata']
    required_metadata = ['source_file', 'total_sections', 'sections_with_quizzes', 'sections_without_quizzes']
    if not all(field in metadata for field in required_metadata):
        return False
    
    if not isinstance(data['sections'], list):
        return False
    
    for section in data['sections']:
        if not isinstance(section, dict):
            return False
        
        required_section_fields = ['section_id', 'section_title', 'quiz_data']
        if not all(field in section for field in required_section_fields):
            return False
        
        quiz_data = section['quiz_data']
        if not isinstance(quiz_data, dict):
            return False
        
        if 'quiz_needed' not in quiz_data:
            return False
    
    return True

def fix_quiz_file_structure(data):
    """Attempt to fix common issues in quiz file structure"""
    print("🔧 Attempting to fix quiz file structure...")
    
    # Ensure metadata exists and has required fields
    if 'metadata' not in data:
        data['metadata'] = {}
    
    metadata = data['metadata']
    if 'source_file' not in metadata:
        metadata['source_file'] = 'unknown'
    if 'total_sections' not in metadata:
        metadata['total_sections'] = len(data.get('sections', []))
    if 'sections_with_quizzes' not in metadata:
        sections_with_quizzes = sum(1 for s in data.get('sections', []) if s.get('quiz_data', {}).get('quiz_needed', False))
        metadata['sections_with_quizzes'] = sections_with_quizzes
    if 'sections_without_quizzes' not in metadata:
        sections_without_quizzes = sum(1 for s in data.get('sections', []) if not s.get('quiz_data', {}).get('quiz_needed', False))
        metadata['sections_without_quizzes'] = sections_without_quizzes
    
    # Ensure sections exist
    if 'sections' not in data:
        data['sections'] = []
    
    # Fix individual sections
    for i, section in enumerate(data['sections']):
        if not isinstance(section, dict):
            data['sections'][i] = {
                'section_id': f'#sec-fixed-{i}',
                'section_title': f'Fixed Section {i}',
                'quiz_data': {'quiz_needed': False, 'rationale': 'Section was corrupted and fixed'}
            }
            continue
        
        # Ensure required section fields
        if 'section_id' not in section:
            section['section_id'] = f'#sec-missing-id-{i}'
        if 'section_title' not in section:
            section['section_title'] = f'Section {i}'
        if 'quiz_data' not in section:
            section['quiz_data'] = {'quiz_needed': False, 'rationale': 'Missing quiz data'}
        
        # Fix quiz_data structure
        quiz_data = section['quiz_data']
        if not isinstance(quiz_data, dict):
            section['quiz_data'] = {'quiz_needed': False, 'rationale': 'Invalid quiz data structure'}
            continue
        
        if 'quiz_needed' not in quiz_data:
            quiz_data['quiz_needed'] = False
            quiz_data['rationale'] = 'Missing quiz_needed field'
    
    print("✅ Quiz file structure has been fixed")
    return data

def validate_quiz_file(quiz_file_path):
    """Validate that a quiz file exists and has the correct structure"""
    if not os.path.exists(quiz_file_path):
        return False, f"Quiz file not found: {quiz_file_path}"
    
    try:
        with open(quiz_file_path, 'r', encoding='utf-8') as f:
            quiz_data = json.load(f)
        
        if not isinstance(quiz_data, dict):
            return False, "Invalid JSON structure: root must be an object"
        
        if 'sections' not in quiz_data:
            return False, "Missing 'sections' key in quiz data"
        
        if not isinstance(quiz_data['sections'], list):
            return False, "Sections must be a list"
        
        return True, "Valid quiz file"
        
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON in quiz file: {str(e)}"
    except Exception as e:
        return False, f"Error reading quiz file: {str(e)}"

def run_review_mode(args):
    """Review and edit existing quizzes using GUI"""
    print("=== Quiz Review Mode ===")
    
    # Get file path from positional argument
    if not args.file_path:
        print("Error: File path is required for review mode")
        print("Usage:")
        print("  python quizzes.py --mode review <file_path>  # Quiz JSON or markdown file")
        return
    
    file_path = args.file_path
    
    # Determine if it's a markdown file or quiz file
    if file_path.lower().endswith(('.qmd', '.md', '.markdown')):
        # It's a markdown file, try to find the quiz file from metadata
        print(f"🔍 Looking for quiz file referenced in: {file_path}")
        quiz_file_path = find_quiz_file_from_qmd(file_path)
        if not quiz_file_path:
            print(f"❌ No quiz file found for markdown file: {file_path}")
            print("   Make sure the markdown file has 'quiz: filename.json' in its frontmatter")
            return
        print(f"✅ Found quiz file: {quiz_file_path}")
    else:
        # Assume it's a quiz file path
        quiz_file_path = file_path
    
    # Validate the quiz file
    is_valid, message = validate_quiz_file(quiz_file_path)
    if not is_valid:
        print(f"Error: {message}")
        return
    
    run_gui(quiz_file_path)

def run_insert_mode(args):
    """Insert quizzes into markdown files"""
    print("=== Quiz Insert Mode ===")
    
    # Get file path from positional argument
    if not args.file_path:
        print("Error: File path is required for insert mode")
        print("Usage:")
        print("  python quizzes.py --mode insert <file_path>  # Quiz JSON or markdown file")
        return
    
    file_path = args.file_path
    
    # Determine if it's a markdown file or quiz file
    if file_path.lower().endswith(('.qmd', '.md', '.markdown')):
        # It's a markdown file, try to find the quiz file from metadata
        print(f"🔍 Looking for quiz file referenced in: {file_path}")
        quiz_file_path = find_quiz_file_from_qmd(file_path)
        if not quiz_file_path:
            print(f"❌ No quiz file found for markdown file: {file_path}")
            print("   Make sure the markdown file has 'quiz: filename.json' in its frontmatter")
            return
        print(f"✅ Found quiz file: {quiz_file_path}")
        qmd_file_path = file_path
    else:
        # Assume it's a quiz file path, try to find the markdown file
        print(f"🔍 Looking for markdown file referenced in: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                quiz_data = json.load(f)
            qmd_file_path = find_qmd_file_from_quiz(file_path, quiz_data)
            if not qmd_file_path:
                print(f"❌ No markdown file found for quiz file: {file_path}")
                print("   The quiz file should have a 'source_file' in its metadata")
                return
            print(f"✅ Found markdown file: {qmd_file_path}")
            quiz_file_path = file_path
        except Exception as e:
            print(f"❌ Error reading quiz file: {str(e)}")
            return
    
    # Quick verification using existing code (brief mode)
    print("🔍 Verifying files...")
    
    # Verify quiz file
    is_valid, message = validate_quiz_file(quiz_file_path)
    if not is_valid:
        print(f"❌ Quiz file validation failed: {message}")
        return
    print("✅ Quiz file is valid")
    
    # Verify markdown file
    qmd_sections = analyze_qmd_file(qmd_file_path)
    if not qmd_sections:
        print("❌ Markdown file validation failed")
        return
    print("✅ Markdown file is valid")
    
    # Now insert the quizzes
    insert_quizzes_into_markdown(qmd_file_path, quiz_file_path)

def check_existing_quizzes(qmd_file_path):
    """Check if there are existing quiz questions in the markdown file"""
    try:
        with open(qmd_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for quiz question callouts
        question_pattern = re.compile(
            r'::: \{\.' + re.escape(QUIZ_QUESTION_CALLOUT_CLASS) + r' #[^}]*\}',
            re.DOTALL
        )
        existing_questions = question_pattern.findall(content)
        
        return len(existing_questions) > 0, len(existing_questions)
        
    except Exception as e:
        print(f"⚠️  Warning: Could not check for existing quizzes: {str(e)}")
        return False, 0

def insert_quizzes_into_markdown(qmd_file_path, quiz_file_path):
    """Insert quizzes from JSON file into markdown file"""
    print(f"📝 Inserting quizzes from {quiz_file_path} into {qmd_file_path}")
    
    # Check for existing quizzes
    has_existing, count = check_existing_quizzes(qmd_file_path)
    if has_existing:
        print(f"⚠️  Found {count} existing quiz question(s) in the file")
        print("   This may cause duplicate quizzes or conflicts.")
        
        while True:
            response = input("   Do you want to clean existing quizzes first? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                print("🧹 Cleaning existing quizzes...")
                try:
                    with open(qmd_file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    cleaned_content, removed_count = clean_quiz_content(content)
                    with open(qmd_file_path, 'w', encoding='utf-8') as f:
                        f.write(cleaned_content)
                    print(f"✅ Removed {removed_count} existing quiz elements")
                    break
                except Exception as e:
                    print(f"❌ Error cleaning existing quizzes: {str(e)}")
                    return
            elif response in ['n', 'no']:
                print("   Proceeding with insertion (may create duplicates)...")
                break
            else:
                print("   Please enter 'y' or 'n'")
    
    try:
        # Load the quiz data
        with open(quiz_file_path, 'r', encoding='utf-8') as f:
            quiz_data = json.load(f)
        
        # Load the markdown file
        with open(qmd_file_path, 'r', encoding='utf-8') as f:
            qmd_content = f.read()
        
        # Extract sections from markdown
        qmd_sections = extract_sections_with_ids(qmd_content)
        if not qmd_sections:
            print("❌ No sections found in markdown file")
            return
        
        # Create a mapping of section IDs to quiz data
        quiz_sections = {section['section_id']: section for section in quiz_data.get('sections', [])}
        
        # Process each section and insert quizzes
        modified_content = qmd_content
        sections_modified = 0
        quiz_answers = []  # Collect answers for the end
        
        for qmd_section in qmd_sections:
            section_id = qmd_section['section_id']
            quiz_section = quiz_sections.get(section_id)
            
            if quiz_section and quiz_section.get('quiz_data', {}).get('quiz_needed', False):
                # Create quiz callout
                quiz_markdown = format_quiz_callout(quiz_section, section_id)
                modified_content = insert_quiz_into_section(modified_content, section_id, quiz_markdown)
                
                # Collect answer for the end
                answer_markdown = format_answer_callout(quiz_section, section_id)
                quiz_answers.append(answer_markdown)
                
                sections_modified += 1
                print(f"  ✓ Added quiz to section: {qmd_section['section_title']}")
        
        # Add quiz answers section at the end
        if quiz_answers:
            # Strip trailing whitespace from each answer block before joining
            # to prevent extra newlines between callouts.
            stripped_answers = [qa.strip() for qa in quiz_answers]
            answers_section = "\n\n## Quiz Answers\n\n" + "\n\n".join(stripped_answers)
            modified_content += answers_section
        
        # Write the modified content back to the file
        with open(qmd_file_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print(f"\n✅ Successfully inserted quizzes into {qmd_file_path}")
        print(f"   - Sections modified: {sections_modified}")
        print(f"   - Total sections in file: {len(qmd_sections)}")
        
    except Exception as e:
        print(f"❌ Error inserting quizzes: {str(e)}")

def format_quiz_callout(quiz_section, section_id):
    """Format quiz data as callout for insertion"""
    quiz_data = quiz_section.get('quiz_data', {})
    questions = quiz_data.get('questions', [])
    
    if not questions:
        return ""
    
    # Extract section key (remove # prefix only, keep hyphens and underscores)
    section_key = section_id.replace('#', '')
    
    # Format the quiz as callout
    quiz_markdown = f"\n::: {{.{QUIZ_QUESTION_CALLOUT_CLASS} #quiz-question-{section_key}}}\n\n"
    
    for i, question in enumerate(questions, 1):
        quiz_markdown += f"{i}. {question['question']}\n\n"
    
    quiz_markdown += f"See Answer \\ref{{quiz-answer-{section_key}}}.\n\n:::\n\n"
    
    return quiz_markdown

def format_answer_callout(quiz_section, section_id):
    """Format answer data as callout for the answers section"""
    quiz_data = quiz_section.get('quiz_data', {})
    questions = quiz_data.get('questions', [])
    
    if not questions:
        return ""
    
    # Extract section key (remove # prefix only, keep hyphens and underscores)
    section_key = section_id.replace('#', '')
    
    # Format the answer as callout
    answer_markdown = f"::: {{.{QUIZ_ANSWER_CALLOUT_CLASS} #quiz-answer-{section_key}}}\n\n"
    
    for i, question in enumerate(questions, 1):
        q_type = question.get('question_type', 'SHORT') # Default to short answer for safety
        
        if q_type == "MCQ":
            # Format MCQ with bold stem and indented choices
            answer_markdown += f"{i}. **{question['question']}**\n\n"
            
            choices = question.get('choices', [])
            for j, choice in enumerate(choices):
                letter = chr(ord('A') + j)
                answer_markdown += f"    {letter}) {choice}\n"
            
            answer_text = question['answer']
            
            # Check if answer already indicates the choice
            if not re.search(r"^[Tt]he correct answer is [A-Z]\.", answer_text):
                correct_choice_letter = None
                for j, choice in enumerate(choices):
                    # Check if the answer text starts with the choice text, ignoring case and leading/trailing whitespace
                    if answer_text.strip().lower().startswith(choice.strip().lower()):
                        correct_choice_letter = chr(ord('A') + j)
                        break
                
                if correct_choice_letter:
                    answer_text = f"The correct answer is {correct_choice_letter}. {answer_text}"
            
            answer_markdown += f"\n    {answer_text}\n\n"
        else:
            # Standard formatting for other types
            answer_text = question['answer']
            if q_type == "FILL":
                # The answer is expected to be the filled word/phrase, optionally followed by a period and an explanation.
                parts = answer_text.split('.', 1)
                if len(parts) > 1:
                    filled_word = parts[0]
                    explanation = parts[1]
                    # Quote the filled word and reconstruct the answer.
                    answer_text = f'"{filled_word.strip()}".{explanation}'
                else:
                    # If no period, quote the entire answer.
                    answer_text = f'"{answer_text.strip()}"'
            elif q_type == "ORDER":
                answer_text = f"The correct order is: {answer_text}"
            
            answer_markdown += f"{i}. **{question['question']}**\n\n   {answer_text}\n\n"
    
    answer_markdown += ":::"
    
    return answer_markdown

def run_verify_mode(args):
    """Verify quiz files and validate their structure"""
    print("=== Quiz Verify Mode ===")
    
    if args.directory:
        run_verify_directory(args.directory)
    elif args.quiz_file:
        # Check if the quiz file is actually a markdown file
        if args.quiz_file.lower().endswith(('.qmd', '.md', '.markdown')):
            print(f"🔍 Detected markdown file provided with -q/--quiz-file: {args.quiz_file}")
            print("   Treating as markdown file and finding corresponding quiz file...")
            run_verify_qmd_to_quiz(args.quiz_file)
        else:
            # Quiz file provided - find corresponding markdown file and verify
            run_verify_quiz_to_qmd(args.quiz_file)
    elif args.file:
        # Markdown file provided - find corresponding quiz file and verify
        run_verify_qmd_to_quiz(args.file)
    else:
        print("Error: One of --quiz-file, --directory, or --file is required for verify mode")
        print("Usage:")
        print("  python quizzes.py --mode verify --quiz-file <quiz_file>")
        print("  python quizzes.py --mode verify --quiz-file <qmd_file>  # Will detect markdown file")
        print("  python quizzes.py --mode verify --directory <path>")
        print("  python quizzes.py --mode verify --file <qmd_file>")
        return

def run_verify_quiz_to_qmd(quiz_file_path):
    """Verify quiz file and find corresponding markdown file for validation"""
    print(f"🔍 Verifying quiz file and finding corresponding markdown: {quiz_file_path}")
    
    # Stage 1: Validate quiz file schema
    print("\n=== Stage 1: Quiz File Schema Validation ===")
    quiz_data = validate_quiz_schema(quiz_file_path)
    if not quiz_data:
        return
    
    # Stage 2: Find corresponding markdown file
    print("\n=== Stage 2: Finding Corresponding Markdown File ===")
    qmd_file_path = find_qmd_file_from_quiz(quiz_file_path, quiz_data)
    if not qmd_file_path:
        print("❌ No corresponding markdown file found")
        print("   The quiz file should have a 'source_file' in its metadata")
        return
    
    # Stage 3: Analyze markdown file
    print("\n=== Stage 3: Markdown File Analysis ===")
    qmd_sections = analyze_qmd_file(qmd_file_path)
    if not qmd_sections:
        return
    
    # Stage 4: Correspondence validation
    print("\n=== Stage 4: Correspondence Validation ===")
    validate_correspondence(qmd_sections, quiz_data, qmd_file_path, quiz_file_path)

def run_verify_qmd_to_quiz(qmd_file_path):
    """Verify markdown file and find corresponding quiz file for validation"""
    print(f"🔍 Verifying markdown file and finding corresponding quiz: {qmd_file_path}")
    
    # Stage 1: Analyze markdown file
    print("\n=== Stage 1: Markdown File Analysis ===")
    qmd_sections = analyze_qmd_file(qmd_file_path)
    if not qmd_sections:
        return
    
    # Stage 2: Find corresponding quiz file
    print("\n=== Stage 2: Finding Corresponding Quiz File ===")
    quiz_file_path = find_quiz_file_from_qmd(qmd_file_path)
    if not quiz_file_path:
        print("❌ No corresponding quiz file found")
        print("   Make sure the markdown file has 'quiz: filename.json' in its frontmatter")
        return
    
    # Stage 3: Validate quiz file schema
    print("\n=== Stage 3: Quiz File Schema Validation ===")
    quiz_data = validate_quiz_schema(quiz_file_path)
    if not quiz_data:
        return
    
    # Stage 4: Correspondence validation
    print("\n=== Stage 4: Correspondence Validation ===")
    validate_correspondence(qmd_sections, quiz_data, qmd_file_path, quiz_file_path)

def find_qmd_file_from_quiz(quiz_file_path, quiz_data):
    """Find the corresponding markdown file for a quiz file"""
    # Get source file from quiz metadata
    metadata = quiz_data.get('metadata', {})
    source_file = metadata.get('source_file')
    
    if not source_file:
        print("❌ Quiz file missing 'source_file' in metadata")
        return None
    
    # Try to find the source file
    if os.path.exists(source_file):
        # Absolute path or relative to current directory
        return source_file
    
    # Try relative to quiz file directory
    quiz_dir = os.path.dirname(quiz_file_path)
    relative_path = os.path.join(quiz_dir, source_file)
    if os.path.exists(relative_path):
        return relative_path
    
    # Try with different extensions
    for ext in ['.qmd', '.md', '.markdown']:
        if not source_file.endswith(ext):
            test_path = os.path.join(quiz_dir, source_file + ext)
            if os.path.exists(test_path):
                return test_path
    
    print(f"❌ Source file not found: {source_file}")
    print(f"   Tried:")
    print(f"   - {source_file}")
    print(f"   - {relative_path}")
    return None

def validate_quiz_schema(quiz_file_path):
    """Validate quiz file against schema and return parsed data"""
    print(f"🔍 Validating quiz schema: {quiz_file_path}")
    
    # Basic file validation
    is_valid, message = validate_quiz_file(quiz_file_path)
    if not is_valid:
        print(f"❌ Quiz file validation failed: {message}")
        return None
    
    try:
        with open(quiz_file_path, 'r', encoding='utf-8') as f:
            quiz_data = json.load(f)
        
        # Use JSON Schema validation
        try:
            validate(instance=quiz_data, schema=QUIZ_FILE_SCHEMA)
            print("✅ JSON Schema validation passed")
        except ValidationError as e:
            print(f"❌ JSON Schema validation failed:")
            print(f"   - Path: {' -> '.join(str(p) for p in e.path)}")
            print(f"   - Error: {e.message}")
            return None
        
        # Count statistics for reporting
        sections = quiz_data.get('sections', [])
        valid_sections = len(sections)
        total_questions = 0
        valid_questions = 0
        
        for section in sections:
            quiz_data_section = section.get('quiz_data', {})
            if quiz_data_section.get('quiz_needed', False):
                questions = quiz_data_section.get('questions', [])
                total_questions += len(questions)
                valid_questions += len(questions)  # All questions should be valid if schema passed
        
        print(f"✅ Schema validation passed:")
        print(f"   - Valid sections: {valid_sections}")
        print(f"   - Total questions: {total_questions}")
        print(f"   - Valid questions: {valid_questions}")
        
        return quiz_data
        
    except Exception as e:
        print(f"❌ Error validating quiz schema: {str(e)}")
        return None

def manual_schema_validation(quiz_data):
    """Manual validation fallback when jsonschema is not available"""
    # Validate schema structure
    if not isinstance(quiz_data, dict):
        print("❌ Quiz data must be an object")
        return False
    
    if 'sections' not in quiz_data:
        print("❌ Quiz data missing 'sections' key")
        return False
    
    if not isinstance(quiz_data['sections'], list):
        print("❌ Quiz 'sections' must be an array")
        return False
    
    # Validate each section's quiz data
    for i, section in enumerate(quiz_data['sections']):
        if not isinstance(section, dict):
            print(f"❌ Section {i+1}: Must be an object")
            return False
        
        required_fields = ['section_id', 'section_title', 'quiz_data']
        missing_fields = [field for field in required_fields if field not in section]
        if missing_fields:
            print(f"❌ Section {i+1}: Missing fields: {missing_fields}")
            return False
        
        quiz_data_section = section['quiz_data']
        if not isinstance(quiz_data_section, dict):
            print(f"❌ Section {i+1}: quiz_data must be an object")
            return False
        
        if 'quiz_needed' not in quiz_data_section:
            print(f"❌ Section {i+1}: Missing quiz_needed field")
            return False
        
        if quiz_data_section['quiz_needed']:
            if 'questions' not in quiz_data_section:
                print(f"❌ Section {i+1}: Missing questions array")
                return False
            
            questions = quiz_data_section['questions']
            if not isinstance(questions, list):
                print(f"❌ Section {i+1}: questions must be an array")
                return False
            
            # Validate each question
            for j, question in enumerate(questions):
                if not isinstance(question, dict):
                    print(f"❌ Section {i+1}, Question {j+1}: Must be an object")
                    return False
                
                required_question_fields = ['question', 'answer', 'learning_objective']
                missing_question_fields = [field for field in required_question_fields if field not in question]
                if missing_question_fields:
                    print(f"❌ Section {i+1}, Question {j+1}: Missing fields: {missing_question_fields}")
                    return False
    
    return True

def validate_correspondence(qmd_data, quiz_data, qmd_file_path, quiz_file_path):
    """Validate correspondence between QMD sections and quiz sections"""
    print(f"🔍 Validating correspondence between QMD and quiz files")
    
    qmd_sections = qmd_data['sections']
    quiz_sections = quiz_data['sections']
    
    # Create lookup dictionaries
    qmd_section_ids = {section['section_id']: section for section in qmd_sections}
    quiz_section_ids = {section['section_id']: section for section in quiz_sections}
    
    print(f"\n📊 Section Analysis:")
    print(f"   - QMD sections: {len(qmd_sections)}")
    print(f"   - Quiz sections: {len(quiz_sections)}")
    
    # Check for quiz sections that don't exist in QMD (this is the critical check)
    missing_in_qmd = []
    for quiz_section in quiz_sections:
        if quiz_section['section_id'] not in qmd_section_ids:
            missing_in_qmd.append({
                'section_id': quiz_section['section_id'],
                'title': quiz_section['section_title']
            })
    
    # Check for mismatched titles (when section exists in both but titles differ)
    mismatched_titles = []
    for quiz_section in quiz_sections:
        qmd_section = qmd_section_ids.get(quiz_section['section_id'])
        if qmd_section and qmd_section['section_title'] != quiz_section['section_title']:
            mismatched_titles.append({
                'section_id': quiz_section['section_id'],
                'qmd_title': qmd_section['section_title'],
                'quiz_title': quiz_section['section_title']
            })
    
    # Optional: Show which QMD sections don't have quizzes (for information only)
    qmd_sections_without_quizzes = []
    for qmd_section in qmd_sections:
        if qmd_section['section_id'] not in quiz_section_ids:
            qmd_sections_without_quizzes.append(qmd_section['section_title'])
    
    # Report results
    print(f"\n📋 Correspondence Results:")
    
    if missing_in_qmd:
        print(f"❌ Quiz sections that don't exist in QMD file:")
        for missing in missing_in_qmd:
            print(f"   - {missing['title']} ({missing['section_id']})")
    else:
        print(f"✅ All quiz sections exist in QMD file")
    
    if mismatched_titles:
        print(f"❌ Mismatched section titles:")
        for mismatch in mismatched_titles:
            print(f"   - {mismatch['section_id']}:")
            print(f"     QMD:  {mismatch['qmd_title']}")
            print(f"     Quiz: {mismatch['quiz_title']}")
    else:
        print(f"✅ All section titles match")
    
    # Informational: Show QMD sections without quizzes
    if qmd_sections_without_quizzes:
        print(f"\nℹ️  QMD sections without quizzes (this is normal):")
        for title in qmd_sections_without_quizzes:
            print(f"   - {title}")
    else:
        print(f"\nℹ️  All QMD sections have corresponding quiz entries")
    
    # Summary - only count critical issues
    critical_issues = len(missing_in_qmd) + len(mismatched_titles)
    
    print(f"\n{'='*60}")
    if critical_issues == 0:
        print(f"🎉 Perfect correspondence! All quiz sections exist in QMD file.")
    else:
        print(f"⚠️  Found {critical_issues} critical correspondence issue(s)")
        if missing_in_qmd:
            print(f"   - {len(missing_in_qmd)} quiz sections missing from QMD")
        if mismatched_titles:
            print(f"   - {len(mismatched_titles)} mismatched section titles")
    print(f"{'='*60}")

def run_verify_directory(directory_path):
    """Verify all quiz files in a directory"""
    if not os.path.exists(directory_path):
        print(f"❌ Directory not found: {directory_path}")
        return
    
    if not os.path.isdir(directory_path):
        print(f"❌ Path is not a directory: {directory_path}")
        return
    
    print(f"🔍 Scanning directory: {directory_path}")
    
    # Find all JSON files
    json_files = []
    for file in os.listdir(directory_path):
        if file.endswith('.json'):
            json_files.append(os.path.join(directory_path, file))
    
    if not json_files:
        print("❌ No JSON files found in directory")
        return
    
    print(f"📁 Found {len(json_files)} JSON files")
    
    # Verify each file
    results = []
    for json_file in json_files:
        print(f"\n{'='*60}")
        print(f"Verifying: {os.path.basename(json_file)}")
        print(f"{'='*60}")
        
        result = verify_single_file_detailed(json_file)
        results.append(result)
    
    # Print summary
    print_summary(results)

def run_verify_single_file(quiz_file_path):
    """Verify a single quiz file"""
    print(f"🔍 Verifying single file: {quiz_file_path}")
    result = verify_single_file_detailed(quiz_file_path)
    
    # For single file, show detailed output
    if result['valid']:
        print(f"\n🎉 Quiz file is valid!")
    else:
        print(f"\n❌ Quiz file has issues!")

def verify_single_file_detailed(quiz_file_path):
    """Verify a single quiz file and return detailed results"""
    result = {
        'file': quiz_file_path,
        'valid': False,
        'error': None,
        'metadata': {},
        'sections': {
            'total': 0,
            'valid': 0,
            'with_quizzes': 0,
            'without_quizzes': 0
        },
        'questions': {
            'total': 0,
            'valid': 0
        }
    }
    
    # Basic file validation
    is_valid, message = validate_quiz_file(quiz_file_path)
    if not is_valid:
        result['error'] = message
        return result
    
    # Load and perform detailed validation
    try:
        with open(quiz_file_path, 'r', encoding='utf-8') as f:
            quiz_data = json.load(f)
        
        # Check metadata
        metadata = quiz_data.get('metadata', {})
        result['metadata'] = {
            'source_file': metadata.get('source_file', 'Not specified'),
            'total_sections': metadata.get('total_sections', 'Not specified'),
            'sections_with_quizzes': metadata.get('sections_with_quizzes', 'Not specified'),
            'sections_without_quizzes': metadata.get('sections_without_quizzes', 'Not specified')
        }
        
        # Validate sections
        sections = quiz_data.get('sections', [])
        result['sections']['total'] = len(sections)
        
        for section in sections:
            section_id = section.get('section_id', '')
            section_title = section.get('section_title', '')
            quiz_data_section = section.get('quiz_data', {})
            
            # Check required fields
            if not section_id or not section_title:
                continue
            
            # Check quiz data structure
            if not isinstance(quiz_data_section, dict):
                continue
            
            quiz_needed = quiz_data_section.get('quiz_needed', None)
            if quiz_needed is None:
                continue
            
            result['sections']['valid'] += 1
            
            if quiz_needed:
                result['sections']['with_quizzes'] += 1
                questions = quiz_data_section.get('questions', [])
                result['questions']['total'] += len(questions)
                
                # Validate questions
                for question in questions:
                    if isinstance(question, dict):
                        required_fields = ['question', 'answer', 'learning_objective']
                        if all(field in question for field in required_fields):
                            result['questions']['valid'] += 1
            else:
                result['sections']['without_quizzes'] += 1
        
        result['valid'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result

def print_summary(results):
    """Print a summary of verification results for multiple files"""
    print(f"\n{'='*80}")
    print(f"📊 VERIFICATION SUMMARY")
    print(f"{'='*80}")
    
    total_files = len(results)
    valid_files = sum(1 for r in results if r['valid'])
    invalid_files = total_files - valid_files
    
    print(f"📁 Total files processed: {total_files}")
    print(f"✅ Valid files: {valid_files}")
    print(f"❌ Invalid files: {invalid_files}")
    
    if invalid_files > 0:
        print(f"\n❌ Invalid files:")
        for result in results:
            if not result['valid']:
                print(f"   - {os.path.basename(result['file'])}: {result['error']}")
    
    # Summary statistics
    total_sections = sum(r['sections']['total'] for r in results if r['valid'])
    valid_sections = sum(r['sections']['valid'] for r in results if r['valid'])
    sections_with_quizzes = sum(r['sections']['with_quizzes'] for r in results if r['valid'])
    sections_without_quizzes = sum(r['sections']['without_quizzes'] for r in results if r['valid'])
    total_questions = sum(r['questions']['total'] for r in results if r['valid'])
    valid_questions = sum(r['questions']['valid'] for r in results if r['valid'])
    
    print(f"\n📝 Section Statistics:")
    print(f"   - Total sections: {total_sections}")
    print(f"   - Valid sections: {valid_sections}")
    print(f"   - Sections with quizzes: {sections_with_quizzes}")
    print(f"   - Sections without quizzes: {sections_without_quizzes}")
    
    print(f"\n❓ Question Statistics:")
    print(f"   - Total questions: {total_questions}")
    print(f"   - Valid questions: {valid_questions}")
    
    if total_questions > 0:
        question_validity = (valid_questions / total_questions) * 100
        print(f"   - Question validity rate: {question_validity:.1f}%")
    
    # File-by-file breakdown
    print(f"\n📋 File-by-file breakdown:")
    for result in results:
        filename = os.path.basename(result['file'])
        status = "✅" if result['valid'] else "❌"
        sections_info = f"{result['sections']['valid']}/{result['sections']['total']} sections"
        questions_info = f"{result['questions']['valid']}/{result['questions']['total']} questions"
        
        print(f"   {status} {filename}: {sections_info}, {questions_info}")
    
    print(f"\n{'='*80}")
    if invalid_files == 0:
        print(f"🎉 All files are valid!")
    else:
        print(f"⚠️  {invalid_files} file(s) have issues that need attention.")

def analyze_qmd_file(qmd_file_path):
    """Analyze QMD file and extract sections with metadata"""
    if not os.path.exists(qmd_file_path):
        print(f"❌ QMD file not found: {qmd_file_path}")
        return None
    
    try:
        with open(qmd_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract quiz metadata
        quiz_metadata = extract_quiz_metadata(content)
        if quiz_metadata:
            print(f"📋 Found quiz metadata: {quiz_metadata}")
        
        # Extract sections
        sections = extract_sections_with_ids(content)
        if not sections:
            print("❌ No sections found in QMD file")
            return None
        
        print(f"📝 Found {len(sections)} sections in QMD file:")
        for section in sections:
            print(f"   - {section['section_title']} ({section['section_id']})")
        
        return {
            'file_path': qmd_file_path,
            'quiz_metadata': quiz_metadata,
            'sections': sections
        }
        
    except Exception as e:
        print(f"❌ Error reading QMD file: {str(e)}")
        return None

def extract_quiz_metadata(content):
    """Extract quiz metadata from QMD file frontmatter"""
    # Look for YAML frontmatter
    frontmatter_match = re.search(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if not frontmatter_match:
        return None
    
    frontmatter = frontmatter_match.group(1)
    
    # Look for quiz: field
    quiz_match = re.search(r'^quiz:\s*(.+)$', frontmatter, re.MULTILINE)
    if quiz_match:
        return quiz_match.group(1).strip()
    
    return None

def find_quiz_file_from_qmd(qmd_file_path):
    """Find the corresponding quiz file for a QMD file"""
    # First, try to get quiz path from QMD metadata
    try:
        with open(qmd_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        quiz_metadata = extract_quiz_metadata(content)
        if quiz_metadata:
            # Try relative to QMD file directory
            qmd_dir = os.path.dirname(qmd_file_path)
            quiz_path = os.path.join(qmd_dir, quiz_metadata)
            if os.path.exists(quiz_path):
                return quiz_path
            
            # Try absolute path
            if os.path.exists(quiz_metadata):
                return quiz_metadata
    
    except Exception:
        pass
    
    # Fallback: look for quiz file with same name
    qmd_dir = os.path.dirname(qmd_file_path)
    qmd_name = os.path.splitext(os.path.basename(qmd_file_path))[0]
    quiz_path = os.path.join(qmd_dir, f"{qmd_name}_quizzes.json")
    
    if os.path.exists(quiz_path):
        return quiz_path
    
    return None

def insert_quiz_into_section(content, section_id, quiz_markdown):
    """Insert quiz markdown at the end of a specific section"""
    # Remove the # prefix if present
    if section_id.startswith('#'):
        section_id = section_id[1:]
    
    # Find the section in the content
    section_pattern = re.compile(rf'^##\s+.*?\{{\#{re.escape(section_id)}\}}.*?$', re.MULTILINE)
    match = section_pattern.search(content)
    
    if not match:
        print(f"⚠️  Warning: Could not find section {section_id} in content")
        return content
    
    # Find the start and end of this section
    start_pos = match.start()
    
    # Find the next section or end of file
    next_section_pattern = re.compile(r'^##\s+', re.MULTILINE)
    next_match = next_section_pattern.search(content, start_pos + 1)
    
    if next_match:
        end_pos = next_match.start()
    else:
        end_pos = len(content)
    
    # Insert the quiz before the next section
    before_section = content[:end_pos]
    after_section = content[end_pos:]
    
    return before_section + quiz_markdown + after_section

def run_clean_mode(args):
    """Clean all quizzes from markdown files"""
    print("=== Quiz Clean Mode ===")
    
    if not args.file and not args.directory:
        print("Error: In clean mode, you must specify either:")
        print("  -f/--file <path>     for a single .qmd file")
        print("  -d/--directory <path> for a directory containing .qmd files")
        print("\nExamples:")
        print("  python quizzes.py --mode clean --file chapter1.qmd")
        print("  python quizzes.py --mode clean --directory ./chapters/")
        return
    
    if args.file:
        # Enforce .qmd extension
        if not args.file.lower().endswith('.qmd'):
            print("Error: The input file must have a .qmd extension for clean mode.")
            print("Please provide a file with .qmd extension.")
            return
        print("=== Quiz Clean Mode (Single File) ===")
        clean_single_file(args.file, args)
    elif args.directory:
        print("=== Quiz Clean Mode (Directory) ===")
        clean_directory(args.directory, args)

def clean_single_file(qmd_file, args):
    """Clean quizzes from a single QMD file"""
    print(f"🧹 Cleaning quizzes from: {qmd_file}")
    
    try:
        # Read the file
        with open(qmd_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create backup if requested
        if args.backup:
            backup_file = f"{qmd_file}.backup"
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"📦 Created backup: {backup_file}")
        
        # Clean the content
        cleaned_content, removed_count = clean_quiz_content(content)
        
        if args.dry_run:
            print(f"🔍 Dry run - would remove {removed_count} quiz elements from {qmd_file}")
            return
        
        # Write the cleaned content back
        with open(qmd_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        print(f"✅ Cleaned {qmd_file}")
        print(f"   - Removed {removed_count} quiz elements")
        
    except Exception as e:
        print(f"❌ Error cleaning {qmd_file}: {str(e)}")

def clean_directory(directory, args):
    """Clean quizzes from all QMD files in a directory"""
    if not os.path.exists(directory):
        print(f"❌ Directory not found: {directory}")
        return
    
    if not os.path.isdir(directory):
        print(f"❌ Path is not a directory: {directory}")
        return
    
    print(f"🔍 Scanning directory: {directory}")
    
    # Find all .qmd files
    qmd_files = []
    for file in os.listdir(directory):
        if file.lower().endswith('.qmd'):
            qmd_files.append(os.path.join(directory, file))
    
    if not qmd_files:
        print("❌ No .qmd files found in directory")
        return
    
    print(f"📁 Found {len(qmd_files)} .qmd files")
    
    # Clean each file
    total_removed = 0
    for qmd_file in qmd_files:
        print(f"\n--- Cleaning: {os.path.basename(qmd_file)} ---")
        try:
            # Read the file
            with open(qmd_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create backup if requested
            if args.backup:
                backup_file = f"{qmd_file}.backup"
                with open(backup_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"📦 Created backup: {os.path.basename(backup_file)}")
            
            # Clean the content
            cleaned_content, removed_count = clean_quiz_content(content)
            total_removed += removed_count
            
            if args.dry_run:
                print(f"🔍 Dry run - would remove {removed_count} quiz elements")
                continue
            
            # Write the cleaned content back
            with open(qmd_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            
            print(f"✅ Cleaned {os.path.basename(qmd_file)} ({removed_count} elements removed)")
            
        except Exception as e:
            print(f"❌ Error cleaning {os.path.basename(qmd_file)}: {str(e)}")
    
    if args.dry_run:
        print(f"\n🔍 Dry run complete - would remove {total_removed} quiz elements total")
    else:
        print(f"\n✅ Clean complete - removed {total_removed} quiz elements total")

def clean_quiz_content(content):
    """Remove all quiz callouts and Quiz Answers section from content, preserving all line breaks and spaces around section headers and blocks."""
    removed_count = 0

    # Remove quiz question callouts (match any number of colons, preserve whitespace)
    question_pattern = re.compile(
        r'\n+:{3,}\s*\{\.' + re.escape(QUIZ_QUESTION_CALLOUT_CLASS) + r' #[^}]*\}\s*\n.*?\n:{3,}\s*\n',
        re.DOTALL
    )
    content, question_count = question_pattern.subn('\n', content)
    removed_count += question_count

    # Remove quiz answer callouts (match any number of colons, preserve whitespace)
    answer_pattern = re.compile(
        r'\n+:{3,}\s*\{\.' + re.escape(QUIZ_ANSWER_CALLOUT_CLASS) + r' #[^}]*\}\s*\n.*?\n:{3,}\s*\n',
        re.DOTALL
    )
    content, answer_count = answer_pattern.subn('\n', content)
    removed_count += answer_count

    # Remove the Quiz Answers section entirely, but preserve line breaks before/after
    answers_section_pattern = re.compile(
        r'(\n+## Quiz Answers[^\n]*\n(?:.|\n)*?)(?=\n## |\Z)',
        re.DOTALL
    )
    content, section_count = answers_section_pattern.subn('\n', content)
    removed_count += section_count

    return content, removed_count

def regenerate_section_quiz(client, section_title, section_text, current_quiz_data, user_prompt, model="gpt-4o"):
    """Regenerate quiz for a specific section based on user prompt"""
    
    # Format the current quiz data as JSON for context
    current_quiz_json = json.dumps(current_quiz_data, indent=2, ensure_ascii=False)
    
    # Build the regeneration prompt
    regeneration_prompt = f"""
You are being asked to REGENERATE the quiz questions for this section.

IMPORTANT: The user's regeneration request takes PRIORITY over the original quiz evaluation criteria. 
If the user wants a quiz generated, you should generate one regardless of whether the original analysis 
determined a quiz was needed or not. Focus on fulfilling the user's specific request while keeping 
the educational quality standards in mind.

User's regeneration request: {user_prompt}

{SYSTEM_PROMPT}

Current section: "{section_title}"

Current quiz data (in JSON format):
{current_quiz_json}

Please regenerate the quiz questions for this section. Return your response in exactly the same JSON format as shown above, but with new questions based on the regeneration instructions provided.

IMPORTANT: After the JSON response, add a brief comment (2-3 sentences) explaining what changes you made based on the regeneration request. Format it like this:

<!-- REGENERATION_COMMENT -->
Brief explanation of what was changed based on the user's request.
<!-- END_REGENERATION_COMMENT -->
""".strip()
    
    # Call OpenAI with the same parameters as original generation
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an educational content specialist with expertise in machine learning systems."},
                {"role": "user", "content": regeneration_prompt}
            ],
            temperature=0.4
        )
        content = response.choices[0].message.content
        
        # Extract the comment if present
        comment = ""
        comment_match = re.search(r'<!-- REGENERATION_COMMENT -->(.*?)<!-- END_REGENERATION_COMMENT -->', content, re.DOTALL)
        if comment_match:
            comment = comment_match.group(1).strip()
            # Remove the comment from content to get clean JSON
            content = re.sub(r'<!-- REGENERATION_COMMENT -->.*?<!-- END_REGENERATION_COMMENT -->', '', content, flags=re.DOTALL).strip()
        
        # Parse the JSON response
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                return {"quiz_needed": False, "rationale": "No JSON found in regeneration response"}
        
        # Validate the response against JSON_SCHEMA
        try:
            validate(instance=data, schema=JSON_SCHEMA)
        except ValidationError as e:
            print(f"⚠️  Warning: Regeneration response doesn't match schema: {e.message}")
            return {"quiz_needed": False, "rationale": f"Schema validation failed: {e.message}"}
        
        # Add the comment to the data for display purposes (won't be saved to JSON)
        data['_regeneration_comment'] = comment
        
        return data
        
    except APIError as e:
        return {"quiz_needed": False, "rationale": f"API error: {str(e)}"}
    except Exception as e:
        return {"quiz_needed": False, "rationale": f"Unexpected error: {str(e)}"}

if __name__ == "__main__":
    main()

