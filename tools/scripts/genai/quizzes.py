#!/usr/bin/env python3
"""
Quiz Generation and Management Tool for ML Systems Textbook

This script provides a comprehensive tool for generating, managing, and maintaining
quiz questions for a machine learning systems textbook. It supports multiple modes
of operation and integrates with Quarto markdown files.

OVERVIEW:
---------
The tool generates pedagogically sound quiz questions using AI, manages quiz files,
and provides a GUI for reviewing and editing generated questions. It's designed to
work with a 20-chapter textbook structure and provides progressive difficulty based
on chapter position.

FEATURES:
---------
- AI-powered quiz generation with chapter-aware difficulty progression
- Interactive GUI for reviewing and editing questions
- Robust quiz insertion with reverse-order processing (prevents line number conflicts)
- Quiz removal and cleanup from markdown files
- File validation and verification with comprehensive error checking
- Support for multiple question types (MCQ, TF, SHORT, FILL, ORDER, CALC)
- Automatic frontmatter management
- Backup and dry-run capabilities for safe operation

MODES OF OPERATION:
-------------------
1. generate: Create new quiz files from QMD files
2. review: Open GUI to review/edit existing quizzes
3. insert: Insert quiz callouts into markdown files
4. verify: Validate quiz file structure and correspondence
5. clean: Remove quiz content from markdown files

USAGE EXAMPLES:
---------------
# Generate quizzes for a single chapter
python quizzes.py --mode generate -f contents/core/introduction/introduction.qmd

# Generate quizzes for all chapters in a directory
python quizzes.py --mode generate -d contents/core/

# Review quizzes with GUI
python quizzes.py --mode review -f introduction_quizzes.json

# Clean quiz content from files
python quizzes.py --mode clean --backup -f introduction.qmd

# Verify quiz file structure
python quizzes.py --mode verify -f introduction_quizzes.json

CHAPTER PROGRESSION:
-------------------
The tool automatically detects chapter position and adjusts question difficulty:
- Chapters 1-5: Foundational concepts and basic understanding
- Chapters 6-10: Intermediate complexity with practical applications
- Chapters 11-15: Advanced topics requiring system-level reasoning
- Chapters 16-20: Specialized topics requiring integration across concepts

QUESTION TYPES:
--------------
- MCQ: Multiple choice questions with 3-5 options
- TF: True/False questions with justification
- SHORT: Short answer questions for deeper reflection
- FILL: Fill-in-the-blank for specific terminology
- ORDER: Sequencing questions for processes/workflows
- CALC: Mathematical calculation questions

FILE STRUCTURE:
--------------
Quiz files are stored as JSON with the following structure:
{
  "metadata": {
    "source_file": "path/to/source.qmd",
    "total_sections": 5,
    "sections_with_quizzes": 3,
    "sections_without_quizzes": 2
  },
  "sections": [
    {
      "section_id": "#sec-introduction",
      "section_title": "Introduction",
      "quiz_data": {
        "quiz_needed": true,
        "rationale": {...},
        "questions": [...]
      }
    }
  ]
}

DEPENDENCIES:
------------
- openai: For AI-powered quiz generation
- gradio: For the interactive GUI
- jsonschema: For JSON validation
- pyyaml: For YAML frontmatter processing
- argparse: For command-line interface

MAINTENANCE:
-----------
- Global variables at the top control patterns and constants
- Question type configuration is easily modifiable
- System prompts can be updated for different requirements
- Chapter mapping can be adjusted for different book structures

AUTHOR: [Your Name]
VERSION: 1.0
DATE: [Current Date]
"""

import argparse
import os
import re
import json
from pathlib import Path
from openai import OpenAI, APIError
from datetime import datetime
import yaml
import concurrent.futures
import threading
import time
import sys

# Gradio imports
try:
    import gradio as gr
except ImportError:
    gr = None

# JSON Schema validation
from jsonschema import validate, ValidationError

# Callout class names for quiz insertion
QUIZ_QUESTION_CALLOUT_CLASS = ".callout-quiz-question"
QUIZ_ANSWER_CALLOUT_CLASS = ".callout-quiz-answer"

# Additional constants for quiz insertion (adapted from existing code)
QUIZ_CALLOUT_CLASS = QUIZ_QUESTION_CALLOUT_CLASS  # Alias for compatibility
ANSWER_CALLOUT_CLASS = QUIZ_ANSWER_CALLOUT_CLASS  # Alias for compatibility
QUESTION_ID_PREFIX = "quiz-question-"
ANSWER_ID_PREFIX = "quiz-answer-"
REFERENCE_TEXT = "See Answer"

# Quiz section headers and patterns for easy maintenance
SELF_CHECK_ANSWERS_HEADER = "Self-Check Answers"
SELF_CHECK_ANSWERS_HEADER_LOWER = SELF_CHECK_ANSWERS_HEADER.lower()
SELF_CHECK_ANSWERS_SECTION_HEADER = f"## {SELF_CHECK_ANSWERS_HEADER}"
SELF_CHECK_ANSWERS_SECTION_PATTERN = rf"^##\s+{re.escape(SELF_CHECK_ANSWERS_HEADER)}[\s\S]*?(?=^##\s|\Z)"
QUIZ_ANSWERS_SECTION_HEADER = SELF_CHECK_ANSWERS_SECTION_HEADER
QUIZ_ANSWERS_SECTION_PATTERN = SELF_CHECK_ANSWERS_SECTION_PATTERN

# These will be constructed dynamically since they need re.escape
QUIZ_QUESTION_CALLOUT_PATTERN = None  # Constructed dynamically
QUIZ_ANSWER_CALLOUT_PATTERN = None    # Constructed dynamically

# Frontmatter patterns
FRONTMATTER_PATTERN = r'^(---\s*\n.*?\n---\s*\n)'
YAML_FRONTMATTER_PATTERN = r'^(---\s*\n.*?\n---\s*\n)'

# Section patterns
SECTION_PATTERN = r"^##\s+(.+?)(\s*\{[^}]*\})?\s*$"

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
You are a professor and an educational content specialist with deep expertise in machine learning systems. You are tasked with creating pedagogically sound self-check questions for the university-level introduction to machine learning systems textbook.

**CRITICAL JSON REQUIREMENTS:**
You MUST return a valid JSON object with EXACTLY these fields:
- "quiz_needed": boolean (true or false, NOT a string "true" or "false")
- "rationale": 
  - When quiz_needed is true: Must be an object with EXACTLY these fields:
    - "focus_areas": array of 2-3 strings
    - "question_strategy": string
    - "difficulty_progression": string  
    - "integration": string
    - "ranking_explanation": string
  - When quiz_needed is false: Must be a single string explaining why no quiz is needed
- "questions": array of question objects (ONLY present when quiz_needed is true)

**TARGET AUDIENCE:**
This textbook is designed for:
- Advanced undergraduate students (juniors/seniors) in Computer Science, Engineering, or related fields
- Graduate students (Masters/PhD) in Computer Science, Data Science, or related disciplines
- Students with varying backgrounds in machine learning - some may have taken ML theory courses, others may be learning ML concepts alongside systems concepts
- Students with basic programming experience (Python preferred) and understanding of algorithms and data structures

**ASSUMED PREREQUISITES:**
- Basic programming experience (Python preferred)
- Understanding of algorithms and data structures
- Mathematical fundamentals (algebra, basic calculus concepts)
- Basic computer architecture concepts
- **Note: Students may have varying levels of ML theory background - some may be learning ML concepts for the first time in this course**

Your task is to first evaluate whether a self-check would be pedagogically valuable for the given section, and if so, generate 1 to 5 self-check questions and answers. Decide the number of questions based on the section's length and complexity. Each chapter has about 10 sections. So be careful not to generate too many questions.

## ML Systems Focus

Machine learning systems encompasses the full lifecycle: data pipelines, model training infrastructure, deployment, monitoring, serving, scaling, reliability, and operational concerns. Focus on system-level reasoning rather than algorithmic theory. When ML concepts are introduced, explain them clearly without assuming deep prior knowledge.

## Self-Check Evaluation Criteria

First, evaluate if this section warrants a self-check by considering:
1. Does it contain concepts that students need to actively understand and apply?
2. Are there potential misconceptions that need to be addressed?
3. Does it present system design tradeoffs or operational implications?
4. Does it build on previous knowledge in ways that should be reinforced?

**CRITICAL: If the section does not introduce technical tradeoffs, system components, or operational implications, set quiz_needed: false and justify.**

**Sections that typically DO NOT need self-checks:**
- Pure introductions or context-setting sections
- Sections that primarily provide historical context or motivation
- Sections that are purely descriptive without actionable concepts
- Overview sections without technical depth
- Motivational or high-level conceptual sections (e.g., "AI Pervasiveness," "Looking Ahead")

**Sections that typically DO need self-checks:**
- Sections introducing new technical concepts or system components
- Sections presenting design decisions, tradeoffs, or operational considerations
- Sections addressing common pitfalls or misconceptions
- Sections requiring application of concepts to real scenarios
- Sections building on previous knowledge in critical ways

## Chapter-Level Variety and Coherence

When previous self-check context is provided for the chapter:
- **Analyze the existing questions** to understand what concepts and question types have been covered
- **Ensure conceptual variety** - avoid repeating the same learning objectives or approaches
- **Complement rather than duplicate** - if similar concepts appear, approach them from different angles
- **Maintain chapter coherence** - questions should build on each other while covering distinct aspects
- **Balance question types** - if previous sections used mostly MCQs, consider SHORT, FILL, or other types
- **Focus on different system aspects** - if previous questions focused on tradeoffs, focus on implementation, operational concerns, or real-world applications

## Question Guidelines

**Content Focus:**
- Prioritize system-level reasoning: tradeoffs in deployment environments, impact of data pipeline design on model accuracy, scaling infrastructure for inference workloads, etc.
- Include quantitative analysis when applicable: resource consumption calculations, performance trade-off analysis, scaling estimates, cost comparisons, latency budgets, throughput analysis
- Include at least one question about design tradeoffs or operational implications
- Address common misconceptions when applicable
- Connect to practical ML systems scenarios
- Check whether similar concepts have been addressed in earlier sections and avoid repetition unless extending or applying in novel ways

**Question Types (use a variety based on content):**
{QUESTION_GUIDELINES}

**When to use different question types:**
- **CALC questions for:** Memory or storage requirements, latency calculations for multi-tier architectures, power consumption estimates, cost analysis for deployment options, throughput calculations for data pipelines, scaling factor analysis
- **MCQ questions for:** Comparing system architectures, identifying appropriate design patterns, selecting deployment strategies
- **SHORT questions for:** Explaining system tradeoffs, justifying design decisions, analyzing failure scenarios
- **FILL questions for:** Specific technical terminology, protocol names, architectural components (only when precise recall is required)
- **TF questions for:** Challenging common misconceptions, testing understanding of system constraints
- **ORDER questions for:** System deployment workflows, data pipeline stages, model lifecycle phases

**Question Type Specifics:**
- **MCQ**: Provide 3-5 plausible distractors. The correct answer should not be obvious from the question stem alone. Do not embed options (e.g., A, B, C) in the question string; use the choices array instead. **CRITICAL: Distribute correct answers evenly across choices (A, B, C, D) to avoid guessing patterns. Do NOT favor any particular choice - ensure equal distribution. If you have 4 MCQ questions, aim for one correct answer for each letter (A, B, C, D). IMPORTANT: Randomize the position of correct answers. Use all four choices (A, B, C, D) across your MCQs. AVOID: Questions where the answer is obvious (e.g., "Which pillar focuses on training?" with "Training" as a choice).**
- **SHORT**: Should encourage synthesis or justification (e.g., "Explain why X matters in Y context").
- **FILL**: Should test key terms, but avoid placing the answer immediately before or after the blank. Use only when the term is central and non-obvious.
- **TF**: Must include justification that addresses common misconceptions and explains why the statement matters in ML systems context.
- **ORDER**: Focus on processes where sequence matters for system outcomes.
- **CALC**: Include real-world context and explain the practical significance of the result. Must include realistic parameters, show calculation steps clearly, and explain what the numerical result means for system design decisions.

## Language and Writing Guidelines

**Use precise technical language:**
- Avoid generic academic terms like "crucial," "essential," "vital," "various," "numerous"
- Replace vague phrases like "in the context of," "in terms of," "with regard to" with specific technical scenarios
- Eliminate academic filler like "it is important to note," "furthermore," "moreover"
- Use specific technical impacts instead of general importance statements
- Frame questions with concrete technical situations rather than abstract contexts

**Direct technical framing:**
- Start with specific deployment scenarios, system constraints, or technical requirements
- Use active voice and precise terminology
- Focus on measurable impacts and specific system behaviors
- Replace qualitative importance with quantitative or specific technical relationships

**Quality Standards:**
- For `MCQ` questions, the `answer` string MUST start with `The correct answer is [LETTER].` followed by an explanation of why this choice is correct. Do NOT repeat the answer text in the explanation. In MCQ explanations, do not rephrase the question or answer. Instead, explain why the correct answer is correct in light of other distractors.
- **Maintain a professional textbook tone** - Use clear, academically appropriate language suitable for university-level instruction
- **Avoid basic definitional questions** - Focus on application, analysis, and system-level implications rather than simple recall
- **Keep answers concise** - Aim for 50-100 words per answer explanation, not 75-150 words
- **Every answer must include a "why" — not just what is right, but why it matters in an ML system context**
- Use the first question to address the most foundational or essential system insight from the section
- If multiple questions are included, ensure they span distinct ideas (e.g., tradeoffs, lifecycle stages, deployment implications)
- Progress from basic understanding to application/analysis when multiple questions are used
- **At least one question per quiz should apply the concept to a real-world systems scenario, such as latency tradeoffs, retraining risks, or data pipeline bottlenecks**
- **CRITICAL: Make questions self-contained and holistic**
  - Avoid phrases like "in this section," "as discussed above," "from the text," or any other context-dependent references
  - Questions should be able to stand alone and provide complete learning value when viewed in isolation
  - Include necessary context within the question itself rather than assuming the reader has just read the section
  - Frame questions to be educational and complete learning experiences on their own
  - Use phrases like "When designing ML systems," "For machine learning applications," etc.

**CRITICAL ANTI-PATTERNS TO AVOID:**
- Questions where the answer is obvious from the question itself or is trivially inferable
- **BAD EXAMPLE: "Which pillar focuses on AI training?" with choices A) Data B) Training C) Deployment D) Operations - the answer "Training" is obvious from the question**
- MCQs where distractors are implausible or where the answer is telegraphed
- **MCQ answer bias - DO NOT default to B or any particular choice. Ensure equal distribution across A, B, C, D**
- Questions that test surface-level recall without deeper understanding
- Questions where multiple reasonable answers could be correct
- Questions that simply repeat information from the text without requiring analysis
- Questions that test trivial facts rather than conceptual understanding
- Questions where the explanation just restates what's already obvious from the question
- Questions that reference "this section," "as discussed above," "from the text," or other context-dependent phrases
- Questions that assume the reader has just read the section and can't stand alone as complete learning experiences
- Generic academic filler language and vague contextual phrases
- Imprecise qualifiers instead of specific technical terms

**Bloom's Taxonomy Mix:**
- Remember: Key terms and concepts
- Understand: Explain implications and relationships  
- Apply: Use concepts in new scenarios
- Analyze: Compare approaches and identify tradeoffs
- Evaluate: Justify design decisions
- Create: Propose solutions to system challenges

**Integration Guidelines:**
- When appropriate, build on concepts introduced in earlier sections to show how foundational ideas evolve into more complex system-level considerations
- Ensure questions collectively cover the section's main learning objectives
- Questions should reinforce different facets of system-level thinking

**Quality Check**
Before finalizing, ensure:
- Questions test different aspects of the content (avoid redundancy)
- At least one question addresses system-level implications
- Questions are appropriate for the textbook's target audience
- Answer explanations help reinforce learning, not just state correctness
- The response strictly follows the JSON schema provided above
- No questions fall into the anti-patterns listed above
- Questions are distinct, avoid overlap, and reinforce different facets of system-level thinking
- Language is precise and technical rather than generic or academic filler
- If previous self-check context was provided: Do these questions complement rather than duplicate previous questions? Do they use different question types and focus areas?

## Required JSON Schema

You MUST return a valid JSON object that strictly follows this schema:

```json
{json.dumps(JSON_SCHEMA, indent=2)}
```

## Output Format

If you determine a self-check is NOT needed, return the standard `quiz_needed: false` object.

If a self-check IS needed, follow the structure below. For "MCQ" questions, provide the question stem in the `question` field and the options in the `choices` array. For all other types, the `choices` field should be omitted.

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
            "answer": "The correct answer is [LETTER]. This is the explanation for why this choice is correct.",
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

"""

def update_qmd_frontmatter(qmd_file_path, quiz_file_name):
    """
    Add or update the 'quiz' key in a QMD file's YAML frontmatter.
    
    This function safely modifies the YAML frontmatter of a Quarto markdown file
    to include a reference to the corresponding quiz file. It handles cases where
    frontmatter doesn't exist and preserves the existing structure.
    
    Args:
        qmd_file_path (str): Path to the QMD file to modify
        quiz_file_name (str): Name of the quiz file to reference
        
    Returns:
        None
        
    Raises:
        Exception: If there's an error reading or writing the file
    """
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
                
                # Ensure there's a line break after the closing ---
                new_frontmatter = f"---\n{new_yaml_content.strip()}\n---\n\n"
                
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
    Extract all level-2 sections (##) with their content and section references.
    
    This function parses markdown content to find all level-2 headers that have
    section reference labels (e.g., {#sec-...}). It validates that all sections
    have proper IDs and returns structured data for each section.
    
    Args:
        markdown_text (str): The markdown content to parse
        
    Returns:
        list: List of dictionaries containing section data, or None if validation fails.
              Each dict has keys: 'section_id', 'section_title', 'section_text'
              
    Note:
        - Filters out "Quiz Answers" sections automatically
        - Requires regular sections to have reference labels for consistency
        - Excludes unnumbered sections and special sections that don't need IDs
        - Ignores ## lines that appear inside any Quarto blocks
        - Returns None if any regular section is missing a reference label
        
    Excluded sections (don't need IDs):
        - Sections with {.unnumbered} attribute
        - Sections with {.appendix} attribute  
        - Sections with {.backmatter} attribute
        - "Quiz Answers" sections
        - ## lines inside any Quarto blocks:
          - ::: blocks (callouts, content-visible, layout, figures, etc.)
          - :::: blocks (4-colon divs)
          - ``` code blocks
          - --- YAML frontmatter blocks
    """
    lines = markdown_text.split('\n')
    all_matches = []
    
    # Track context to ignore ## lines inside any Quarto blocks
    in_div_block = False      # ::: blocks
    in_div4_block = False     # :::: blocks  
    in_code_block = False     # ``` blocks
    in_yaml_block = False     # --- blocks
    
    div_depth = 0
    div4_depth = 0
    code_block_depth = 0
    yaml_depth = 0
    
    for line_num, line in enumerate(lines):
        stripped_line = line.strip()
        
        # Track ::: div block boundaries (callouts, content-visible, layout, etc.)
        if stripped_line.startswith(':::'):
            if not in_div_block:
                in_div_block = True
                div_depth = 1
            else:
                div_depth -= 1
                if div_depth == 0:
                    in_div_block = False
        
        # Track :::: div block boundaries (4-colon divs)
        elif stripped_line.startswith('::::'):
            if not in_div4_block:
                in_div4_block = True
                div4_depth = 1
            else:
                div4_depth -= 1
                if div4_depth == 0:
                    in_div4_block = False
        
        # Track code block boundaries
        elif stripped_line.startswith('```'):
            if not in_code_block:
                in_code_block = True
                code_block_depth = 1
            else:
                code_block_depth -= 1
                if code_block_depth == 0:
                    in_code_block = False
        
        # Track YAML frontmatter boundaries
        elif stripped_line.startswith('---'):
            if not in_yaml_block:
                in_yaml_block = True
                yaml_depth = 1
            else:
                yaml_depth -= 1
                if yaml_depth == 0:
                    in_yaml_block = False
        
        # Only process ## lines if we're not inside any block
        if (stripped_line.startswith('##') and 
            not in_div_block and 
            not in_div4_block and
            not in_code_block and
            not in_yaml_block and
            not stripped_line.startswith('###')):  # Ignore level 3+ headers
            
            # Use regex to extract section info
            section_pattern = re.compile(r'^##\s+(.+?)(\s*\{[^}]*\})?\s*$')
            match = section_pattern.match(stripped_line)
            if match:
                all_matches.append((line_num, match))
    
    # Filter out "Quiz Answers" sections and special sections
    content_matches = []
    for line_num, match in all_matches:
        title = match.group(1).strip()
        attributes = match.group(2) if match.group(2) else ""
        
        # Skip Quiz Answers sections and Self-Check Answers sections
        if title.lower() in ['quiz answers', 'self-check answers']:
            continue
            
        # Skip unnumbered sections and other special sections
        if '.unnumbered' in attributes or '.unnumbered' in title:
            continue
            
        # Skip sections with other special attributes that don't need IDs
        if any(special in attributes for special in ['.unnumbered', '.appendix', '.backmatter']):
            continue
            
        content_matches.append((line_num, match))
    
    # Check which sections need IDs (regular sections without special attributes)
    missing_refs = []
    for line_num, match in content_matches:
        title = match.group(1).strip()
        attributes = match.group(2) if match.group(2) else ""
        
        # Check if this section has a proper ID (starts with #)
        has_id = re.search(r'\{#([\w\-]+)\}', attributes)
        if not has_id:
            missing_refs.append(title)
    
    if missing_refs:
        print("ERROR: The following sections are missing section reference labels (e.g., {#sec-...}):")
        for title in missing_refs:
            print(f"  - {title}")
        print("\nPlease add section references to all sections and re-run the script.")
        print("Note: Unnumbered sections (with {.unnumbered}) are automatically excluded.")
        return None
    
    # If all sections have IDs, proceed with extraction
    sections = []
    for i, (line_num, match) in enumerate(content_matches):
        title = match.group(1).strip()
        attributes = match.group(2) if match.group(2) else ""
        
        # Find the start and end of this section
        start_line = line_num + 1  # Start after the header line
        end_line = len(lines)
        
        # Find the next section boundary
        if i + 1 < len(content_matches):
            end_line = content_matches[i + 1][0]  # Next section starts here
        
        # Extract content between this section and the next
        section_content = '\n'.join(lines[start_line:end_line]).strip()
        
        # Extract the section ID
        id_match = re.search(r'\{#([\w\-]+)\}', attributes)
        ref = id_match.group(1) if id_match else None
        
        if ref:
            # Store the full section reference including the # symbol
            full_ref = f"#{ref}"
            sections.append({
                "section_id": full_ref,
                "section_title": title,
                "section_text": section_content
            })
    
    return sections

def call_openai(client, system_prompt, user_prompt, model="gpt-4o"):
    """
    Make an API call to OpenAI for quiz generation.
    
    This function handles the communication with OpenAI's API, including error
    handling, JSON parsing, and schema validation. It ensures the response
    follows the expected structure for quiz data.
    
    Args:
        client (OpenAI): Initialized OpenAI client instance
        system_prompt (str): The system prompt defining the AI's role and constraints
        user_prompt (str): The user prompt containing the section content
        model (str): OpenAI model to use (default: "gpt-4o")
        
    Returns:
        dict: Validated quiz response data, or fallback response on error
        
    Note:
        - Includes fallback JSON extraction if the response isn't pure JSON
        - Validates response against JSON_SCHEMA
        - Returns structured error responses for debugging
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.4,
            response_format={"type": "json_object"}
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
    """
    Manually validate individual quiz response structure.
    
    This function performs a thorough validation of quiz response data to ensure
    it meets all requirements before processing. It checks data types, required
    fields, and structural integrity.
    
    Args:
        data: The quiz response data to validate
        
    Returns:
        bool: True if the data is valid, False otherwise
        
    Note:
        This is a backup validation method in addition to JSON schema validation.
        It provides more detailed error checking for debugging purposes.
    """
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
        
        # Allow empty questions arrays (when user removes all questions)
        if len(data['questions']) > 5:
            return False
        
        # Validate each question (only if there are questions)
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

def build_user_prompt(section_title, section_text, chapter_number=None, chapter_title=None, previous_quizzes=None):
    """
    Build a user prompt with chapter context and previous quiz data for variety.
    
    This function constructs the user prompt sent to the AI, incorporating
    chapter-specific context, difficulty guidelines, and previous quiz data
    to ensure variety and avoid overlap within the chapter.
    
    Args:
        section_title (str): Title of the section being processed
        section_text (str): Content of the section
        chapter_number (int, optional): Chapter number (1-20)
        chapter_title (str, optional): Chapter title
        previous_quizzes (list, optional): List of previous quiz data from earlier sections in this chapter
        
    Returns:
        str: Formatted user prompt with chapter context, difficulty guidelines, and previous quiz context
        
    Note:
        - Chapters 1-5: Foundational concepts and basic understanding
        - Chapters 6-10: Intermediate complexity with practical applications
        - Chapters 11-15: Advanced topics requiring system-level reasoning
        - Chapters 16-20: Specialized topics requiring integration across concepts
        - Previous quiz data helps avoid redundancy and ensures variety
        - Uses global BOOK_OUTLINE to connect concepts across chapters and build progression
    """
    # Define chapter progression context
    chapter_context = ""
    difficulty_guidelines = ""
    book_progression_context = ""
    
    if chapter_number is not None:
        # Ensure book outline is built
        book_outline = build_book_outline_from_quarto_yml()
        
        if book_outline:
            chapter_context = f"""
**Chapter Context:**
This is Chapter {chapter_number}: {chapter_title or 'Unknown'} in a {len(book_outline)}-chapter textbook on Machine Learning Systems.
The book progresses from foundational concepts to advanced topics and operational concerns.
"""
            
            # Add book outline context for better progression understanding using dynamic BOOK_OUTLINE
            book_progression_context = f"""
**Book Progression Context:**
This chapter builds upon and connects to the broader textbook structure:

"""
            # Show previous chapters that this builds upon
            if chapter_number > 1:
                book_progression_context += "**Builds upon:**\n"
                for i in range(1, chapter_number):
                    if i <= len(book_outline):
                        book_progression_context += f"- Chapter {i}: {book_outline[i-1]}\n"
                book_progression_context += "\n"
            
            # Show upcoming chapters this connects to
            if chapter_number < len(book_outline):
                book_progression_context += "**Connects to:**\n"
                for i in range(chapter_number + 1, min(chapter_number + 4, len(book_outline) + 1)):
                    if i <= len(book_outline):
                        book_progression_context += f"- Chapter {i}: {book_outline[i-1]}\n"
                book_progression_context += "\n"
            
            book_progression_context += """
**Progression Guidelines:**
- Questions should acknowledge concepts from earlier chapters where relevant
- Build upon foundational knowledge established in previous chapters
- Prepare students for more advanced topics in upcoming chapters
- Create connections that show how ML systems concepts evolve and integrate
"""
        else:
            # No book outline available
            chapter_context = f"""
**Chapter Context:**
This is Chapter {chapter_number}: {chapter_title or 'Unknown'} in a Machine Learning Systems textbook.
"""
            book_progression_context = ""
        
        # Progressive difficulty guidelines based on chapter position
        if chapter_number <= 5:
            difficulty_guidelines = """
**Chapter Difficulty Guidelines (Chapters 1-5):**
- Focus on foundational concepts and basic understanding
- Emphasize core definitions and fundamental principles
- Questions should test comprehension of basic ML systems concepts
- Avoid overly complex scenarios or advanced technical details
- Establish building blocks for later chapters
"""
        elif chapter_number <= 10:
            difficulty_guidelines = """
**Chapter Difficulty Guidelines (Chapters 6-10):**
- Intermediate complexity with practical applications
- Questions should test understanding of technical implementation
- Include questions about tradeoffs and design decisions
- Connect concepts to real-world ML system scenarios
- Build upon foundational concepts from Chapters 1-5
"""
        elif chapter_number <= 15:
            difficulty_guidelines = """
**Chapter Difficulty Guidelines (Chapters 11-15):**
- Advanced topics requiring system-level reasoning
- Questions should test deep understanding of optimization and operational concerns
- Focus on integration of multiple concepts from earlier chapters
- Emphasize practical implications and real-world challenges
- Prepare for specialized topics in final chapters
"""
        else:
            difficulty_guidelines = """
**Chapter Difficulty Guidelines (Chapters 16-20):**
- Specialized topics requiring integration across multiple concepts
- Questions should test synthesis of knowledge from throughout the book
- Focus on ethical, societal, and advanced operational considerations
- Emphasize critical thinking about ML systems in broader contexts
- Integrate concepts from all previous chapters
"""
    
    # Build previous quiz context if available
    previous_quiz_context = ""
    if previous_quizzes and len(previous_quizzes) > 0:
        previous_quiz_context = f"""
**Previous Quiz Context (Avoid Overlap):**
The following sections in this chapter already have quizzes. Ensure your questions are distinct and avoid conceptual overlap:

"""
        for i, quiz_data in enumerate(previous_quizzes, 1):
            if quiz_data.get('quiz_needed', False):
                questions = quiz_data.get('questions', [])
                previous_quiz_context += f"\nSection {i} Questions:\n"
                for j, question in enumerate(questions, 1):
                    q_type = question.get('question_type', 'Unknown')
                    q_text = question.get('question', '')
                    previous_quiz_context += f"  Q{j} ({q_type}): {q_text}\n"
                previous_quiz_context += "\n"
        
        previous_quiz_context += """
**Variety Guidelines:**
- Use different question types than those already used in this chapter
- Focus on different aspects of the content (e.g., if previous questions focused on tradeoffs, focus on implementation or operational concerns)
- Ensure your questions complement rather than repeat the learning objectives covered in previous sections
- If similar concepts appear, approach them from a different angle or application context
- Consider how this section's concepts connect to and build upon earlier chapters
"""
    
    return f"""
This section is titled "{section_title}".

{chapter_context}
{book_progression_context}
{difficulty_guidelines}
{previous_quiz_context}

Section content:
{section_text}

Generate a self-check quiz with 1 to 5 well-structured questions and answers based on this section. Include a rationale explaining your question generation strategy and focus areas. Return your response in the specified JSON format.
""".strip()
def regenerate_section_quiz(client, section_title, section_text, current_quiz_data, user_prompt, chapter_number=None, chapter_title=None, previous_quizzes=None):
    """
    Regenerate quiz questions for a section with custom instructions.
    
    This function allows users to regenerate quiz questions with specific
    instructions while maintaining all quality standards and anti-patterns.
    It's used by the GUI for interactive quiz editing and can incorporate
    previous quiz context for better variety.
    
    Args:
        client (OpenAI): Initialized OpenAI client instance
        section_title (str): Title of the section
        section_text (str): Content of the section
        current_quiz_data (dict): Current quiz data for the section
        user_prompt (str): User's regeneration instructions
        chapter_number (int, optional): Chapter number (1-20)
        chapter_title (str, optional): Chapter title
        previous_quizzes (list, optional): List of previous quiz data from earlier sections in this chapter
        
    Returns:
        dict: New quiz data following the user's instructions
        
    Note:
        - Maintains all quality standards from the original prompt
        - Adds user instructions to the system prompt
        - Includes a regeneration comment for tracking changes
        - Can incorporate previous quiz context for variety if provided
        - Uses global BOOK_OUTLINE for chapter progression context
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
- If previous quiz context is provided, ensure the new questions complement rather than duplicate previous questions
- Provide a brief comment explaining how you addressed the user's instructions
"""
    
    # Build the user prompt with chapter context and previous quiz data
    user_prompt_text = build_user_prompt(section_title, section_text, chapter_number, chapter_title, previous_quizzes)
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
    """
    Gradio-based GUI for reviewing and editing quiz questions.
    
    This class provides an interactive web interface for reviewing generated quiz
    questions, regenerating questions with custom instructions, and managing
    question selection. It integrates with the quiz file format and provides
    a user-friendly way to interact with quiz data.
    
    Attributes:
        quiz_data (dict): The loaded quiz data structure
        sections (list): List of sections from the quiz file
        current_section_index (int): Index of the currently displayed section
        initial_file_path (str): Path to the initial quiz file
        original_qmd_content (str): Content of the original QMD file
        qmd_file_path (str): Path to the original QMD file
        question_states (dict): Track checked/unchecked state for each question
        
    Methods:
        load_quiz_file: Load and parse a quiz JSON file
        navigate_section: Move between sections
        save_changes: Save modifications to the quiz file
        regenerate_questions: Regenerate questions with custom instructions
    """
    
    def __init__(self, initial_file_path=None):
        """
        Initialize the QuizEditorGradio instance.
        
        Args:
            initial_file_path (str, optional): Path to the initial quiz file to load
        """
        self.quiz_data = None
        self.sections = []
        self.current_section_index = 0
        self.initial_file_path = initial_file_path
        self.original_qmd_content = None
        self.qmd_file_path = None
        self.question_states = {}  # Track checked/unchecked state for each question
        
    def load_quiz_file(self, file_path=None):
        """
        Load a quiz JSON file and initialize the editor state.
        
        This method loads a quiz file, validates its structure, and prepares
        the editor for interaction. It also attempts to load the corresponding
        QMD file for context.
        
        Args:
            file_path (str, optional): Path to the quiz file. If None, uses initial_file_path
            
        Returns:
            tuple: (section_title, nav_info, section_text, questions_text, status)
                   Status information for the loaded file
        """
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
        """
        Initialize checked state for all questions (all checked by default).
        
        This method sets up the question selection state for all sections,
        ensuring that all questions are initially marked as selected for
        inclusion in the final quiz.
        """
        self.question_states = {}
        for i, section in enumerate(self.sections):
            section_id = section['section_id']
            quiz_data = section.get('quiz_data', {})
            if quiz_data.get('quiz_needed', False):
                questions = quiz_data.get('questions', [])
                self.question_states[section_id] = [True] * len(questions)  # All checked by default
    
    def update_question_state(self, section_id, question_index, checked):
        """
        Update the checked state of a specific question.
        
        Args:
            section_id (str): The section identifier
            question_index (int): Index of the question within the section
            checked (bool): Whether the question should be included
        """
        if section_id not in self.question_states:
            self.question_states[section_id] = []
        
        # Ensure the list is long enough
        while len(self.question_states[section_id]) <= question_index:
            self.question_states[section_id].append(True)
        
        self.question_states[section_id][question_index] = checked
    
    def load_original_qmd_file(self, quiz_file_path):
        """
        Try to load the original .qmd file based on the quiz file path.
        
        This method attempts to find and load the original QMD file that
        corresponds to the quiz file. It first checks the metadata in the
        quiz file, then falls back to searching the directory.
        
        Args:
            quiz_file_path (str): Path to the quiz file
        """
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
        """
        Get the full section content from the original .qmd file.
        
        This method retrieves the complete section content from the original
        QMD file, including the header, for better context in the editor.
        
        Args:
            section (dict): Section data containing section_id and section_title
            
        Returns:
            str: Full section content including header, or fallback text
        """
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
        
        # Remove any section where quiz_needed is true but questions is empty
        modified_data['sections'] = [
            s for s in modified_data['sections']
            if not (s.get('quiz_data', {}).get('quiz_needed', False) and len(s.get('quiz_data', {}).get('questions', [])) == 0)
        ]
        
        # Save to the original file
        try:
            with open(self.initial_file_path, 'w', encoding='utf-8') as f:
                json.dump(modified_data, f, indent=2, ensure_ascii=False)
            return f"Saved changes to {os.path.basename(self.initial_file_path)}"
        except Exception as e:
            return f"Error saving file: {str(e)}"
    
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
        
        # Remove any section where quiz_needed is true but questions is empty
        modified_data['sections'] = [
            s for s in modified_data['sections']
            if not (s.get('quiz_data', {}).get('quiz_needed', False) and len(s.get('quiz_data', {}).get('questions', [])) == 0)
        ]
        
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

def format_quiz_information(section, quiz_data):
    """
    Format quiz information for display including rationale and metadata.
    
    This function creates a formatted display of quiz information for
    the GUI, including focus areas, question strategy, and learning
    objectives.
    
    Args:
        section (dict): Section data containing title and ID
        quiz_data (dict): Quiz data for the section
        
    Returns:
        str: Formatted quiz information text
    """
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
    """
    Format a question for display in the Gradio interface based on its type.
    
    This function takes a question object and formats it appropriately
    for display in the GUI, handling different question types (MCQ,
    TF, SHORT, etc.) with proper formatting.
    
    Args:
        question (dict): Question data containing type, text, and options
        question_number (int): The question number for display
        
    Returns:
        str: Formatted question text ready for display
    """
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
                    with gr.Column(scale=1):  # Checkbox without text
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
        
        # Status display for regeneration operations (after regeneration section)
        regenerate_status_display = gr.Textbox(label="Regeneration Status", interactive=False, visible=True, lines=3)
        
        # Quiz Information section (moved to bottom)
        gr.Markdown("### Quiz Information")
        quiz_info_display = gr.Markdown(quiz_info, visible=True)
        
        # Status display for overall/save operations (at bottom)
        overall_status_display = gr.Textbox(label="Save Status", interactive=False, visible=True, lines=2)
        
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
        save_btn.click(save_changes, inputs=question_checkboxes, outputs=[overall_status_display])
        
        # Regenerate button - updates regeneration status and refreshes the current section
        def regenerate_and_refresh(user_prompt):
            status = regenerate_questions(user_prompt)
            if status.startswith("✅"):
                section_data = get_section_data(editor.current_section_index)
                return [status, ""] + section_data
            else:
                return [status, user_prompt] + [""] * (1 + 2 + 1 + 1 + 5 + 5 + 5 + 5)
        
        regenerate_btn.click(
            regenerate_and_refresh, 
            inputs=[prompt_input], 
            outputs=[regenerate_status_display, prompt_input, section_title_box, nav_info_box, section_text_box, quiz_info_display] + question_checkboxes + question_markdowns + answer_markdowns + learning_obj_markdowns
        )
        
        interface.load(initial_load, outputs=[section_title_box, nav_info_box, section_text_box, quiz_info_display] + question_checkboxes + question_markdowns + answer_markdowns + learning_obj_markdowns)
        
    return interface

def run_gui(quiz_file_path=None):
    """
    Launch the Gradio GUI for quiz review and editing.
    
    This function creates and launches the interactive web interface
    for reviewing and editing quiz questions. It provides a user-friendly
    way to navigate through sections, view questions, and regenerate
    questions with custom instructions.
    
    Args:
        quiz_file_path (str, optional): Path to the quiz file to load initially
        
    Note:
        - Launches on localhost:7860 by default
        - Provides full quiz editing capabilities
        - Supports question regeneration with custom prompts
        - Allows saving changes back to the quiz file
    """
    if not quiz_file_path:
        print("Error: Quiz file path is required for GUI mode")
        print("Usage: python quizzes.py --mode review <file_path>")
        return
    
    print(f"Launching quiz review GUI for: {quiz_file_path}")
    interface = create_gradio_interface(quiz_file_path)
    interface.launch(share=False, server_name="0.0.0.0", server_port=7860)

def show_usage_examples():
    """
    Display comprehensive usage examples for all modes of operation.
    
    This function provides detailed examples of how to use the quiz tool
    in different scenarios, helping users understand the various modes
    and their applications.
    """
    print("\n=== Usage Examples ===")
    print("\n1. Generate quizzes from a markdown file:")
    print("   python quizzes.py --mode generate -f chapter1.qmd")
    print("   python quizzes.py --mode generate -f chapter1.qmd -o my_quizzes.json")
    print("   python quizzes.py --mode generate -d ./chapters/")
    print("   python quizzes.py --mode generate -d ./chapters/ --parallel")
    print("   python quizzes.py --mode generate -d ./chapters/ --parallel --max-workers 2")
    
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
    """
    Main entry point for the quiz generation and management tool.
    
    This function parses command line arguments and routes to the appropriate
    mode of operation. It validates input parameters and provides helpful
    error messages for incorrect usage.
    
    Modes:
        - generate: Create new quiz files from QMD files
        - review: Open GUI to review/edit existing quizzes
        - insert: Insert quiz callouts into markdown files
        - verify: Validate quiz file structure and correspondence
        - clean: Remove quiz content from markdown files
        
    Usage:
        python quizzes.py --mode <mode> -f <file> | -d <directory> [options]
        
    Note:
        - Requires either -f (file) or -d (directory) for all modes
        - Automatically detects file types (JSON vs QMD) and routes appropriately
        - Provides comprehensive help and usage examples
    """
    parser = argparse.ArgumentParser(
        description="Quiz generation and management tool for markdown files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode generate -f chapter1.qmd
  %(prog)s --mode generate -d ./some_dir/
  %(prog)s --mode generate -d ./some_dir/ --parallel
  %(prog)s --mode generate -d ./some_dir/ --parallel --max-workers 2
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
Use --parallel for faster directory processing (one thread per file).
        """
    )
    parser.add_argument("--mode", choices=["generate", "review", "insert", "verify", "clean"], 
                       required=False, help="Mode of operation")
    parser.add_argument("-f", "--file", help="Path to a file (.qmd, .md, or .json)")
    parser.add_argument("-d", "--directory", help="Path to directory")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use (generate mode only)")
    parser.add_argument("-o", "--output", default="quizzes.json", help="Path to output JSON file (generate mode only)")
    parser.add_argument("--backup", action="store_true", help="Create backup files before cleaning")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--parallel", action="store_true", help="Process multiple files in parallel (directory mode only)")
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum number of parallel workers (default: number of files, max 4)")
    parser.add_argument("--examples", action="store_true", help="Show usage examples")
    args = parser.parse_args()

    if args.examples:
        show_usage_examples()
        return

    if not args.mode:
        print("Error: --mode is required unless --examples is used.")
        parser.print_help()
        return

    # Validate that either -f or -d is provided
    if not args.file and not args.directory:
        print("Error: You must specify either -f (file) or -d (directory)")
        parser.print_help()
        return

    # Validate parallel processing options
    if args.parallel and args.file:
        print("Error: --parallel can only be used with directory mode (-d), not with single files (-f)")
        return
    
    if args.max_workers and not args.parallel:
        print("Warning: --max-workers is only effective when used with --parallel")

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
            run_insert_mode_directory(args.directory)
        elif args.mode == "verify":
            run_verify_mode_directory(args.directory)
        elif args.mode == "clean":
            run_clean_mode_directory(args.directory, args)

def run_generate_mode_simple(qmd_file, args):
    """
    Generate new quizzes from a single markdown file.
    
    Args:
        qmd_file (str): Path to the QMD file to process
        args (argparse.Namespace): Command line arguments
    """
    print(f"=== Quiz Generation Mode (Single File) ===")
    generate_for_file(qmd_file, args)

def run_generate_mode_directory(directory, args):
    """
    Generate new quizzes from all QMD files in a directory.
    
    Args:
        directory (str): Path to the directory containing QMD files
        args (argparse.Namespace): Command line arguments
    """
    if getattr(args, 'parallel', False):
        print(f"=== Quiz Generation Mode (Directory - Parallel) ===")
        generate_for_directory_parallel(directory, args)
    else:
        print(f"=== Quiz Generation Mode (Directory - Sequential) ===")
        generate_for_directory(directory, args)

def run_review_mode_simple(file_path):
    """
    Review and edit existing quizzes from a file (JSON or QMD).
    
    This function launches the interactive GUI for reviewing quiz questions.
    It can handle both JSON quiz files and QMD files (by finding the
    corresponding quiz file).
    
    Args:
        file_path (str): Path to the file to review (JSON or QMD)
    """
    print("=== Quiz Review Mode ===")
    
    # Determine file type
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext in ['.json']:
        # JSON file - run GUI directly
        run_gui(file_path)
    elif file_ext in ['.qmd', '.md']:
        # QMD file - find corresponding quiz file first
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            quiz_file = extract_quiz_metadata(content)
            if not quiz_file:
                print(f"❌ No 'quiz:' variable found in the frontmatter of {file_path}.")
                print("   Please add 'quiz: <quizfile>.json' to the YAML frontmatter.")
                return
            quiz_path = os.path.join(os.path.dirname(file_path), quiz_file)
            if not os.path.exists(quiz_path):
                print(f"❌ The quiz file '{quiz_file}' referenced in the frontmatter of {file_path} does not exist.")
                return
            run_gui(quiz_path)
        except Exception as e:
            print(f"❌ Error reading file: {str(e)}")
    else:
        print(f"❌ Unsupported file type: {file_ext}")
        print("   Supported types: .json, .qmd, .md")

def run_insert_mode_simple(file_path):
    """
    Insert quizzes into markdown files.
    
    This function handles the insertion of quiz callouts into QMD files.
    It can work with either JSON quiz files or QMD files (by finding
    the corresponding quiz file).
    
    Args:
        file_path (str): Path to the file (JSON or QMD)
        
    Note:
        This is a placeholder function - insert functionality is not yet implemented.
    """
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
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            quiz_file = extract_quiz_metadata(content)
            if not quiz_file:
                print(f"❌ No 'quiz:' variable found in the frontmatter of {file_path}.")
                print("   Please add 'quiz: <quizfile>.json' to the YAML frontmatter.")
                return
            quiz_path = os.path.join(os.path.dirname(file_path), quiz_file)
            if not os.path.exists(quiz_path):
                print(f"❌ The quiz file '{quiz_file}' referenced in the frontmatter of {file_path} does not exist.")
                return
            insert_quizzes_into_markdown(file_path, quiz_path)
        except Exception as e:
            print(f"❌ Error reading file: {str(e)}")
    else:
        print(f"❌ Unsupported file type: {file_ext}")
        print("   Supported types: .json, .qmd, .md")

def run_verify_mode_simple(file_path):
    """
    Verify quiz files and validate their structure.
    
    This function performs comprehensive validation of quiz files and
    their correspondence with QMD files. It checks file structure,
    metadata, and cross-references.
    
    Args:
        file_path (str): Path to the file to verify (JSON or QMD)
    """
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
                quiz_file = extract_quiz_metadata(content)
                if not quiz_file:
                    print(f"❌ No 'quiz:' variable found in the frontmatter of {file_path}.")
                    print("   Please add 'quiz: <quizfile>.json' to the YAML frontmatter.")
                    return
                quiz_path = os.path.join(os.path.dirname(file_path), quiz_file)
                if os.path.exists(quiz_path):
                    print(f"✅ Found corresponding quiz file: {quiz_path}")
                else:
                    print(f"⚠️  The quiz file '{quiz_file}' referenced in the frontmatter of {file_path} does not exist.")
            else:
                print("❌ No sections found in QMD file")
                
        except Exception as e:
            print(f"❌ Error reading file: {str(e)}")
        
    else:
        print(f"❌ Unsupported file type: {file_ext}")
        print("   Supported types: .json, .qmd, .md")

def run_clean_mode_simple(qmd_file, args):
    """
    Clean all quizzes from a single markdown file.
    
    Args:
        qmd_file (str): Path to the QMD file to clean
        args (argparse.Namespace): Command line arguments
    """
    print("=== Quiz Clean Mode (Single File) ===")
    clean_single_file(qmd_file, args)

def run_clean_mode_directory(directory, args):
    """
    Clean all quizzes from all QMD files in a directory.
    
    Args:
        directory (str): Path to the directory containing QMD files
        args (argparse.Namespace): Command line arguments
    """
    print("=== Quiz Clean Mode (Directory) ===")
    clean_directory(directory, args)

def run_verify_mode_directory(directory_path):
    """
    Verify all quiz files in a directory.
    
    This function would perform verification on all quiz files in a
    directory. Currently a placeholder for future implementation.
    
    Args:
        directory_path (str): Path to the directory to verify
        
    Note:
        This functionality is not yet implemented. It would involve:
        - Finding all JSON and QMD files in the directory
        - Running verification on each file
        - Providing a summary report
    """
    print("=== Quiz Verify Mode (Directory) ===")
    run_verify_directory(directory_path)

def extract_quiz_metadata(content):
    """Extract quiz file name from YAML frontmatter using proper YAML parsing"""
    frontmatter_pattern = re.compile(FRONTMATTER_PATTERN, re.DOTALL)
    match = frontmatter_pattern.match(content)
    if match:
        frontmatter_str = match.group(1)
        yaml_content_str = frontmatter_str.strip().strip('---').strip()
        
        try:
            frontmatter_data = yaml.safe_load(yaml_content_str)
            if isinstance(frontmatter_data, dict):
                return frontmatter_data.get('quiz')
        except yaml.YAMLError:
            # If YAML parsing fails, fall back to regex for backward compatibility
            quiz_match = re.search(r'^quiz:\s*(.+)$', frontmatter_str, re.MULTILINE)
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
    """
    Extract chapter number and title from file path.
    
    This function maps file paths to chapter numbers based on the book's
    structure as defined in _quarto.yml. It uses the global BOOK_OUTLINE 
    to identify which chapter a file belongs to for difficulty progression.
    
    Args:
        file_path (str): Path to the QMD file
        
    Returns:
        tuple: (chapter_number, chapter_title) or (None, None) if not found
        
    Note:
        The chapter mapping is based on the order specified in _quarto.yml.
        The function looks for the file in the book outline and returns its position.
    """
    # Get the dynamic book outline
    book_outline = build_book_outline_from_quarto_yml()
    
    # Extract the file path relative to the project root
    try:
        # Get absolute path and convert to relative path from project root
        abs_path = os.path.abspath(file_path)
        project_root = os.getcwd()
        if abs_path.startswith(project_root):
            relative_path = os.path.relpath(abs_path, project_root)
        else:
            relative_path = file_path
    except:
        relative_path = file_path
    
    # Look for this file in the book outline
    for i, qmd_file in enumerate(get_qmd_order_from_quarto_yml(QUARTO_YML_PATH)):
        if qmd_file == relative_path or qmd_file.endswith(os.path.basename(relative_path)):
            # Found the file in the book outline, return its position (1-indexed)
            chapter_number = i + 1
            chapter_title = book_outline[i] if i < len(book_outline) else "Unknown"
            return chapter_number, chapter_title
    
    return None, None

# ===== THREE-PHASE QUIZ GENERATION PIPELINE =====

def pre_process_section(section):
    """
    PHASE 1: PRE-PROCESSING
    Analyze section content and prepare metadata for quiz generation.
    Currently a placeholder for future enhancements.
    
    Args:
        section: Dictionary with section_title and section_text
        
    Returns:
        dict: Metadata about the section (currently empty)
    """
    # Placeholder for future pre-processing logic
    # Could analyze content type, complexity, key concepts, etc.
    return {}

def process_section(client, section, chapter_context, previous_quizzes, args):
    """
    PHASE 2: PROCESSING
    Main quiz generation phase - generates questions using OpenAI.
    
    Args:
        client: OpenAI client
        section: Dictionary with section info
        chapter_context: Chapter number and title
        previous_quizzes: List of previous quiz data
        args: Command line arguments
        
    Returns:
        dict: Quiz response from OpenAI
    """
    chapter_number, chapter_title = chapter_context
    
    # Build user prompt with chapter context and previous quiz data
    user_prompt = build_user_prompt(
        section['section_title'], 
        section['section_text'],
        chapter_number,
        chapter_title,
        previous_quizzes
    )
    
    # Call OpenAI
    response = call_openai(client, SYSTEM_PROMPT, user_prompt, args.model)
    return response

def post_process_section(response):
    """
    PHASE 3: POST-PROCESSING
    Redistribute MCQ answers to ensure balanced distribution across A, B, C, D.
    
    Args:
        response: Quiz response containing questions
        
    Returns:
        dict: Modified response with redistributed MCQ answers
    """
    import random
    
    if not response.get('quiz_needed', False):
        return response
    
    questions = response.get('questions', [])
    mcq_questions = [q for q in questions if q.get('question_type') == 'MCQ']
    
    if not mcq_questions:
        return response
    
    # Analyze current MCQ answer distribution
    current_distribution = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    for q in mcq_questions:
        answer_text = q.get('answer', '')
        # Extract current answer position (e.g., "The correct answer is B.")
        import re
        match = re.search(r'The correct answer is ([A-D])', answer_text)
        if match:
            current_distribution[match.group(1)] += 1
    
    print(f"     Current MCQ distribution: {current_distribution}")
    
    # Create a balanced distribution of answer positions
    positions = ['A', 'B', 'C', 'D']
    target_positions = []
    
    # For each MCQ, assign a position trying to balance the distribution
    for i in range(len(mcq_questions)):
        # Try to use each position equally
        if i < 4:
            # First 4 questions: use each position once
            available = [p for p in positions if p not in target_positions]
            if available:
                target_positions.append(random.choice(available))
            else:
                target_positions.append(random.choice(positions))
        else:
            # After first 4: choose randomly but prefer less-used positions
            counts = {p: target_positions.count(p) for p in positions}
            min_count = min(counts.values())
            least_used = [p for p, c in counts.items() if c == min_count]
            target_positions.append(random.choice(least_used))
    
    # Apply the new positions to MCQ questions
    mcq_index = 0
    for q in questions:
        if q.get('question_type') == 'MCQ':
            if mcq_index < len(target_positions):
                new_position = target_positions[mcq_index]
                
                # Find current correct answer position
                answer_text = q.get('answer', '')
                match = re.search(r'The correct answer is ([A-D])', answer_text)
                
                if match:
                    current_position = match.group(1)
                    
                    if current_position != new_position:
                        # Need to shuffle the choices
                        choices = q.get('choices', [])
                        if len(choices) == 4:
                            # Get indices
                            current_idx = ord(current_position) - ord('A')
                            new_idx = ord(new_position) - ord('A')
                            
                            # Swap the choices
                            if 0 <= current_idx < 4 and 0 <= new_idx < 4:
                                choices[current_idx], choices[new_idx] = choices[new_idx], choices[current_idx]
                                q['choices'] = choices
                                
                                # Update the answer text
                                q['answer'] = re.sub(
                                    r'The correct answer is [A-D]',
                                    f'The correct answer is {new_position}',
                                    answer_text
                                )
                
                mcq_index += 1
    
    # Report final distribution
    final_distribution = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    for i, pos in enumerate(target_positions):
        if i < len(mcq_questions):
            final_distribution[pos] += 1
    
    print(f"     Redistributed MCQ answers: {final_distribution}")
    
    return response

def generate_for_file(qmd_file, args):
    """
    Generate quizzes for a single QMD file.
    
    This function processes a single QMD file to generate quiz questions
    for each section. It extracts sections, determines chapter context,
    and calls the AI to generate appropriate questions. It also passes
    previous quiz data to ensure variety and avoid overlap within the chapter.
    
    Args:
        qmd_file (str): Path to the QMD file to process
        args (argparse.Namespace): Command line arguments including model and output
        
    Note:
        - Automatically detects chapter number and title for difficulty progression
        - Updates QMD frontmatter with quiz file reference
        - Creates structured JSON output with metadata
        - Passes previous quiz data to each section for variety and coherence
    """
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
        
        # Generate quizzes for each section, passing previous quiz data
        quiz_sections = []
        sections_with_quizzes = 0
        sections_without_quizzes = 0
        previous_quizzes = []  # Track previous quiz data for variety
        
        for i, section in enumerate(sections):
            print(f"\nProcessing section {i+1}/{len(sections)}: {section['section_title']}")
            print("-" * 40)
            
            # PHASE 1: PRE-PROCESSING
            print("  📊 PRE-PROCESSING: Analyzing section...")
            metadata = pre_process_section(section)
            
            # PHASE 2: PROCESSING
            print("  🤖 PROCESSING: Generating questions...")
            response = process_section(
                client, 
                section,
                (chapter_number, chapter_title),
                previous_quizzes,
                args
            )
            
            # PHASE 3: POST-PROCESSING
            if response.get('quiz_needed', False):
                print("  🔄 POST-PROCESSING: Redistributing MCQ answers...")
                response = post_process_section(response)
            
            if response.get('quiz_needed', False):
                sections_with_quizzes += 1
                questions = response.get('questions', [])
                print(f"  ✅ Generated quiz with {len(questions)} questions")
                
                # Show distribution of question types
                type_counts = {}
                for q in questions:
                    q_type = q.get('question_type', 'UNKNOWN')
                    type_counts[q_type] = type_counts.get(q_type, 0) + 1
                if type_counts:
                    type_str = ', '.join([f"{count} {qtype}" for qtype, count in type_counts.items()])
                    print(f"     Distribution: {type_str}")
                
                # Add to previous quizzes for next section
                previous_quizzes.append(response)
            else:
                sections_without_quizzes += 1
                print(f"  ⏭️  No quiz needed: {response.get('rationale', 'No rationale provided')}")
                # Still add to previous quizzes to maintain section count
                previous_quizzes.append(response)
            
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
        
        # Create output file path in the same directory as the QMD file
        qmd_dir = os.path.dirname(qmd_file)
        qmd_basename = os.path.splitext(os.path.basename(qmd_file))[0]
        
        # If output is the default "quizzes.json", use the QMD filename as prefix
        if args.output == "quizzes.json":
            output_file = os.path.join(qmd_dir, f"{qmd_basename}_quizzes.json")
        else:
            # If user specified a custom output name, use it in the QMD directory
            output_file = os.path.join(qmd_dir, args.output)
        
        # Save to output file
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

# Global variable for _quarto.yml path
QUARTO_YML_PATH = os.path.join(os.getcwd(), 'quarto', '_quarto.yml')

# Global book outline for Machine Learning Systems textbook
# This will be populated automatically from _quarto.yml and QMD files
BOOK_OUTLINE = None

# Thread-local storage for parallel processing
thread_local = threading.local()

class ProgressTracker:
    """
    Real-time progress tracker for parallel quiz generation.
    Shows progress bars for all files simultaneously.
    """
    
    def __init__(self, total_files):
        self.total_files = total_files
        self.progress_lock = threading.Lock()
        self.file_progress = {}  # file -> (current_section, total_sections, status)
        self.completed_files = 0
        self.start_time = time.time()
        
    def update_file_progress(self, file_name, current_section=None, total_sections=None, status="running"):
        """Update progress for a specific file"""
        with self.progress_lock:
            if file_name not in self.file_progress:
                self.file_progress[file_name] = (0, total_sections or 0, status)
            else:
                current, total, _ = self.file_progress[file_name]
                if current_section is not None:
                    current = current_section
                if total_sections is not None:
                    total = total_sections
                self.file_progress[file_name] = (current, total, status)
            self._redraw_progress()
    
    def complete_file(self, file_name, success=True):
        """Mark a file as completed"""
        with self.progress_lock:
            self.completed_files += 1
            status = "✅ completed" if success else "❌ failed"
            self.file_progress[file_name] = (self.file_progress.get(file_name, (0, 0, ""))[1], 
                                           self.file_progress.get(file_name, (0, 0, ""))[1], 
                                           status)
            self._redraw_progress()
    
    def _redraw_progress(self):
        """Redraw all progress bars"""
        # Clear screen (works on most terminals)
        print("\033[2J\033[H", end="", flush=True)
        
        # Print header
        elapsed = time.time() - self.start_time
        print(f"🚀 Parallel Quiz Generation - {self.completed_files}/{self.total_files} completed ({elapsed:.1f}s elapsed)")
        print("=" * 80)
        print()  # Add spacing
        
        # Only show active files (running, recently completed, or failed)
        active_files = []
        for file_name, (current, total, status) in self.file_progress.items():
            if status == "running" or "completed" in status or "failed" in status:
                active_files.append((file_name, current, total, status))
        
        # Sort active files by chapter number from book outline
        sorted_files = self._sort_active_files_by_chapter(active_files)
        
        for i, (file_name, current, total, status) in enumerate(sorted_files):
            # Add spacing between files
            if i > 0:
                print()
            
            if total > 0:
                # Show progress bar
                progress = current / total
                bar_length = 30
                filled_length = int(bar_length * progress)
                bar = "█" * filled_length + "░" * (bar_length - filled_length)
                percentage = int(progress * 100)
                
                if status == "running":
                    print(f"🔄 {file_name:<25} [{bar}] {percentage:3d}% ({current}/{total} sections)")
                elif "completed" in status:
                    print(f"✅ {file_name:<25} [{bar}] {percentage:3d}% {status}")
                elif "failed" in status:
                    print(f"❌ {file_name:<25} [{bar}] {percentage:3d}% {status}")
            else:
                # No sections found or not started
                if status == "running":
                    print(f"🔄 {file_name:<25} [{'░' * 30}] ---% (processing...)")
                elif "completed" in status:
                    print(f"✅ {file_name:<25} [{'█' * 30}] 100% {status}")
                elif "failed" in status:
                    print(f"❌ {file_name:<25} [{'░' * 30}] ---% {status}")
        
        print()  # Add spacing
        print("=" * 80)
        print(f"Total completed: {self.completed_files}/{self.total_files}")
        if self.completed_files < self.total_files:
            print("Press Ctrl+C to stop...")
    
    def _sort_active_files_by_chapter(self, active_files):
        """Sort active files by their chapter number from the book outline"""
        try:
            # Get the book outline to determine chapter order
            book_outline = build_book_outline_from_quarto_yml()
            ordered_files = get_qmd_order_from_quarto_yml(QUARTO_YML_PATH)
            
            # Create a mapping of file names to their position in the book
            file_order = {}
            for i, qmd_file in enumerate(ordered_files):
                file_name = Path(qmd_file).name
                file_order[file_name] = i
            
            # Sort the active files by their position in the book outline
            def sort_key(file_data):
                file_name = file_data[0]
                return file_order.get(file_name, 999)  # Put unknown files at the end
            
            return sorted(active_files, key=sort_key)
            
        except Exception:
            # Fallback to alphabetical sorting if there's an error
            return sorted(active_files, key=lambda x: x[0])
    
    def _sort_files_by_chapter(self):
        """Sort files by their chapter number from the book outline"""
        try:
            # Get the book outline to determine chapter order
            book_outline = build_book_outline_from_quarto_yml()
            ordered_files = get_qmd_order_from_quarto_yml(QUARTO_YML_PATH)
            
            # Create a mapping of file names to their position in the book
            file_order = {}
            for i, qmd_file in enumerate(ordered_files):
                file_name = Path(qmd_file).name
                file_order[file_name] = i
            
            # Sort the files by their position in the book outline
            def sort_key(file_name):
                return file_order.get(file_name, 999)  # Put unknown files at the end
            
            return sorted(self.file_progress.keys(), key=sort_key)
            
        except Exception:
            # Fallback to alphabetical sorting if there's an error
            return sorted(self.file_progress.keys())

def extract_chapter_title_from_qmd(qmd_file_path):
    """
    Extract the chapter title from the main # header in a QMD file.
    
    Args:
        qmd_file_path (str): Path to the QMD file
        
    Returns:
        str: Chapter title from the main header, or None if not found
    """
    try:
        with open(qmd_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the first # header (main chapter title)
        lines = content.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('# ') and not stripped.startswith('## '):
                # Extract the title (remove the # and any attributes)
                title = stripped[2:]  # Remove '# '
                # Remove any attributes like {#sec-...}
                title = re.sub(r'\s*\{[^}]*\}\s*$', '', title)
                return title.strip()
        
        return None
    except Exception as e:
        print(f"Warning: Could not extract chapter title from {qmd_file_path}: {e}")
        return None

def build_book_outline_from_quarto_yml():
    """
    Automatically build the book outline from _quarto.yml and QMD files.
    
    This function reads the _quarto.yml file to get the chapter order,
    then extracts chapter titles from the main # headers in each QMD file.
    
    Returns:
        list: List of chapter titles in the order specified by _quarto.yml
    """
    global BOOK_OUTLINE
    
    # If already built, return it
    if BOOK_OUTLINE is not None:
        return BOOK_OUTLINE
    
    # Thread-safe building with a lock
    if not hasattr(thread_local, 'building_outline'):
        thread_local.building_outline = False
    
    # If another thread is already building, wait
    if thread_local.building_outline:
        # Wait for the outline to be built
        while BOOK_OUTLINE is None:
            time.sleep(0.1)
        return BOOK_OUTLINE
    
    # Mark that we're building the outline
    thread_local.building_outline = True
    
    yml_path = QUARTO_YML_PATH
    if not os.path.exists(yml_path):
        print("Warning: _quarto.yml not found, cannot build book outline")
        BOOK_OUTLINE = []
        thread_local.building_outline = False
        return BOOK_OUTLINE
    
    try:
        # Get ordered QMD files from _quarto.yml
        ordered_qmd_files = get_qmd_order_from_quarto_yml(yml_path)
        
        if not ordered_qmd_files:
            print("Warning: No QMD files found in _quarto.yml, cannot build book outline")
            BOOK_OUTLINE = []
            thread_local.building_outline = False
            return BOOK_OUTLINE
        
        # Extract chapter titles from each QMD file in the order specified by _quarto.yml
        chapter_titles = []
        
        for qmd_file in ordered_qmd_files:
            # Convert relative path to absolute path
            abs_path = os.path.join(os.getcwd(), qmd_file)
            if os.path.exists(abs_path):
                title = extract_chapter_title_from_qmd(abs_path)
                if title:
                    chapter_titles.append(title)
                else:
                    # If we can't extract the title, skip this file and log a warning
                    print(f"Warning: Could not extract chapter title from {qmd_file}, skipping from book outline")
                    continue
            else:
                print(f"Warning: QMD file not found: {abs_path}, skipping from book outline")
                continue
        
        BOOK_OUTLINE = chapter_titles
        print(f"✅ Built book outline from _quarto.yml with {len(chapter_titles)} chapters")
        thread_local.building_outline = False
        return BOOK_OUTLINE
        
    except Exception as e:
        print(f"Warning: Error building book outline from _quarto.yml: {e}")
        print("Cannot build book outline")
        BOOK_OUTLINE = []
        thread_local.building_outline = False
        return BOOK_OUTLINE

def generate_for_file_parallel(qmd_file, args, progress_tracker=None):
    """
    Thread-safe version of generate_for_file for parallel processing.
    
    Args:
        qmd_file (str): Path to QMD file
        args: Command line arguments
        progress_tracker (ProgressTracker): Progress tracker for real-time updates
    
    Returns:
        dict: Result summary with success/error info
    """
    thread_id = threading.get_ident()
    start_time = time.time()
    
    file_name = Path(qmd_file).name
    
    try:
        # Initialize thread-local OpenAI client to avoid sharing between threads
        if not hasattr(thread_local, 'openai_client'):
            thread_local.openai_client = OpenAI()
        
        # Call the existing generate_for_file logic but with isolated client and progress tracker
        result = _generate_for_file_with_client(qmd_file, args, thread_local.openai_client, progress_tracker)
        
        elapsed = time.time() - start_time
        
        # Mark as completed
        if progress_tracker:
            progress_tracker.complete_file(file_name, True)
        
        return {
            "file": qmd_file,
            "success": True,
            "elapsed": elapsed,
            "thread_id": thread_id,
            **result
        }
    except Exception as e:
        elapsed = time.time() - start_time
        
        # Mark as failed
        if progress_tracker:
            progress_tracker.complete_file(file_name, False)
        
        return {
            "file": qmd_file,
            "success": False,
            "elapsed": elapsed,
            "thread_id": thread_id,
            "error": str(e)
        }

def _generate_for_file_with_client(qmd_file, args, client, progress_tracker=None):
    """
    Core logic of generate_for_file that accepts a specific OpenAI client.
    This allows for thread-safe parallel processing.
    """
    file_name = Path(qmd_file).name
    
    # Read the QMD file
    with open(qmd_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract sections
    sections = extract_sections_with_ids(content)
    if not sections:
        raise ValueError("No sections found or sections missing IDs")
    
    # Update progress with total sections found
    if progress_tracker:
        progress_tracker.update_file_progress(file_name, 0, len(sections))
    
    # Extract chapter info
    chapter_number, chapter_title = extract_chapter_info(qmd_file)
    
    # Generate quizzes for each section, passing previous quiz data
    quiz_sections = []
    sections_with_quizzes = 0
    sections_without_quizzes = 0
    previous_quizzes = []  # Track previous quiz data for variety
    
    for i, section in enumerate(sections):
        # Update progress for current section
        if progress_tracker:
            progress_tracker.update_file_progress(file_name, i + 1, len(sections))
        
        # Build user prompt with chapter context and previous quiz data
        user_prompt = build_user_prompt(
            section['section_title'], 
            section['section_text'],
            chapter_number,
            chapter_title,
            previous_quizzes  # Pass previous quiz data for variety
        )
        
        # Call OpenAI with provided client
        response = call_openai(client, SYSTEM_PROMPT, user_prompt, args.model)
        
        if response.get('quiz_needed', False):
            sections_with_quizzes += 1
            # Add to previous quizzes for next section
            previous_quizzes.append(response)
        else:
            sections_without_quizzes += 1
            # Still add to previous quizzes to maintain section count
            previous_quizzes.append(response)
        
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
    
    # Create output file path in the same directory as the QMD file
    qmd_dir = os.path.dirname(qmd_file)
    qmd_basename = os.path.splitext(os.path.basename(qmd_file))[0]
    
    # If output is the default "quizzes.json", use the QMD filename as prefix
    if args.output == "quizzes.json":
        output_file = os.path.join(qmd_dir, f"{qmd_basename}_quizzes.json")
    else:
        # If user specified a custom output name, use it in the QMD directory
        output_file = os.path.join(qmd_dir, args.output)
    
    # Save to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(quiz_data, f, indent=2, ensure_ascii=False)
    
    # Update QMD frontmatter
    update_qmd_frontmatter(qmd_file, os.path.basename(output_file))
    
    return {
        "output_file": output_file,
        "sections_with_quizzes": sections_with_quizzes,
        "sections_without_quizzes": sections_without_quizzes,
        "total_sections": len(sections)
    }

def generate_for_directory(directory, args):
    """
    Generate quizzes for all QMD files in a directory, in the order specified by _quarto.yml if present.
    """
    print(f"Generating quizzes for directory: {directory}")
    
    # Use the global QUARTO_YML_PATH
    yml_path = QUARTO_YML_PATH
    if os.path.exists(yml_path):
        print(f"Using _quarto.yml for chapter order: {yml_path}")
    else:
        print("No _quarto.yml found in project root. Using default file order.")
    ordered_files = []
    if os.path.exists(yml_path):
        try:
            ordered_files = get_qmd_order_from_quarto_yml(yml_path)
            print(f"Found {len(ordered_files)} .qmd files in _quarto.yml chapters section.")
        except Exception as e:
            print(f"⚠️  Could not parse _quarto.yml for chapter order: {e}")
    
    # Find all .qmd files in the directory (recursively)
    qmd_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.qmd') or file.endswith('.md'):
                qmd_files.append(os.path.relpath(os.path.join(root, file), os.getcwd()))
    
    # Order files: first those in ordered_files, then the rest
    ordered_qmds = []
    seen = set()
    for f in ordered_files:
        # Try both as-is and with/without leading './'
        f_norm = f.lstrip('./')
        for q in qmd_files:
            if q == f or q == f_norm or q.endswith(f_norm):
                ordered_qmds.append(q)
                seen.add(q)
                break
    # Add remaining files not in chapters
    for q in qmd_files:
        if q not in seen:
            ordered_qmds.append(q)
    
    if not ordered_qmds:
        print("❌ No QMD files found in directory")
        return
    
    print(f"Found {len(ordered_qmds)} QMD files (ordered by _quarto.yml where possible)")
    
    for qmd_file in ordered_qmds:
        print(f"\n{'='*60}")
        base_name = os.path.splitext(os.path.basename(qmd_file))[0]
        args.output = f"{base_name}_quizzes.json"
        generate_for_file(qmd_file, args)

def generate_for_directory_parallel(directory, args):
    """
    Parallel version of generate_for_directory using ThreadPoolExecutor.
    
    Processes multiple QMD files simultaneously with one thread per file.
    This provides significant speedup for directories with many files.
    
    Args:
        directory (str): Directory containing QMD files
        args: Command line arguments  
    """
    print(f"🚀 Generating quizzes for directory: {directory} (parallel mode)")
    
    # Use the global QUARTO_YML_PATH
    yml_path = QUARTO_YML_PATH
    if os.path.exists(yml_path):
        print(f"Using _quarto.yml for chapter order: {yml_path}")
    else:
        print("No _quarto.yml found in project root. Using default file order.")
    
    # Get ordered files (same logic as existing sequential version)
    ordered_files = []
    if os.path.exists(yml_path):
        try:
            ordered_files = get_qmd_order_from_quarto_yml(yml_path)
            print(f"Found {len(ordered_files)} .qmd files in _quarto.yml chapters section.")
        except Exception as e:
            print(f"⚠️  Could not parse _quarto.yml for chapter order: {e}")
    
    # Find all .qmd files in the directory (recursively)
    qmd_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.qmd') or file.endswith('.md'):
                qmd_files.append(os.path.relpath(os.path.join(root, file), os.getcwd()))
    
    # Order files: first those in ordered_files, then the rest
    ordered_qmds = []
    seen = set()
    for f in ordered_files:
        # Try both as-is and with/without leading './'
        f_norm = f.lstrip('./')
        for q in qmd_files:
            if q == f or q == f_norm or q.endswith(f_norm):
                ordered_qmds.append(q)
                seen.add(q)
                break
    # Add remaining files not in chapters
    for q in qmd_files:
        if q not in seen:
            ordered_qmds.append(q)
    
    if not ordered_qmds:
        print("❌ No QMD files found in directory")
        return
    
    # Set reasonable defaults for threading
    max_workers = args.max_workers
    if max_workers is None:
        # Default: one thread per file, but cap at 4 to avoid overwhelming OpenAI API
        max_workers = min(len(ordered_qmds), 4)
    
    print(f"📚 Processing {len(ordered_qmds)} files with {max_workers} parallel threads")
    
    # Initialize progress tracker
    progress_tracker = ProgressTracker(len(ordered_qmds))
    results = []
    
    # Execute in parallel with controlled concurrency
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks with individual args for each file
        future_to_file = {}
        for qmd_file in ordered_qmds:
            # Create individual args for each file to avoid conflicts
            file_args = argparse.Namespace(**vars(args))
            base_name = os.path.splitext(os.path.basename(qmd_file))[0]
            file_args.output = f"{base_name}_quizzes.json"
            
            future = executor.submit(generate_for_file_parallel, qmd_file, file_args, progress_tracker)
            future_to_file[future] = qmd_file
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            file = future_to_file[future]
            try:
                # Add timeout to prevent hanging
                result = future.result(timeout=300)  # 5 minute timeout per file
                results.append(result)
            except concurrent.futures.TimeoutError:
                file_name = Path(file).name
                progress_tracker.complete_file(file_name, False)
                results.append({
                    "file": file,
                    "success": False,
                    "error": "Timeout after 5 minutes"
                })
            except Exception as e:
                file_name = Path(file).name
                progress_tracker.complete_file(file_name, False)
                results.append({
                    "file": file,
                    "success": False,
                    "error": str(e)
                })
    
    # Summary report
    total_elapsed = time.time() - start_time
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    total_cpu_time = sum(r.get("elapsed", 0) for r in results)
    speedup = total_cpu_time / total_elapsed if total_elapsed > 0 else 1
    
    # Clear the progress display and show final summary
    print("\033[2J\033[H", end="", flush=True)
    print(f"🏁 Parallel Generation Complete!")
    print("=" * 80)
    print(f"   ✅ Successful: {len(successful)}")
    print(f"   ❌ Failed: {len(failed)}")
    print(f"   ⏱️  Total CPU time: {total_cpu_time:.1f}s")
    print(f"   ⚡ Wall clock time: {total_elapsed:.1f}s")
    print(f"   🚀 Speedup: {speedup:.1f}x")
    
    if successful:
        total_sections = sum(r.get("total_sections", 0) for r in successful)
        total_with_quiz = sum(r.get("sections_with_quizzes", 0) for r in successful)
        print(f"   📊 Generated quizzes for {total_with_quiz}/{total_sections} sections across all files")
    
    if failed:
        print(f"\n❌ Failed files:")
        for r in failed:
            print(f"   - {Path(r['file']).name}: {r.get('error', 'Unknown error')}")
    
    print("=" * 80)

def clean_slug(title):
    """Creates a URL-friendly slug from a title."""
    import re
    # Simple slug creation - convert to lowercase, replace spaces/special chars with hyphens
    slug = re.sub(r'[^\w\s-]', '', title.lower())
    slug = re.sub(r'[-\s]+', '-', slug)
    return slug.strip('-')

def format_mcq_question(question_text):
    """Format MCQ question with proper choice formatting."""
    # Detect MCQ options (A), B), etc.) and reformat to a), b), ...
    option_pattern = re.compile(r"([A-D])\) ?(.*?)(?=(?:[A-D]\)|$))", re.DOTALL)
    # Find the question and options
    lines = question_text.split("\n")
    q = []
    opts = []
    for line in lines:
        if option_pattern.search(line):
            opts.extend(option_pattern.findall(line))
        else:
            q.append(line)
    qtext = " ".join(q).strip()
    if opts:
        opt_lines = [f"   {chr(96+ord(opt[0].upper())-64)}) {opt[1].strip()}" for opt in opts]  # a), b), ...
        return f"{qtext}\n" + "\n".join(opt_lines)
    else:
        return question_text

def is_only_options(qtext):
    """Returns True if qtext is only a list of options (a), b), etc.) and no question stem."""
    lines = [l.strip() for l in qtext.split('\n') if l.strip()]
    return all(re.match(r'^[a-dA-D]\)', l) for l in lines)

def format_quiz_block(qa_pairs, answer_ref, section_id):
    """Formats the questions into a Quarto callout block."""
    # Handle the case where quiz_needed is False
    if isinstance(qa_pairs, dict) and qa_pairs.get("quiz_needed", True) is False:
        return ""
    
    # Extract questions from the nested structure
    questions = qa_pairs.get("questions", qa_pairs) if isinstance(qa_pairs, dict) else qa_pairs
    if not questions or (isinstance(questions, list) and len(questions) == 0):
        return ""
    
    quiz_id = f"{QUESTION_ID_PREFIX}{section_id}"
    formatted_questions = []
    
    for i, qa in enumerate(questions):
        qtext = qa['question'] if isinstance(qa, dict) else qa
        # Skip questions that are only options
        if is_only_options(qtext):
            continue
        
        # Handle different question types
        if isinstance(qa, dict) and qa.get('question_type') == 'MCQ':
            # Format MCQ with choices
            question_text = qtext
            choices = qa.get('choices', [])
            if choices:
                formatted_q = f"{question_text}\n"
                for j, choice in enumerate(choices):
                    letter = chr(ord('a') + j)
                    formatted_q += f"   {letter}) {choice}\n"
                formatted_questions.append(f"{i+1}. {formatted_q.rstrip()}")
            else:
                formatted = format_mcq_question(qtext)
                formatted_questions.append(f"{i+1}. {formatted}")
        else:
            # For other question types, use the original formatting
            formatted = format_mcq_question(qtext)
            formatted_questions.append(f"{i+1}. {formatted}")
    
    if not formatted_questions:
        return ""
    
    # Add a blank line before closing :::
    return (
        f"::: {{{QUIZ_CALLOUT_CLASS} #{quiz_id}}}\n\n"
        + "\n\n".join(formatted_questions)
        + f"\n\n{REFERENCE_TEXT} \\ref{{{answer_ref}}}.\n\n:::\n"
    )

def indent_answer_explanation(answer):
    """Indent answer explanations properly."""
    return '\n'.join([f"   {line}" if line.strip() else "" for line in answer.strip().split("\n")])

def format_mcq_answer(question, answer):
    """Format MCQ answer with question and indented explanation."""
    q_lines = format_mcq_question(question).split("\n")
    answer_lines = indent_answer_explanation(answer)
    return "\n".join(q_lines) + "\n\n" + answer_lines

def format_answer_block(section_id, qa_pairs):
    """Formats the answers into a Quarto callout block."""
    # Handle the case where quiz_needed is False
    if isinstance(qa_pairs, dict) and qa_pairs.get("quiz_needed", True) is False:
        return ""
    
    # Extract questions from the nested structure
    questions = qa_pairs.get("questions", qa_pairs) if isinstance(qa_pairs, dict) else qa_pairs
    if not questions or (isinstance(questions, list) and len(questions) == 0):
        return ""
    
    lines = []
    for i, qa in enumerate(questions):
        qtext = qa['question'] if isinstance(qa, dict) else qa
        ans = qa['answer'] if isinstance(qa, dict) else ''
        learning_obj = qa.get('learning_objective', '') if isinstance(qa, dict) else ''
        qtype = qa.get('question_type', '') if isinstance(qa, dict) else ''
        
        # Handle different question types for formatting
        if isinstance(qa, dict) and qa.get('question_type') == 'MCQ':
            # Format MCQ with choices - only bold the main question line
            question_text = qtext
            choices = qa.get('choices', [])
            if choices:
                formatted_q = f"**{question_text}**\n"
                for j, choice in enumerate(choices):
                    letter = chr(ord('a') + j)
                    formatted_q += f"   {letter}) {choice}\n"
                formatted_q = formatted_q.rstrip()
            else:
                # MCQ without choices array - format the question text
                formatted_q = format_mcq_question(qtext)
                # Make only the main question line bold
                formatted_q = formatted_q.replace(f"{i+1}. ", f"{i+1}. **")
                formatted_q = formatted_q.replace("\n", "**\n", 1)
        else:
            # For other question types, format directly and make the main question line bold
            formatted_q = f"**{qtext}**"
        
        # Construct the answer string for all types
        if qtype == 'FILL' and ans:
            # Extract the answer up to the first period (or the whole answer if no period)
            first_period = ans.find('.')
            if first_period != -1:
                fill_word = ans[:first_period].strip()
                rest = ans[first_period+1:].strip()
            else:
                fill_word = ans.strip()
                rest = ''
            answer_str = f'The answer is "{fill_word}".'
            if rest:
                answer_str += f' {rest}'
        else:
            answer_str = ans
        
        # Special handling for ORDER-type answers
        if qtype == 'ORDER':
            formatted_a = indent_answer_explanation(f'*Answer*: The order is as follows: {answer_str}')
        else:
            formatted_a = indent_answer_explanation(f'*Answer*: {answer_str}')
        
        # Format learning objective with proper indentation to match answer
        if learning_obj:
            formatted_lo = f"\n\n   *Learning Objective*: {learning_obj}"
        else:
            formatted_lo = ""
        
        lines.append(f"{i+1}. {formatted_q}\n\n{formatted_a}{formatted_lo}")
    
    # Add a blank line after opening ::: and only one before closing :::
    return (
        f":::{{{ANSWER_CALLOUT_CLASS} #{ANSWER_ID_PREFIX}{section_id}}}\n\n"
        + "\n\n".join(lines)
        + "\n\n:::\n"
    )

def insert_quiz_at_end(match, quiz_block):
    """Helper function to insert quiz block at the end of a section."""
    section_text = match.group(0)  # Keep original newlines
    # Remove any existing quiz callout in this section
    section_text = re.sub(rf"::: \{{{QUIZ_CALLOUT_CLASS}[\s\S]*?:::\n?", "", section_text)
    # Only insert if quiz_block is not empty and not already present
    if quiz_block.strip() and quiz_block.strip() not in section_text:
        return section_text.rstrip() + '\n\n' + quiz_block.strip() + '\n\n'
    return section_text  # Return original section text with its newlines intact

def clean_existing_quiz_blocks(markdown_text):
    """
    Remove all quiz and answer callouts, and the entire '## Quiz Answers' section with its content.
    Returns:
        cleaned_text (str): the cleaned markdown
        changed (bool): whether anything was removed
        quiz_removed_count (int): number of quiz blocks removed
        answer_removed_count (int): number of answer blocks removed
    """
    original_len = len(markdown_text)
    quiz_removed_count = 0
    answer_removed_count = 0

    # --- Remove quiz callouts ---
    quiz_callout_pattern = re.compile(
        r":::\s*\{[^}]*?" + re.escape(QUIZ_CALLOUT_CLASS.lstrip('.')) + r"[^}]*?\}[\s\S]*?:::\s*\n?",
        re.DOTALL | re.IGNORECASE
    )
    cleaned, quiz_removed_count = quiz_callout_pattern.subn("", markdown_text)

    # --- Remove all answer callouts ---
    answer_callout_pattern = re.compile(
        r":::\s*\{[^}]*?" + re.escape(ANSWER_CALLOUT_CLASS.lstrip('.')) + r"[^}]*?\}[\s\S]*?:::\s*\n?",
        re.DOTALL | re.IGNORECASE
    )
    cleaned, answer_removed_count = answer_callout_pattern.subn("", cleaned)

    # --- Remove the entire '## Quiz Answers' section (header + all content) ---
    quiz_answers_section_pattern = re.compile(
        r"(^##\s+" + re.escape(SELF_CHECK_ANSWERS_HEADER) + r"[\s\S]*?)(?=^##\s|\Z)", re.MULTILINE
    )
    cleaned, section_removed_count = quiz_answers_section_pattern.subn("", cleaned)

    changed = len(cleaned) != original_len
    return cleaned, changed, quiz_removed_count, answer_removed_count

def insert_quizzes_into_markdown(qmd_file_path, quiz_file_path):
    """
    Insert quizzes into a markdown file using robust existing insertion logic.
    This function inserts quiz callouts into QMD files based on the quiz data.
    It uses YAML processing to validate quiz integrity and includes a complete
    Self-Check Answers section at the end.
    """
    try:
        # Read the QMD file as lines
        with open(qmd_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        content = ''.join(lines)

        # If quiz_file_path not provided, extract from QMD frontmatter using existing function
        if quiz_file_path is None:
            quiz_file_path = find_quiz_file_from_qmd(qmd_file_path)
            if not quiz_file_path:
                print("❌ No quiz file specified in QMD frontmatter")
                print("   Make sure the QMD file has 'quiz: filename.json' in its frontmatter")
                return

        print(f"Inserting quizzes from {os.path.basename(quiz_file_path)} into {os.path.basename(qmd_file_path)}")

        # Validate quiz file exists
        if not os.path.exists(quiz_file_path):
            print(f"❌ Quiz file not found: {quiz_file_path}")
            return

        # Read and validate the quiz JSON file
        try:
            with open(quiz_file_path, 'r', encoding='utf-8') as f:
                quiz_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in quiz file: {str(e)}")
            return

        # Validate quiz data structure using existing schema validation
        try:
            validate(instance=quiz_data, schema=QUIZ_FILE_SCHEMA)
            print("  ✅ Quiz file structure is valid")
        except ValidationError as e:
            print(f"❌ Quiz file validation failed: {e.message}")
            return

        # Extract and validate sections from the markdown file
        sections = extract_sections_with_ids(content)
        if not sections:
            print("❌ No sections found in QMD file or sections missing IDs")
            return

        print(f"  ✅ Found {len(sections)} sections in QMD file")

        # Clean up any existing quiz/answer callouts first
        content, cleaned_something, quiz_count, answer_count = clean_existing_quiz_blocks(content)
        if cleaned_something:
            print(f"  🧹 Cleaned up existing content: Removed {quiz_count} quiz callout(s) and {answer_count} answer callout(s)")
        lines = content.splitlines(keepends=True)

        # Find safe insertion points
        insertion_points = find_safe_insertion_points(lines)
        # Map section title to insertion index
        insertion_map = {(title, sid): idx for (title, sid, idx) in insertion_points}

        # Create mapping of section_id to quiz data and validate
        qa_by_section = {}
        valid_quiz_count = 0
        for section_data in quiz_data.get('sections', []):
            section_id = section_data['section_id']
            quiz_info = section_data.get('quiz_data', {})
            section_title = None
            for section in sections:
                if section['section_id'] == section_id:
                    section_title = section['section_title']
                    break
            if not section_title:
                print(f"  ⚠️  Warning: Section {section_id} not found in QMD file, skipping")
                continue
            if quiz_info.get('quiz_needed', False):
                questions = quiz_info.get('questions', [])
                if len(questions) == 0:
                    print(f"  ⏭️  Skipping section {section_id} - all questions were removed by user")
                    continue
                if validate_individual_quiz_response(quiz_info):
                    qa_by_section[(section_title, section_id.lstrip('#'))] = quiz_info
                    valid_quiz_count += 1
                else:
                    print(f"  ⚠️  Warning: Invalid quiz data for section {section_id}, skipping")

        if not qa_by_section:
            print("⚠️  No quiz sections to insert (all questions may have been removed by user)")
            print("  Proceeding to update frontmatter only...")
            quiz_filename = os.path.basename(quiz_file_path)
            update_qmd_frontmatter(qmd_file_path, quiz_filename)
            print(f"✅ Updated frontmatter with quiz: {quiz_filename}")
            return

        print(f"  ✅ Found {valid_quiz_count} valid quiz section(s)")

        # Insert quizzes using line-based logic with reverse-order insertion
        # CRITICAL: We must insert from bottom to top (highest line number to lowest)
        # because each insertion shifts all subsequent line numbers down by the number
        # of lines inserted. If we inserted top-to-bottom, all calculated insertion
        # points for later sections would become incorrect.
        #
        # Example: If we have quizzes at lines 100, 200, 300 and we insert top-to-bottom:
        # 1. Insert at line 100 (adds 5 lines) -> quiz at 200 is now at 205, quiz at 300 is now at 305
        # 2. Insert at line 200 (but actual position is now 205) -> WRONG POSITION!
        #
        # By inserting bottom-to-top (300, 200, 100), each insertion doesn't affect
        # the line numbers of the remaining insertions above it.
        inserted_count = 0
        answer_blocks = []
        
        # Collect all insertions with their indices for reverse-order processing
        insertions_to_make = []
        for (section_title, section_id), qa_pairs in qa_by_section.items():
            quiz_block = format_quiz_block(qa_pairs, f"{ANSWER_ID_PREFIX}{section_id}", section_id)
            answer_block = format_answer_block(section_id, qa_pairs)
            
            if quiz_block.strip():
                # Find insertion index - section_id is already stripped of # in qa_by_section
                idx = insertion_map.get((section_title, section_id))
                if idx is not None:
                    # Only insert if not already present
                    already_present = any(quiz_block.strip() in l for l in lines[max(0, idx-5):idx+5])
                    if not already_present:
                        insertions_to_make.append((idx, quiz_block, section_title))
                else:
                    print(f"    ⚠️  No valid insertion point found for section: {section_title}")
            else:
                print(f"    ⏭️  No quiz block generated for section '{section_title}' (quiz not needed)")
                
            if answer_block.strip():
                answer_blocks.append(answer_block)
        
        # Sort insertions by line number in descending order (bottom to top)
        # This is the KEY to preventing line number shifting issues:
        # - Higher line numbers are processed first
        # - Each insertion only affects line numbers below it
        # - Remaining insertion points above stay valid
        insertions_to_make.sort(key=lambda x: x[0], reverse=True)
        
        # Execute insertions from bottom to top to maintain line number accuracy
        for idx, quiz_block, section_title in insertions_to_make:
            # Insert quiz block with exactly one empty line before and after
            # quiz_block already ends with '\n' after ':::', so ensure one blank line after
            lines.insert(idx, '\n' + quiz_block.rstrip() + '\n\n')
            inserted_count += 1
            print(f"    ✅ Inserted quiz for section: {section_title}")

        # Only add non-empty answer blocks
        nonempty_answer_blocks = [b for b in answer_blocks if b.strip() and not b.strip().isspace() and ANSWER_ID_PREFIX in b]
        print(f"  📝 Found {len(nonempty_answer_blocks)} non-empty answer blocks to append")
        if nonempty_answer_blocks:
            print(f"  📚 Appending final '{SELF_CHECK_ANSWERS_HEADER}' section...")
            # Remove trailing whitespace/newlines at end of file
            while lines and lines[-1].strip() == '':
                lines.pop()
            # Ensure exactly one blank line before Self-Check Answers
            lines.append(f"\n{SELF_CHECK_ANSWERS_SECTION_HEADER}\n")
            lines.append("\n" + "\n\n".join([block.strip() for block in nonempty_answer_blocks]) + "\n")
            print(f"✅ Added {SELF_CHECK_ANSWERS_HEADER} section with {len(nonempty_answer_blocks)} answer block(s)")
        else:
            print(f"  ⏭️  No answer blocks to append")

        # Write the modified content back to the file
        with open(qmd_file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        # Update frontmatter to include quiz reference
        quiz_filename = os.path.basename(quiz_file_path)
        update_qmd_frontmatter(qmd_file_path, quiz_filename)
        print(f"✅ Successfully inserted {inserted_count} quiz(es) into {os.path.basename(qmd_file_path)}")
        if nonempty_answer_blocks:
            print(f"✅ Added {SELF_CHECK_ANSWERS_HEADER} section with {len(nonempty_answer_blocks)} answer block(s)")
        print(f"✅ Updated frontmatter with quiz: {quiz_filename}")
    except Exception as e:
        print(f"❌ Error inserting quizzes: {str(e)}")
        import traceback
        traceback.print_exc()

def clean_single_file(qmd_file, args):
    """
    Clean all quiz content from a single QMD file.
    
    This function removes all quiz-related content from a QMD file,
    including quiz callouts, the quiz answers section, and quiz
    frontmatter entries. It can create backups and perform dry runs.
    
    Args:
        qmd_file (str): Path to the QMD file to clean
        args (argparse.Namespace): Command line arguments including backup and dry_run flags
        
    Note:
        - Removes quiz question and answer callouts using global patterns
        - Removes the entire "Quiz Answers" section
        - Removes quiz entry from YAML frontmatter
        - Supports backup creation and dry-run mode
    """
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
        quiz_question_pattern = re.compile(
            r":::\s*\{[^}]*?" + re.escape(QUIZ_QUESTION_CALLOUT_CLASS.lstrip('.')) + r"[^}]*?\}[\s\S]*?:::\s*\n?",
            re.DOTALL | re.IGNORECASE
        )
        
        # Pattern to match quiz answer callouts (more flexible for additional attributes)
        quiz_answer_pattern = re.compile(
            r":::\s*\{[^}]*?" + re.escape(QUIZ_ANSWER_CALLOUT_CLASS.lstrip('.')) + r"[^}]*?\}[\s\S]*?:::\s*\n?",
            re.DOTALL | re.IGNORECASE
        )
        
        # Count how many callouts we find
        question_matches = quiz_question_pattern.findall(content)
        answer_matches = quiz_answer_pattern.findall(content)
        
        # Count quiz answers sections
        quiz_answers_pattern = re.compile(QUIZ_ANSWERS_SECTION_PATTERN, re.DOTALL | re.MULTILINE)
        quiz_answers_matches = quiz_answers_pattern.findall(content)
        
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
                    # Keep the quiz key but remove all quiz content from the body
                    # This allows the Lua filter to work but removes inserted content
                    print(f"  ✅ Preserved quiz frontmatter entry (quiz: {frontmatter_data['quiz']})")
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
    """
    Clean all quiz content from all QMD files in a directory.
    
    This function recursively finds all QMD files in a directory and
    cleans quiz content from each one. It supports backup creation
    and dry-run mode for safe operation.
    
    Args:
        directory (str): Path to the directory containing QMD files
        args (argparse.Namespace): Command line arguments including backup and dry_run flags
        
    Note:
        - Processes files recursively through subdirectories
        - Creates backups for each file if requested
        - Supports dry-run mode to preview changes
        - Provides progress tracking for large directories
    """
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
    """
    Verify all quiz files in a directory.
    
    This function would perform verification on all quiz files in a
    directory. Currently a placeholder for future implementation.
    
    Args:
        directory_path (str): Path to the directory to verify
        
    Note:
        This functionality is not yet implemented. It would involve:
        - Finding all JSON and QMD files in the directory
        - Running verification on each file
        - Providing a summary report
    """
    print(f"Verifying quiz files in directory: {directory_path}")
    # Implementation would go here - this is a placeholder
    print("❌ Verify directory functionality not yet implemented")

def get_qmd_order_from_quarto_yml(yml_path):
    """Extract the ordered list of .qmd files from the chapters section of _quarto.yml, including commented ones."""
    with open(yml_path, 'r') as f:
        content = f.read()
    # Find the chapters section
    chapters_match = re.search(r'chapters:\s*\n(.*?)(?=\n\w+:|$)', content, re.DOTALL)
    if not chapters_match:
        return []
    chapters_content = chapters_match.group(1)
    # Extract all .qmd files, including commented ones
    qmd_files = []
    lines = chapters_content.split('\n')
    for line in lines:
        line = line.strip()
        if '.qmd' in line:
            clean_line = re.sub(r'^\s*#\s*', '', line)
            clean_line = re.sub(r'^\s*-\s*', '', clean_line)
            file_match = re.search(r'contents/.*?\.qmd', clean_line)
            if file_match:
                qmd_files.append(file_match.group(0))
    return qmd_files

# Utility function to get ordered .qmd files for a directory based on _quarto.yml

def get_ordered_qmd_files(directory):
    """Return a list of .qmd files in the order specified by _quarto.yml, with unlisted files after."""
    yml_path = QUARTO_YML_PATH
    ordered_files = []
    if os.path.exists(yml_path):
        try:
            ordered_files = get_qmd_order_from_quarto_yml(yml_path)
        except Exception as e:
            print(f"⚠️  Could not parse _quarto.yml for chapter order: {e}")
    # Find all .qmd files in the directory (recursively)
    qmd_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.qmd') or file.endswith('.md'):
                qmd_files.append(os.path.relpath(os.path.join(root, file), os.getcwd()))
    # Order files: first those in ordered_files, then the rest
    ordered_qmds = []
    seen = set()
    for f in ordered_files:
        f_norm = f.lstrip('./')
        for q in qmd_files:
            if q == f or q == f_norm or q.endswith(f_norm):
                ordered_qmds.append(q)
                seen.add(q)
                break
    for q in qmd_files:
        if q not in seen:
            ordered_qmds.append(q)
    return ordered_qmds

# Update all directory-based commands to use get_ordered_qmd_files

def generate_for_directory(directory, args):
    print(f"Generating quizzes for directory: {directory}")
    if os.path.exists(QUARTO_YML_PATH):
        print(f"Using _quarto.yml for chapter order: {QUARTO_YML_PATH}")
    else:
        print("No _quarto.yml found in project root. Using default file order.")
    ordered_qmds = get_ordered_qmd_files(directory)
    if not ordered_qmds:
        print("❌ No QMD files found in directory")
        return
    print(f"Found {len(ordered_qmds)} QMD files (ordered by _quarto.yml where possible)")
    for qmd_file in ordered_qmds:
        print(f"\n{'='*60}")
        base_name = os.path.splitext(os.path.basename(qmd_file))[0]
        args.output = f"{base_name}_quizzes.json"
        generate_for_file(qmd_file, args)

def run_clean_mode_directory(directory, args):
    print(f"=== Quiz Clean Mode (Directory) ===")
    print(f"Cleaning quizzes from directory: {directory}")
    if os.path.exists(QUARTO_YML_PATH):
        print(f"Using _quarto.yml for chapter order: {QUARTO_YML_PATH}")
    else:
        print("No _quarto.yml found in project root. Using default file order.")
    ordered_qmds = get_ordered_qmd_files(directory)
    if not ordered_qmds:
        print("❌ No QMD files found in directory")
        return
    print(f"Found {len(ordered_qmds)} QMD files (ordered by _quarto.yml where possible)")
    if args.dry_run:
        print("🔍 DRY RUN - Would clean the following files:")
        for qmd_file in ordered_qmds:
            print(f"  - {qmd_file}")
        return
    for i, qmd_file in enumerate(ordered_qmds, 1):
        print(f"\n[{i}/{len(ordered_qmds)}] Cleaning: {qmd_file}")
        try:
            file_args = argparse.Namespace()
            file_args.backup = args.backup
            file_args.dry_run = args.dry_run
            clean_single_file(qmd_file, file_args)
        except Exception as e:
            print(f"❌ Error cleaning {qmd_file}: {str(e)}")
    print(f"\n✅ Clean operation complete for {len(ordered_qmds)} files")

def run_verify_mode_directory(directory_path):
    print("=== Quiz Verify Mode (Directory) ===")
    print(f"Verifying quiz files in directory: {directory_path}")
    if os.path.exists(QUARTO_YML_PATH):
        print(f"Using _quarto.yml for chapter order: {QUARTO_YML_PATH}")
    else:
        print("No _quarto.yml found in project root. Using default file order.")
    ordered_qmds = get_ordered_qmd_files(directory_path)
    if not ordered_qmds:
        print("❌ No QMD files found in directory")
        return
    print(f"Found {len(ordered_qmds)} QMD files (ordered by _quarto.yml where possible)")
    for i, qmd_file in enumerate(ordered_qmds, 1):
        print(f"[{i}/{len(ordered_qmds)}] Verifying: {qmd_file}")
        run_verify_mode_simple(qmd_file)
    print(f"\n✅ Verify operation complete for {len(ordered_qmds)} files")

# If you have run_insert_mode_directory or similar, update it similarly.

def run_insert_mode_directory(directory):
    print(f"=== Quiz Insert Mode (Directory) ===")
    print(f"Inserting quizzes for all quiz JSON files in: {directory}")
    # Find all quiz JSON files in the directory (recursively)
    quiz_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(QUIZ_JSON_SUFFIX):
                quiz_files.append(os.path.join(root, file))
    if not quiz_files:
        print(f"❌ No quiz JSON files found in directory with suffix '{QUIZ_JSON_SUFFIX}'")
        return
    print(f"Found {len(quiz_files)} quiz JSON files.")
    for i, quiz_file in enumerate(quiz_files, 1):
        print(f"[{i}/{len(quiz_files)}] Inserting from: {quiz_file}")
        run_insert_mode_simple(quiz_file)

QUIZ_JSON_SUFFIX = '_quizzes.json'

# Add a helper function to find safe insertion points for quiz callouts

def find_safe_insertion_points(markdown_lines):
    """
    Find safe insertion points for quiz callouts in markdown content.
    
    This function identifies where quiz callouts should be inserted by finding section
    boundaries. It scans through the markdown content to locate section headers and
    determines the optimal insertion point (right before the next section header).
    
    IMPORTANT: The returned insertion indices are used in reverse order (bottom-to-top)
    to prevent line number shifting issues during the actual insertion process.
    
    Args:
        markdown_lines (list): List of markdown lines to analyze
        
    Returns:
        list: Tuples of (section_title, section_id, insertion_index) where:
            - section_title: The title of the section (str)
            - section_id: The section ID without # prefix (str or None)
            - insertion_index: Line number where quiz should be inserted (int)
    
    Algorithm:
        1. Track state to ignore headers inside code blocks (```) or div blocks (:::)
        2. For each valid section header found:
           - Extract title and ID from header
           - Scan forward to find the next section header of same/higher level
           - Record the line number of that next header as insertion point
        3. Only consider headers outside of code/div blocks for both detection and scanning
        
    Note: 
        Headers inside callouts, code blocks, or other div structures are properly
        ignored to ensure insertion points are at actual section boundaries.
    """
    from re import match
    state = {'inside_code_block': False, 'inside_div_block': False}
    insertion_points = []
    section_title = None
    section_id = None
    header_level = None
    for i, line in enumerate(markdown_lines):
        stripped = line.strip()
        # Track code block state
        if stripped.startswith('```'):
            state['inside_code_block'] = not state['inside_code_block']
        # Track div block state
        elif stripped.startswith(':::'):
            state['inside_div_block'] = not state['inside_div_block']
        # Only consider headers outside of code/div blocks
        if not state['inside_code_block'] and not state['inside_div_block']:
            # Match headers at any level (##, ###, ####, etc.)
            m = re.match(r'^(#{2,})\s+(.+?)(\s*\{[^}]*\})?\s*$', stripped)
            if m:
                header_level = len(m.group(1))  # Count the number of # symbols
                section_title = m.group(2).strip()
                attrs = m.group(3) or ''
                id_match = re.search(r'\{#([\w\-]+)\}', attrs)
                section_id = id_match.group(1) if id_match else None
                # Find section boundary: scan until we hit the next section header
                # at the same or higher level (respecting block boundaries)
                j = i + 1
                block_state = state.copy()  # Track nested block state during scanning
                while j < len(markdown_lines):
                    line_j = markdown_lines[j].strip()
                    
                    # Maintain block state tracking during forward scan
                    if line_j.startswith('```'):
                        block_state['inside_code_block'] = not block_state['inside_code_block']
                    elif line_j.startswith(':::'):
                        block_state['inside_div_block'] = not block_state['inside_div_block']
                    
                    # Check for section boundary only when outside all blocks
                    if not block_state['inside_code_block'] and not block_state['inside_div_block']:
                        if line_j.startswith('#'):
                            next_level = len(line_j) - len(line_j.lstrip('#'))
                            # Found boundary: next header at same/higher level (fewer #)
                            if next_level <= header_level:
                                break
                    j += 1
                
                # j now points to the line where the quiz should be inserted
                # (right before the next section header or at end of file)
                insertion_points.append((section_title, section_id, j))
    return insertion_points

# INTEGRATION NOTE: This function is used by insert_quizzes_into_markdown() 
# which implements the complete insertion workflow:
# 1. Read the markdown file as lines
# 2. Find safe insertion points using this function
# 3. Collect all insertions and sort in reverse order (CRITICAL for line number stability)
# 4. Insert quiz callouts from bottom to top
# 5. Write the modified content back to file
#
# The reverse-order insertion prevents line number shifting issues that would
# occur with top-to-bottom insertion in a list structure.

if __name__ == "__main__":
    main()

