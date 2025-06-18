import argparse
import os
import re
import json
from pathlib import Path
from openai import OpenAI, APIError
from datetime import datetime

# Gradio imports
import gradio as gr

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
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "answer": {"type": "string"},
                            "learning_objective": {"type": "string"}
                        },
                        "required": ["question", "answer", "learning_objective"],
                        "additionalProperties": False
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

If you determine a quiz is NOT needed, return this JSON:
```json
{{
    "quiz_needed": false,
    "rationale": "Explanation of why this section doesn't need a quiz (e.g., context-setting, descriptive only, no actionable concepts)."
}}
```

If you determine a quiz IS needed, return this JSON:
```json
{{
    "quiz_needed": true,
    "rationale": {{
        "focus_areas": ["List of 2–3 key areas this section focuses on"],
        "question_strategy": "Brief explanation of why these question types were chosen and how they build understanding",
        "difficulty_progression": "How the questions progress in complexity and support deeper learning",
        "integration": "How these questions connect with concepts introduced earlier in the chapter or textbook (if applicable)",
        "ranking_explanation": "Explanation of why questions are ordered this way and how they support learning"
    }},
    "questions": [
        {{
            "question": "The question text",
            "answer": "The answer text",
            "learning_objective": "What specific understanding or skill this question tests"
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

**Question Types (use variety based on content):**
- **Multiple Choice**: Include answer options A), B), C), D) directly in the question text, each on a new line. The answer field must start with a sentence followed by the correct letter followed by a period (e.g., "The correct answer is B. The gradual change in statistical properties..."), then explanation. Every multiple-choice question must begin with a clear question stem, followed by the answer options. Do not generate questions that consist only of options.
- **Short Answer**: Require explanation of concepts, not just definitions
- **Scenario-Based**: Present realistic ML systems situations requiring application
- **Comparison**: Test understanding of tradeoffs between approaches
- **Fill-in-the-Blank**: Use ____ (four underscores) for missing key terms. Answer should contain only the missing word/phrase followed by a period, plus brief explanation.
- **True/False**: Always require justification in the answer

**Question Variety Instructions:**
- Vary the order and types of questions based on the content
- Don't follow the same pattern every time
- Choose question types that best fit the specific concepts being tested
- Mix difficulty levels throughout, not just in progression
- Consider starting with different question types depending on the section's focus

**Quality Standards:**
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

def extract_sections_with_ids(markdown_text):
    """
    Extracts all level-2 sections (##) with their content and section reference (e.g., {#sec-...}).
    Returns a list of dicts: {section_id, section_title, section_text}
    If any section is missing a reference, prints an error and returns None.
    """
    section_pattern = re.compile(r"^##\s+(.+?)(\s*\{#([\w\-]+)\})?\s*$", re.MULTILINE)
    matches = list(section_pattern.finditer(markdown_text))
    
    # First, validate all sections have IDs
    missing_refs = []
    for match in matches:
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
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        ref = match.group(3)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
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
        return data
    except APIError as e:
        return {"quiz_needed": False, "rationale": f"API error: {str(e)}"}
    except Exception as e:
        return {"quiz_needed": False, "rationale": f"Unexpected error: {str(e)}"}

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
        
    def load_quiz_file(self, file_path=None):
        """Load a quiz JSON file"""
        # Use provided file path or initial file path
        path_to_load = file_path or self.initial_file_path
        
        if not path_to_load:
            return "No file path provided", "No file loaded", "No sections", "", "", "Ready"
            
        try:
            # Check if file exists
            if not os.path.exists(path_to_load):
                return f"File not found: {path_to_load}", "File not found", "No sections", "", "", f"Error: File {path_to_load} does not exist"
            
            with open(path_to_load, 'r', encoding='utf-8') as f:
                self.quiz_data = json.load(f)
            
            # Validate JSON structure
            if not isinstance(self.quiz_data, dict):
                return f"Invalid JSON structure in {path_to_load}", "Invalid file format", "No sections", "", "", "Error: File is not a valid JSON object"
            
            self.sections = self.quiz_data.get('sections', [])
            if not self.sections:
                return f"No sections found in {path_to_load}", "No sections found", "No sections", "", "", f"Loaded {path_to_load} but no sections found"
            
            # Try to load the original .qmd file
            self.load_original_qmd_file(path_to_load)
            
            self.current_section_index = 0
            
            # Load first section
            section = self.sections[0]
            title = f"{section['section_title']} ({section['section_id']})"
            section_text = self.get_full_section_content(section)
            questions_text = self.format_questions(section)
            nav_text = f"Section 1 of {len(self.sections)}"
            
            return title, nav_text, section_text, questions_text, f"Loaded {len(self.sections)} sections"
            
        except json.JSONDecodeError as e:
            return f"Invalid JSON in {path_to_load}: {str(e)}", "JSON Error", "No sections", "", f"Error: Invalid JSON format - {str(e)}"
        except Exception as e:
            return f"Error loading {path_to_load}: {str(e)}", "Error loading file", "No sections", "", f"Error: {str(e)}"
    
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
    
    def format_questions(self, section):
        """Format questions for display"""
        quiz_data = section.get('quiz_data', {})
        
        if not quiz_data.get('quiz_needed', False):
            return "No quiz needed for this section"
        
        questions = quiz_data.get('questions', [])
        if not questions:
            return "No questions available"
        
        formatted = []
        for i, question in enumerate(questions, 1):
            q_text = f"**Q{i}:** {question['question']}\n\n"
            a_text = f"**A:** {question['answer']}\n\n"
            
            if 'learning_objective' in question:
                obj_text = f"*Learning Objective:* {question['learning_objective']}\n\n"
            else:
                obj_text = ""
            
            formatted.append(f"{q_text}{a_text}{obj_text}---\n\n")
        
        return "".join(formatted)
    
    def navigate_section(self, direction):
        """Navigate to previous or next section"""
        if not self.sections:
            return "No file loaded", "No sections", "", "", "No sections available"
        
        if direction == "prev" and self.current_section_index > 0:
            self.current_section_index -= 1
        elif direction == "next" and self.current_section_index < len(self.sections) - 1:
            self.current_section_index += 1
        
        section = self.sections[self.current_section_index]
        title = f"{section['section_title']} ({section['section_id']})"
        section_text = self.get_full_section_content(section)
        questions_text = self.format_questions(section)
        nav_text = f"Section {self.current_section_index + 1} of {len(self.sections)}"
        
        return title, nav_text, section_text, questions_text, f"Showing section {self.current_section_index + 1}"

def create_gradio_interface(initial_file_path=None):
    """Create the Gradio interface"""
    editor = QuizEditorGradio(initial_file_path)
    
    with gr.Blocks(title="Quiz Editor", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# Quiz Editor")
        
        # Top row with section title and navigation (50-50)
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Section Title")
                section_title = gr.Textbox(label="", interactive=False, value="No file loaded")
            
            with gr.Column(scale=1):
                gr.Markdown("### Navigation")
                nav_info = gr.Textbox(label="", interactive=False, value="No sections")
        
        # Section content
        gr.Markdown("### Section Content (from .qmd file)")
        section_text = gr.Textbox(label="", lines=15, interactive=False, max_lines=20, value="No content loaded")
        
        # Questions
        gr.Markdown("### Generated Questions")
        questions_display = gr.Markdown("No questions loaded")
        
        # Bottom row with navigation buttons
        with gr.Row():
            with gr.Column(scale=1):
                prev_btn = gr.Button("← Previous", size="lg")
            with gr.Column(scale=1):
                next_btn = gr.Button("Next →", size="lg")
        
        # Status bar
        status_bar = gr.Textbox(label="Status", interactive=False, value="Ready")
        
        # Event handlers
        prev_btn.click(
            fn=lambda: editor.navigate_section("prev"),
            outputs=[section_title, nav_info, section_text, questions_display, status_bar]
        )
        
        next_btn.click(
            fn=lambda: editor.navigate_section("next"),
            outputs=[section_title, nav_info, section_text, questions_display, status_bar]
        )
        
        # Auto-load if initial file path is provided
        if initial_file_path:
            # Use a separate function to handle the initial load
            def initial_load():
                try:
                    return editor.load_quiz_file(initial_file_path)
                except Exception as e:
                    return "Error loading file", "No sections", "Error occurred", "No questions available", f"Error: {str(e)}"
            
            interface.load(initial_load, outputs=[section_title, nav_info, section_text, questions_display, status_bar])
    
    return interface

def run_gui(quiz_file_path=None):
    """Run the Gradio application"""
    interface = create_gradio_interface(quiz_file_path)
    interface.launch(share=False, server_name="0.0.0.0", server_port=7860)

def main():
    parser = argparse.ArgumentParser(description="Generate quizzes for each section in a markdown file. Each section must have a reference label (e.g., {#sec-...}).")
    parser.add_argument("-f", "--file", help="Path to a markdown (.qmd/.md) file.")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use.")
    parser.add_argument("-o", "--output", default="quizzes.json", help="Path to output JSON file (default: quizzes.json)")
    parser.add_argument("--gui", nargs="?", const="", metavar="QUIZ_FILE", help="Launch GUI mode. Optionally provide quiz JSON file path.")
    args = parser.parse_args()

    if args.gui is not None:
        run_gui(args.gui if args.gui else None)
    elif args.file:
        # Original CLI functionality
        with open(args.file, "r", encoding="utf-8") as f:
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
                "source_file": args.file,
                "total_sections": len(sections),
                "sections_with_quizzes": sum(1 for r in results if r["quiz_data"].get("quiz_needed", False)),
                "sections_without_quizzes": sum(1 for r in results if not r["quiz_data"].get("quiz_needed", False))
            },
            "sections": results
        }

        # Use the specified output path or default to quizzes.json in the same directory as input
        out_path = args.output if os.path.isabs(args.output) else os.path.join(os.path.dirname(args.file), args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Wrote quizzes to {out_path}")
        print(f"  - Total sections processed: {len(sections)}")
        print(f"  - Sections with quizzes: {output_data['metadata']['sections_with_quizzes']}")
        print(f"  - Sections without quizzes: {output_data['metadata']['sections_without_quizzes']}")
    else:
        # Default to GUI mode if no arguments provided
        run_gui(None)

if __name__ == "__main__":
    main()
