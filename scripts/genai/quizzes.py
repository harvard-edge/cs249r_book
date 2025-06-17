import argparse
import os
import re
import logging
import json
from pathlib import Path
from openai import OpenAI, APIError
from slugify import slugify
import gradio as gr
import sys
import tempfile
import shutil
import time

# Client is initialized in main() after logging is set up

# --- Global Callout Type Definitions ---
QUIZ_CALLOUT_CLASS = ".callout-quiz-question"
ANSWER_CALLOUT_CLASS = ".callout-quiz-answer"

# --- Global Reference and ID Prefixes ---
REFERENCE_TEXT = "See Answer"
QUESTION_ID_PREFIX = "quiz-question-"
ANSWER_ID_PREFIX = "quiz-answer-"

# --- Constants ---
STOPWORDS = {"in", "of", "the", "and", "to", "for", "on", "a", "an", "with", "by"}

SYSTEM_PROMPT = """
You are a helpful assistant for a university-level textbook on machine learning systems.
Your task is to first evaluate whether a quiz would be pedagogically valuable for the given section, and if so, generate 1 to 5 self-check questions and answers. Decide the number of questions based on the section's length and complexity.

First, evaluate if this section warrants a quiz by considering:
1. Does it contain concepts that students need to actively understand and apply?
2. Are there potential misconceptions that need to be addressed?
3. Does it present system design tradeoffs or implications that students should reflect on?
4. Does it build on previous knowledge in ways that should be reinforced?

Sections that typically DO NOT need quizzes:
- Pure introductions or context-setting sections
- Sections that primarily tell a story or provide historical context
- Sections that are purely descriptive without conceptual depth
- Sections that are primarily motivational or overview in nature

Sections that typically DO need quizzes:
- Sections that introduce new technical concepts
- Sections that present system design decisions or tradeoffs
- Sections that address common misconceptions
- Sections that require application of concepts
- Sections that build on previous knowledge in important ways

If you determine a quiz is NOT needed, return this JSON:
{
    "quiz_needed": false,
    "rationale": "Explanation of why this section doesn't need a quiz (e.g., context-setting, descriptive only, no actionable concepts)."
}

If you determine a quiz IS needed, return this JSON:
{
    "quiz_needed": true,
    "rationale": {
        "focus_areas": ["List of 2–3 key areas this section focuses on"],
        "question_strategy": "Brief explanation of why these question types were chosen and how they build understanding",
        "difficulty_progression": "How the questions progress in complexity and support deeper learning",
        "integration": "How these questions connect with concepts introduced earlier in the chapter or textbook (if applicable)",
        "ranking_explanation": "Explanation of why questions are ordered this way and how they support learning"
    },
    "questions": [
        {
            "question": "The question text",
            "answer": "The answer text (limit to ~75–150 words total for question + answer)",
            "learning_objective": "What specific understanding or skill this question tests"
        }
    ]
}

Guidelines for questions (only if quiz_needed is true):
- Focus on conceptual understanding, system-level reasoning, and meaningful tradeoffs
- Avoid surface-level recall or trivia
- Use clear, academically appropriate language
- Include at least one question about a system design tradeoff or implication
- If applicable, include a question that addresses a common misconception
- Do not repeat exact phrasing from the source text
- Prioritize questions relevant to *machine learning systems*, not just general ML
- Keep answers concise and informative (~75–150 words total per Q&A)

Special format rules:
- For fill-in-the-blank questions: The blank should be a single word or short phrase, clearly indicated as ____ (four underscores). The answer field should contain only the missing word or phrase.
- For multiple choice: Include answer options directly in the question text (e.g., A), B), C)), each on a new line. The answer field must specify the correct letter and include a short justification (e.g., "B) Correct answer text. This is correct because...").

Question Types (use a mix based on what best tests understanding):
- Multiple Choice Questions (MCQ)
- Short answer explanations requiring conceptual understanding
- Scenario-based questions that test application of concepts
- "Why" questions that probe deeper understanding
- "How" questions that test practical knowledge
- Comparison questions that test understanding of tradeoffs
- Fill-in-the-blank for key concepts
- True/False with justification required

Bloom's Taxonomy Levels (aim for a mix):
- Remember: Basic recall of facts, terms, concepts
- Understand: Explain ideas or concepts in your own words
- Apply: Use information in new situations
- Analyze: Draw connections among ideas
- Evaluate: Justify a stand or decision
- Create: Produce new or original work
"""

# --- Core Functions ---

def clean_slug(title):
    """Creates a URL-friendly slug from a title."""
    words = title.lower().split()
    keywords = [w for w in words if w not in STOPWORDS]
    slug = slugify(" ".join(keywords))
    logging.debug(f"Cleaned title '{title}' to slug '{slug}'")
    return slug

def strip_quiz_callouts(text):
    """Remove any existing Self-Check Quiz callout blocks from the text."""
    return re.sub(r"::: \{\.callout-important title=\"Self-Check Quiz\"[\s\S]*?:::\n?", "", text)

def build_user_prompt(section_title, section_text, chapter_title=None, previous_sections=None):
    """Constructs the user prompt for the language model."""
    # Strip quiz callouts from section_text
    section_text = strip_quiz_callouts(section_text)
    prefix = f'This section is titled "{section_title}".'
    if chapter_title:
        prefix += f' It appears in the chapter "{chapter_title}".'
    
    previous_context = ""
    if previous_sections:
        previous_context = "\n\nPrevious sections in this chapter:\n"
        for prev_title, prev_content in previous_sections:
            # Strip quiz callouts from previous content
            prev_content_clean = strip_quiz_callouts(prev_content)
            # Extract any existing quizzes from previous content (for reference, but now always empty)
            quiz_pattern = re.compile(r":::.*?title=\"Self-Check Quiz\".*?:::", re.DOTALL)
            quiz_match = quiz_pattern.search(prev_content)
            quiz_text = quiz_match.group(0) if quiz_match else "No quiz in this section"
            previous_context += f"\nSection: {prev_title}\n"
            previous_context += f"Content Summary: {prev_content_clean[:200]}...\n"
            previous_context += f"Quiz: {quiz_text}\n"
    
    prompt = f"""
{prefix}

Section content:
{section_text}{previous_context}

Generate a self-check quiz with 3 to 5 well-structured questions and answers based on this section.
Include a rationale explaining your question generation strategy and focus areas.

Return your response in this exact JSON format:
{{
    "rationale": {{
        "focus_areas": ["List 2-3 key areas this section focuses on"],
        "question_strategy": "Explain why you chose these question types",
        "difficulty_progression": "How the questions progress in complexity",
        "integration": "How these questions connect with previous sections"
    }},
    "questions": [
        {{
            "question": "The question text",
            "answer": "The answer text"
        }}
    ]
}}
""".strip()
    logging.debug(f"Built user prompt for section '{section_title}'")
    return prompt

def extract_sections(markdown_text):
    """Extracts chapter title and level-2 markdown sections into (title, content) tuples."""
    logging.debug("Starting section extraction...")
    
    # Extract chapter title (first level-1 header)
    chapter_pattern = re.compile(r"^#\s+(.*)", re.MULTILINE)
    chapter_match = chapter_pattern.search(markdown_text)
    chapter_title = chapter_match.group(1).strip() if chapter_match else None
    logging.debug(f"Extracted chapter title: {chapter_title}")
    
    # Normalize newlines
    markdown_text = markdown_text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Find all ## sections
    sections = []
    section_pattern = re.compile(r'^##\s+(.*?)$', re.MULTILINE)
    matches = list(section_pattern.finditer(markdown_text))
    
    # Process each section
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.start()
        # End is either the start of next section or end of file
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        # Don't strip the content to preserve newlines
        content = markdown_text[start:end]
        sections.append((title, content))
        logging.info(f"Found section {i+1}: {title}")
    
    logging.info(f"Extraction complete. Found {len(sections)} level-2 sections.")
    # Print all sections found
    print("\nSections found:")
    print("=" * 50)
    for i, (title, _) in enumerate(sections, 1):
        print(f"{i}. {title}")
    print("=" * 50)
    
    return chapter_title, sections

def call_openai(client, system_prompt, user_prompt, model="gpt-4o"):
    """Calls the OpenAI API and handles potential errors."""
    logging.info(f"Calling OpenAI API with model '{model}'...")
    try:
        # Remove response_format for models that don't support it
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.4
        )
        content = response.choices[0].message.content
        logging.debug(f"Raw response content from OpenAI: {content}")
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # If the response isn't valid JSON, try to extract JSON from the text
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    logging.error("Failed to parse JSON from response")
                    return {
                        "quiz_needed": False,
                        "rationale": "Failed to parse response as JSON"
                    }
            else:
                logging.error("No JSON found in response")
                return {
                    "quiz_needed": False,
                    "rationale": "No JSON found in response"
                }
        
        # Ensure we always return a dictionary with the expected structure
        if isinstance(data, list):
            # Convert list to expected dictionary format
            return {
                "quiz_needed": True,
                "rationale": {
                    "focus_areas": ["Key concepts from the section"],
                    "question_strategy": "Questions designed to test understanding of core concepts",
                    "difficulty_progression": "Questions progress from basic to advanced concepts",
                    "integration": "Questions build on fundamental concepts"
                },
                "questions": data
            }
        elif isinstance(data, dict):
            # If it's already a dict but missing quiz_needed, add it
            if "quiz_needed" not in data:
                data["quiz_needed"] = True
            # If it's missing rationale but has questions, add default rationale
            if "rationale" not in data and "questions" in data:
                data["rationale"] = {
                    "focus_areas": ["Key concepts from the section"],
                    "question_strategy": "Questions designed to test understanding of core concepts",
                    "difficulty_progression": "Questions progress from basic to advanced concepts",
                    "integration": "Questions build on fundamental concepts"
                }
            return data
        else:
            logging.warning(f"Unexpected JSON structure from OpenAI: {type(data)}")
            return {
                "quiz_needed": False,
                "rationale": "Unexpected response structure"
            }

    except APIError as e:
        logging.error(f"OpenAI API error: {e}")
        return {
            "quiz_needed": False,
            "rationale": f"API error: {str(e)}"
        }
    except Exception as e:
        logging.error(f"An unexpected error occurred in call_openai: {e}")
        return {
            "quiz_needed": False,
            "rationale": f"Unexpected error: {str(e)}"
        }

def format_quiz_block(qa_pairs, answer_ref):
    """Formats the questions into a Quarto callout block."""
    # Check if quiz is needed
    if isinstance(qa_pairs, dict) and not qa_pairs.get("quiz_needed", True):
        return ""  # Return empty string if no quiz is needed
    
    # Extract questions from the nested structure
    questions = qa_pairs.get("questions", qa_pairs) if isinstance(qa_pairs, dict) else qa_pairs
    
    # If there are no questions or questions is empty, return empty string
    if not questions or (isinstance(questions, list) and len(questions) == 0):
        return ""
    
    # Extract slug from answer_ref (e.g., 'answer-my-slug' -> 'my-slug')
    slug = answer_ref.replace(ANSWER_ID_PREFIX, "")
    quiz_id = f"{QUESTION_ID_PREFIX}{slug}"

    # Always number questions, even if there is only one
    if isinstance(questions, list):
        formatted_questions = [f"{i+1}. {qa['question']}" if isinstance(qa, dict) else f"{i+1}. {qa}" for i, qa in enumerate(questions)]
    else:
        formatted_questions = [f"1. {questions}"]
    
    return f"""

::: {{{QUIZ_CALLOUT_CLASS} #{quiz_id}}}

{"\n\n".join(formatted_questions)}

{REFERENCE_TEXT} \\ref{{{answer_ref}}}.
:::

"""

def format_answer_block(slug, qa_pairs):
    """Formats the answers into a Quarto callout block."""
    # Check if quiz is needed
    if isinstance(qa_pairs, dict) and not qa_pairs.get("quiz_needed", True):
        return ""  # Return empty string if no quiz is needed
    
    # Extract questions from the nested structure
    questions = qa_pairs.get("questions", qa_pairs) if isinstance(qa_pairs, dict) else qa_pairs
    
    # If there are no questions or questions is empty, return empty string
    if not questions or (isinstance(questions, list) and len(questions) == 0):
        return ""
    
    # Format Q&A pairs with consistent numbering
    lines = [f"**{i+1}. {qa['question']}**\n\n{qa['answer']}" for i, qa in enumerate(questions)]
    
    # Ensure consistent callout formatting
    return f"""::: {{{ANSWER_CALLOUT_CLASS} #{ANSWER_ID_PREFIX}{slug}}}

{"\n\n".join(lines)}
:::

"""

# --- GUI Mode ---

def launch_gui_mode(client, sections, qa_by_section, filepath, model, pre_select_all=False, tmp_path=None, chapter_title=None):
    """Launches an interactive Gradio GUI to review and select Q&A pairs."""
    logging.info("Preparing to launch GUI mode...")
    final_answers = []
    total_sections = len(sections)
    selected_indices_by_section = {}  # NEW: Track selections per section
    
    with gr.Blocks(title="Quiz Generator Review", theme=gr.themes.Soft()) as demo:
        progress_bar = gr.Markdown(f"### Section 1 of {total_sections}")
        
        with gr.Row():
            gr.Markdown("# Review Generated Quizzes")
        gr.Markdown(f"Reviewing sections from `{filepath}`. Use the navigation buttons to move between sections. Click 'Save Current Section' to store your selections.")
        
        current_index = gr.State(0)
        
        section_title_box = gr.Textbox(label="Section Title", interactive=False)
        section_text_box = gr.Textbox(label="Section Content", lines=10, interactive=False)
        
        # Add rationale textbox
        rationale_box = gr.Textbox(
            label="Generation Rationale",
            interactive=False,
            lines=4,
            show_copy_button=True
        )
        
        qa_checkboxes = gr.CheckboxGroup(label="Select Q&A pairs to include", interactive=True)
        
        with gr.Row():
            back_btn = gr.Button("← Previous Section", variant="secondary")
            save_btn = gr.Button("💾 Save Current Section", variant="secondary")
            next_btn = gr.Button("Next Section →", variant="primary")
        
        with gr.Row():
            custom_guidance = gr.Textbox(
                label="Custom Regeneration Guidance (optional)",
                placeholder="E.g., 'Focus more on system design tradeoffs' or 'Include a question about scalability'",
                lines=2
            )
        
        with gr.Row():
            regenerate_btn = gr.Button("🔄 Regenerate Questions", variant="secondary")
            write_exit_btn = gr.Button("📝 Write to File and Exit", variant="stop")
        
        def get_section_data(index):
            if index >= total_sections:
                logging.info("All sections have been reviewed by the user.")
                progress_text = "### Review Complete! This is the last section."
                return progress_text, "All sections reviewed.", "", "", gr.update(choices=[], value=[]), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            
            progress_text = f"### Section {index + 1} of {total_sections}"
            title, text = sections[index]
            logging.info(f"Displaying section {index+1}/{total_sections} in GUI: {title}")
            
            qa_pairs = qa_by_section.get(title, [])
            
            # Extract rationale and questions
            rationale_text = ""
            choices = []
            
            if qa_pairs:
                # If quiz_needed is explicitly False, show only rationale, no Q&A pairs
                if isinstance(qa_pairs, dict) and qa_pairs.get("quiz_needed", True) is False:
                    rationale = qa_pairs.get("rationale", "No quiz needed for this section.")
                    rationale_text = str(rationale)
                    choices = []
                else:
                    if isinstance(qa_pairs, dict):
                        if "rationale" in qa_pairs:
                            rationale = qa_pairs["rationale"]
                            if isinstance(rationale, dict):
                                rationale_text = (
                                    f"Focus Areas: {', '.join(rationale['focus_areas'])}\n"
                                    f"Strategy: {rationale['question_strategy']}\n"
                                    f"Progression: {rationale['difficulty_progression']}\n"
                                    f"Integration: {rationale['integration']}\n"
                                    f"Ranking: {rationale.get('ranking_explanation', 'Questions ordered by learning effectiveness')}"
                                )
                            else:
                                rationale_text = str(rationale)
                        
                        # Add questions
                        questions = qa_pairs.get("questions", qa_pairs)
                        if isinstance(questions, list):
                            for i, qa in enumerate(questions):
                                if isinstance(qa, dict):
                                    formatted_content = f"{i+1}. {qa['question']}\n\n{qa['answer']}"
                                    if 'learning_objective' in qa:
                                        formatted_content += f"\n\nLearning Objective: {qa['learning_objective']}"
                                    choices.append((formatted_content, i))
                                else:
                                    # Handle case where qa is a string
                                    choices.append((f"{i+1}. {qa}", i))
                        else:
                            # Handle case where questions is a string
                            choices.append((f"1. {questions}", 0))
                    else:
                        # Handle case where qa_pairs is a list
                        for i, qa in enumerate(qa_pairs):
                            if isinstance(qa, dict):
                                formatted_content = f"{i+1}. {qa['question']}\n\n{qa['answer']}"
                                if 'learning_objective' in qa:
                                    formatted_content += f"\n\nLearning Objective: {qa['learning_objective']}"
                                choices.append((formatted_content, i))
                            else:
                                # Handle case where qa is a string
                                choices.append((f"{i+1}. {qa}", i))
            
            # Restore previous selection if available
            if title in selected_indices_by_section:
                initial_value = selected_indices_by_section[title]
            else:
                initial_value = list(range(len(choices))) if pre_select_all else []
            
            return progress_text, title, text, rationale_text, gr.update(choices=choices, value=initial_value), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

        def regenerate_questions(index, custom_guidance_text):
            if index >= total_sections:
                return gr.update(choices=[], value=[]), gr.update(value=""), gr.update(value="⚠️ No more sections to process!")
            
            title, text = sections[index]
            logging.info(f"Regenerating questions for section: '{title}'")
            
            # Get previous sections for context
            previous_sections = sections[:index] if index > 0 else None
            
            # Add custom guidance to the prompt if provided
            guidance_context = ""
            if custom_guidance_text and custom_guidance_text.strip():
                guidance_context = f"\n\nAdditional guidance for question generation:\n{custom_guidance_text.strip()}"
            
            user_prompt = build_user_prompt(title, text, chapter_title, previous_sections) + guidance_context
            
            # Show loading state
            yield gr.update(choices=[("Generating new questions...", 0)], value=[]), gr.update(value="Generating..."), gr.update(value="🔄 Generating...")
            
            try:
                qa_pairs = call_openai(client, SYSTEM_PROMPT, user_prompt, model=model)
                
                if qa_pairs:
                    # Check if quiz is needed
                    if isinstance(qa_pairs, dict) and not qa_pairs.get("quiz_needed", True):
                        qa_by_section[title] = qa_pairs
                        yield gr.update(choices=[], value=[]), gr.update(value=str(qa_pairs.get("rationale", "No quiz needed for this section."))), gr.update(value="✅ Section marked as not needing a quiz")
                    else:
                        qa_by_section[title] = qa_pairs
                        questions = qa_pairs.get("questions", qa_pairs) if isinstance(qa_pairs, dict) else qa_pairs
                        logging.info(f"Successfully regenerated {len(questions)} Q&A pairs for section: '{title}'")
                        
                        # Extract rationale and questions
                        rationale_text = ""
                        choices = []
                        
                        if isinstance(qa_pairs, dict) and "rationale" in qa_pairs:
                            rationale = qa_pairs["rationale"]
                            if isinstance(rationale, dict):
                                rationale_text = (
                                    f"Focus Areas: {', '.join(rationale['focus_areas'])}\n"
                                    f"Strategy: {rationale['question_strategy']}\n"
                                    f"Progression: {rationale['difficulty_progression']}\n"
                                    f"Integration: {rationale['integration']}\n"
                                    f"Ranking: {rationale.get('ranking_explanation', 'Questions ordered by learning effectiveness')}"
                                )
                            else:
                                rationale_text = str(rationale)
                        
                        # Add questions
                        for i, qa in enumerate(questions):
                            formatted_content = f"{i+1}. {qa['question']}\n\n{qa['answer']}"
                            if 'learning_objective' in qa:
                                formatted_content += f"\n\nLearning Objective: {qa['learning_objective']}"
                            choices.append((formatted_content, i))
                        
                        # Start with no questions selected
                        yield gr.update(choices=choices, value=[]), gr.update(value=rationale_text), gr.update(value="✅ Questions regenerated!")
                else:
                    yield gr.update(choices=[], value=[]), gr.update(value="Failed to generate questions"), gr.update(value="⚠️ Failed to generate questions")
            except Exception as e:
                logging.error(f"Error regenerating questions: {e}")
                yield gr.update(choices=[], value=[]), gr.update(value="Error generating questions"), gr.update(value="⚠️ Error generating questions")

        def go_next(index, selected_indices):
            # Save current section before moving
            save_current_section(index, selected_indices)
            
            if index >= total_sections - 1:
                # We're at the end, show end message
                progress_text = "### You've reached the end! There are no more sections to review."
                return index, progress_text, "End of Document", "No more sections to review.", "", gr.update(choices=[], value=[]), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
            
            new_index = index + 1
            title, text = sections[new_index]
            
            # Generate questions for the new section if they don't exist
            if title not in qa_by_section:
                logging.info(f"Generating questions for section: '{title}'")
                # Get previous sections for context
                previous_sections = sections[:new_index] if new_index > 0 else None
                user_prompt = build_user_prompt(title, text, chapter_title, previous_sections)
                qa_pairs = call_openai(client, SYSTEM_PROMPT, user_prompt, model=model)
                if qa_pairs:
                    # Check if quiz is needed
                    if isinstance(qa_pairs, dict) and not qa_pairs.get("quiz_needed", True):
                        logging.info(f"No quiz needed for section: '{title}'")
                        qa_by_section[title] = qa_pairs
                    else:
                        questions = qa_pairs.get("questions", qa_pairs) if isinstance(qa_pairs, dict) else qa_pairs
                        logging.info(f"Successfully generated {len(questions)} Q&A pairs for section: '{title}'")
                        qa_by_section[title] = qa_pairs
                else:
                    logging.warning(f"No Q&A pairs were generated for section: '{title}'")
            
            return new_index, *get_section_data(new_index)

        def go_back(index):
            if index > 0:
                new_index = index - 1
                return new_index, *get_section_data(new_index)
            return index, *get_section_data(index)

        def save_current_section(index, selected_indices):
            if index < total_sections:
                title, _ = sections[index]
                qa_pairs = qa_by_section.get(title, [])
                
                # Save the user's selection for this section
                selected_indices_by_section[title] = list(selected_indices)
                
                # Check if quiz is needed
                if isinstance(qa_pairs, dict) and not qa_pairs.get("quiz_needed", True):
                    # For sections that don't need a quiz, save an empty list
                    for i, (existing_title, _) in enumerate(final_answers):
                        if existing_title == title:
                            final_answers[i] = (title, [])
                            break
                    else:
                        final_answers.append((title, []))
                    logging.info(f"Section '{title}' marked as not needing a quiz")
                    return gr.update(value="✅ Section marked as not needing a quiz")
                
                # Get the questions list from the nested structure
                questions = qa_pairs.get("questions", qa_pairs) if isinstance(qa_pairs, dict) else qa_pairs
                
                # Filter out any invalid indices
                valid_indices = [i for i in selected_indices if i < len(questions)]
                selected_qa = [questions[i] for i in valid_indices]
                
                if selected_qa:
                    # Update or add to final_answers
                    for i, (existing_title, _) in enumerate(final_answers):
                        if existing_title == title:
                            final_answers[i] = (title, selected_qa)
                            break
                    else:
                        final_answers.append((title, selected_qa))
                    
                    logging.info(f"Saved {len(selected_qa)} Q&A pairs for section: {title}")
                    return gr.update(value="✅ Saved!")
                else:
                    return gr.update(value="⚠️ No selections to save")
            return gr.update()

        def write_and_exit():
            # print("--- ENTERING WRITE_AND_EXIT FUNCTION ---") # DEBUG: Check if function is called
            if not final_answers:
                return gr.update(value="⚠️ No sections saved!")
            
            logging.info(f"Proceeding to write {len(final_answers)} updated sections to file.")

            # Start with the cleaned content that was written to tmp_path initially
            # We need to read it back because 'content' is not directly accessible here
            with open(tmp_path, "r", encoding="utf-8") as f:
                modified_content = f.read()

            answer_blocks = []
            for title, qa_pairs in final_answers:
                logging.debug(f"Formatting and inserting quiz for section: '{title}'")
                slug = clean_slug(title)
                quiz_block = format_quiz_block(qa_pairs, f"{ANSWER_ID_PREFIX}{slug}")
                answer_block = format_answer_block(slug, qa_pairs)
                
                # Insert quiz only at the end of the ## section, after all its content (including any ### or deeper)
                section_pattern = re.compile(rf"(^##\s+{re.escape(title)}\s*\n)(.*?)(?=^##\s|\Z)", re.DOTALL | re.MULTILINE)
                def insert_quiz_at_end(match):
                    section_text = match.group(0).rstrip('\n') # Remove trailing newlines to control spacing
                    # Remove any existing quiz callout in this section
                    section_text = re.sub(r"::: \{\.callout-important title=\"Self-Check Quiz\"[\s\S]*?:::\n?", "", section_text)
                    # Only insert if quiz_block is not empty and not already present
                    if quiz_block.strip() and quiz_block.strip() not in section_text:
                        return section_text + quiz_block # Removed extra '\n'
                    else:
                        # If quiz_block is empty or already present, just return the cleaned content (with two newlines)
                        return section_text
                modified_content = section_pattern.sub(insert_quiz_at_end, modified_content, count=1)
                answer_blocks.append(answer_block)

            # Only add non-empty answer blocks
            nonempty_answer_blocks = [b for b in answer_blocks if b.strip() and not b.strip().isspace() and not b.strip().startswith('::: {.callout-tip') or (b.strip() and 'answer-' in b)]
            if nonempty_answer_blocks:
                logging.info("Appending final 'Quiz Answers' block to the document.")
                if not re.search(r"^##\s+Quiz Answers", modified_content, re.MULTILINE):
                    modified_content += "\n\n## Quiz Answers\n"
                modified_content += "\n" + "\n\n".join(nonempty_answer_blocks)

            # Write the final content back to the temporary file
            logging.info(f"Writing final content to temporary file: {tmp_path}")
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(modified_content)
            logging.info(f"Successfully wrote updated content to temporary file. Size: {os.path.getsize(tmp_path)} bytes")

            # --- Final Overwrite and Cleanup ---
            logging.info(f"Overwriting original file: {filepath}")
            shutil.copy2(tmp_path, filepath)
            logging.info(f"Successfully updated original file. New size: {os.path.getsize(filepath)} bytes")
            
            # Clean up the temporary file
            if os.path.exists(tmp_path):
                logging.info(f"Cleaning up temporary file: {tmp_path}")
                os.unlink(tmp_path)
                logging.info("Temporary file removed successfully.")

            logging.info(f"GUI mode: Successfully updated file: {filepath}")
            
            # --- Exit the app and process ---
            try:
                demo.close()
            except Exception:
                pass
            sys.exit(0)
            return gr.update(value="✅ File written successfully!")

        # Initial Load for the first section
        progress, title, text, rationale_text, checkboxes_update, next_visible, back_visible, regen_visible = get_section_data(0)
        progress_bar.value = progress
        section_title_box.value = title
        section_text_box.value = text
        rationale_box.value = rationale_text
        qa_checkboxes.choices = checkboxes_update['choices']
        qa_checkboxes.value = checkboxes_update['value']
        
        # Connect buttons
        back_btn.click(
            fn=go_back,
            inputs=[current_index],
            outputs=[current_index, progress_bar, section_title_box, section_text_box, rationale_box, qa_checkboxes, next_btn, back_btn, regenerate_btn]
        )
        
        save_btn.click(
            fn=save_current_section,
            inputs=[current_index, qa_checkboxes],
            outputs=[save_btn]
        )
        
        next_btn.click(
            fn=go_next,
            inputs=[current_index, qa_checkboxes],
            outputs=[current_index, progress_bar, section_title_box, section_text_box, rationale_box, qa_checkboxes, next_btn, back_btn, regenerate_btn]
        )
        
        regenerate_btn.click(
            fn=regenerate_questions,
            inputs=[current_index, custom_guidance],
            outputs=[qa_checkboxes, rationale_box, regenerate_btn]
        )
        
        write_exit_btn.click(
            fn=write_and_exit,
            outputs=[write_exit_btn]
        )

    # --- Interactive Prompt Before Launch ---
    print("\n" + "="*70)
    print("🚀 Interactive Review Session is Ready to Launch")
    print("="*70)
    print("The script will now start a local web server for you to review the questions.")
    print("\nWhat will happen next:")
    print("  1. I will start the server and attempt to open a new tab in your web browser.")
    print("  2. If a tab doesn't open, a URL (like http://127.0.0.1:7860) will appear below.")
    print("     You must copy and paste this URL into your browser.")
    print("  3. After you finish your review, click 'Write to File and Exit' and then close the web page.")
    print("     The script will then automatically process your selections and save the file.")
    
    input("\n➡️  Press Enter to launch the review session...")
    
    logging.info("Launching Gradio interface. Look for the local URL in the output below.")
    demo.launch(inbrowser=True)

# --- Main File Processing Logic ---

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
        r":::\s*\{[^}]*?" + re.escape(QUIZ_CALLOUT_CLASS) + r"[^}]*?\}[\s\S]*?:::\s*\n?",
        re.DOTALL | re.IGNORECASE
    )
    cleaned, quiz_removed_count = quiz_callout_pattern.subn("", markdown_text)

    # --- Remove all answer callouts ---
    answer_callout_pattern = re.compile(
        r":::\s*\{[^}]*?" + re.escape(ANSWER_CALLOUT_CLASS) + r"[^}]*?\}[\s\S]*?:::\s*\n?",
        re.DOTALL | re.IGNORECASE
    )
    cleaned, answer_removed_count = answer_callout_pattern.subn("", cleaned)

    # --- Remove the entire '## Quiz Answers' section (header + all content) ---
    quiz_answers_section_pattern = re.compile(
        r"(^##\s+" + re.escape("Quiz Answers") + r"[\s\S]*?)(?=^##\s|\Z)", re.MULTILINE
    )
    cleaned, section_removed_count = quiz_answers_section_pattern.subn("", cleaned)

    logging.debug(f"Cleaned section removed count: {section_removed_count}")
    changed = len(cleaned) != original_len
    return cleaned, changed, quiz_removed_count, answer_removed_count

def process_file(client, filepath, mode="batch", model="gpt-4o"):
    """Orchestrates the processing of a single markdown file."""
    logging.info(f"--- Starting processing for: {filepath} ---")
    tmp_path = None  # Initialize tmp_path
    try:
        # Read original file
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        logging.info(f"Successfully read file with {len(content)} characters.")

        # Clean up any existing quiz/answer callouts
        content, cleaned_something, quiz_count, answer_count = clean_existing_quiz_blocks(content)
        if cleaned_something:
            logging.info(f"Cleaned up existing content: Removed {quiz_count} quiz callout(s) and {answer_count} answer callout(s).")
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.tmp', encoding='utf-8') as tmp:
            tmp_path = tmp.name
            tmp.write(content)
        logging.info(f"Created temporary file at: {tmp_path}")
        logging.info(f"Temporary file size: {os.path.getsize(tmp_path)} bytes")

        # Now process the cleaned content
        chapter_title, sections = extract_sections(content)
        if not sections:
            logging.warning(f"No level-2 sections (## Section Title) found. Nothing to process.")
            logging.info(f"--- Finished processing for: {filepath} ---")
            return
            
        qa_by_section = {}
                
        if mode == "review":
            # For review mode, generate questions for all sections first
            logging.info("Generating questions for all sections...")
            for title, section_text in sections:
                logging.info(f"Processing section: '{title}'")
                user_prompt = build_user_prompt(title, section_text, chapter_title)
                qa_pairs = call_openai(client, SYSTEM_PROMPT, user_prompt, model=model)
                if qa_pairs:
                    questions = qa_pairs.get("questions", qa_pairs) if isinstance(qa_pairs, dict) else qa_pairs
                    logging.info(f"Successfully generated {len(questions)} Q&A pairs for section: '{title}'")
                    qa_by_section[title] = qa_pairs
                else:
                    logging.warning(f"No Q&A pairs were generated for section: '{title}'")
            
            # Launch GUI with all questions pre-selected
            launch_gui_mode(client, sections, qa_by_section, filepath, model, pre_select_all=True, tmp_path=tmp_path, chapter_title=chapter_title)
        
        elif mode == "interactive":
            # For interactive mode, only generate questions for the first section initially
            if sections:
                title, text = sections[0]
                logging.info(f"Generating initial questions for first section: '{title}'")
                user_prompt = build_user_prompt(title, text, chapter_title)
                qa_pairs = call_openai(client, SYSTEM_PROMPT, user_prompt, model=model)
                if qa_pairs:
                    qa_by_section[title] = qa_pairs
                    questions = qa_pairs.get("questions", qa_pairs) if isinstance(qa_pairs, dict) else qa_pairs
                    logging.info(f"Successfully generated {len(questions)} Q&A pairs for first section")
                else:
                    logging.warning(f"No Q&A pairs were generated for first section")
            
            # Launch GUI with only first section's questions
            launch_gui_mode(client, sections, qa_by_section, filepath, model, tmp_path=tmp_path, chapter_title=chapter_title)
        
        else:  # batch mode
            # For batch mode, process all sections and write to file
            logging.info("Running in batch mode. Writing all generated quizzes to file...")
            for title, section_text in sections:
                logging.info(f"Processing section: '{title}'")
                user_prompt = build_user_prompt(title, section_text, chapter_title)
                qa_pairs = call_openai(client, SYSTEM_PROMPT, user_prompt, model=model)
                if qa_pairs:
                    questions = qa_pairs.get("questions", qa_pairs) if isinstance(qa_pairs, dict) else qa_pairs
                    logging.info(f"Successfully generated {len(questions)} Q&A pairs for section: '{title}'")
                    qa_by_section[title] = qa_pairs
                else:
                    logging.warning(f"No Q&A pairs were generated for section: '{title}'")

            if not qa_by_section:
                logging.warning("No questions were generated for any section. No changes will be made.")
                logging.info(f"--- Finished processing for: {filepath} ---")
                return

            modified_content = content
            answer_blocks = []

            logging.info(f"Starting to insert {len(qa_by_section)} quiz sections into the document...")
            for title, qa_pairs in qa_by_section.items():
                logging.info(f"Processing quiz insertion for section: '{title}'")
                slug = clean_slug(title)
                quiz_block = format_quiz_block(qa_pairs, f"{ANSWER_ID_PREFIX}{slug}")
                answer_block = format_answer_block(slug, qa_pairs)
                
                if quiz_block.strip():
                    logging.info(f"Generated quiz block for section '{title}' ({len(quiz_block)} chars)")
                else:
                    logging.info(f"No quiz block generated for section '{title}' (quiz not needed)")
                
                if answer_block.strip():
                    logging.info(f"Generated answer block for section '{title}' ({len(answer_block)} chars)")
                    answer_blocks.append(answer_block)
                else:
                    logging.info(f"No answer block generated for section '{title}' (quiz not needed)")
                
                # Insert quiz only at the end of the ## section, after all its content (including any ### or deeper)
                section_pattern = re.compile(rf"(^##\s+{re.escape(title)}\s*\n)(.*?)(?=^##\s|\Z)", re.DOTALL | re.MULTILINE)
                def insert_quiz_at_end(match):
                    section_text = match.group(0).rstrip('\n') # Remove trailing newlines to control spacing
                    # Remove any existing quiz callout in this section
                    section_text = re.sub(r"::: \{\.callout-important title=\"Self-Check Quiz\"[\s\S]*?:::\n?", "", section_text)
                    # Only insert if quiz_block is not empty and not already present
                    if quiz_block.strip() and quiz_block.strip() not in section_text:
                        return section_text + quiz_block # Removed extra '\n'
                    else:
                        # If quiz_block is empty or already present, just return the cleaned content (with two newlines)
                        return section_text
                modified_content = section_pattern.sub(insert_quiz_at_end, modified_content, count=1)

            # Only add non-empty answer blocks
            nonempty_answer_blocks = [b for b in answer_blocks if b.strip() and not b.strip().isspace() and not b.strip().startswith('::: {.callout-tip') or (b.strip() and 'answer-' in b)]
            logging.info(f"Found {len(nonempty_answer_blocks)} non-empty answer blocks to append")
            
            if nonempty_answer_blocks:
                logging.info("Appending final 'Quiz Answers' block.")
                if not re.search(r"^##\s+Quiz Answers", modified_content, re.MULTILINE):
                    logging.info("Adding 'Quiz Answers' section header")
                    modified_content += "\n\n## Quiz Answers\n"
                else:
                    logging.info("'Quiz Answers' section header already exists")
                
                logging.info(f"Appending {len(nonempty_answer_blocks)} answer blocks")
                modified_content += "\n" + "\n\n".join(nonempty_answer_blocks)
            else:
                logging.info("No answer blocks to append")

            # Write the final content to the temporary file
            logging.info(f"Writing modified content to temporary file: {tmp_path}")
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(modified_content)
            logging.info(f"Successfully wrote updated content to temporary file. Size: {os.path.getsize(tmp_path)} bytes")

            # Overwrite the original file with the temporary file's contents
            logging.info(f"Overwriting original file: {filepath}")
            shutil.copy2(tmp_path, filepath)
            logging.info(f"Successfully updated original file. New size: {os.path.getsize(filepath)} bytes")

    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return
    except Exception as e:
        logging.error(f"An error occurred while processing {filepath}: {str(e)}")
        return
    finally:
        # Clean up the temporary file if it still exists
        if tmp_path and os.path.exists(tmp_path):
            # This block will be simplified. The temporary file will be handled by GUI mode.
            # For batch mode, it's already handled, so this is primarily for error cleanup if GUI didn't launch.
            logging.info(f"Cleaning up leftover temporary file: {tmp_path}")
            os.unlink(tmp_path)
            logging.info("Leftover temporary file removed successfully.")

def main():
    """Parses command-line arguments and starts the processing."""
    parser = argparse.ArgumentParser(
        description="Generate self-check quizzes from Quarto markdown files using OpenAI.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-f", "--file", help="Path to a single markdown (.qmd/.md) file.")
    parser.add_argument("-d", "--dir", help="Path to a directory containing markdown files to process.")
    parser.add_argument("--mode", required=True, choices=["batch", "review", "interactive"],
        help="""Mode of operation:
        batch: Process all sections and write to file without review
        review: Batch process all sections first, then review them
        interactive: Generate and review questions one section at a time""")
    parser.add_argument("--model", default="gpt-4o", help="The OpenAI model to use (e.g., 'gpt-4o', 'gpt-4o-mini').")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose (DEBUG) logging.")
    
    args = parser.parse_args()

    # --- Setup Logging ---
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logging.info("Script starting...")
    logging.info(f"Running with settings: {args}")

    # --- Initialize OpenAI Client ---
    try:
        client = OpenAI()
        logging.debug("OpenAI client initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client. Ensure OPENAI_API_KEY is set. Error: {e}")
        return

    if args.file:
        process_file(client, Path(args.file), mode=args.mode, model=args.model)
    elif args.dir:
        directory = Path(args.dir)
        if not directory.is_dir():
            logging.error(f"Error: Provided path is not a directory: {args.dir}")
            return
            
        logging.info(f"Scanning directory: {directory}")
        for root, _, files in os.walk(directory):
            for name in files:
                if name.endswith((".qmd", ".md")):
                    filepath = Path(root) / name
                    process_file(client, filepath, mode=args.mode, model=args.model)
                else:
                    logging.debug(f"Skipping non-markdown file: {name}")
    else:
        parser.print_help()
    
    logging.info("Script finished.")

if __name__ == "__main__":
    main()