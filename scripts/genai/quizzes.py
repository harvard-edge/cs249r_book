import argparse
import os
import re
import logging
import json
from pathlib import Path
from openai import OpenAI, APIError
from slugify import slugify
import gradio as gr

# Client is initialized in main() after logging is set up

# --- Constants ---
STOPWORDS = {"in", "of", "the", "and", "to", "for", "on", "a", "an", "with", "by"}

SYSTEM_PROMPT = """
You are a helpful assistant for a university-level textbook on machine learning systems.
Your task is to generate 1 to 5 self-check questions and answers based on the section provided.
These questions should help students and learners reflect on and test their understanding.
These are self-check quizzes, not homework assignments to pause and reflect.
The questions should be appropriate for a university-level course.

Guidelines:
- Focus on conceptual understanding, system-level reasoning, and meaningful tradeoffs.
- Avoid surface-level recall or trivia.
- Use clear, academically appropriate language.
- Include at least one question about a system design tradeoff or implication.
- If applicable, include a question that addresses a common misconception.
- Do not repeat exact phrasing from the source text.
- Keep in mind that the questions should ideally be focused on machine learning systems, not just machine learning.

Question Types (use a mix based on what best tests understanding):
- Multiple Choice Questions (MCQ) with clear, unambiguous options
- Short answer explanations requiring conceptual understanding
- Scenario-based questions that test application of concepts
- "Why" questions that probe deeper understanding
- "How" questions that test practical knowledge
- Comparison questions that test understanding of tradeoffs
- Fill-in-the-blank for key concepts
- True/False with justification required
Choose the question type that will most effectively test understanding of each concept.

Bloom's Taxonomy Levels (aim for a mix):
- Remember: Basic recall of facts, terms, concepts
- Understand: Explain ideas or concepts in your own words
- Apply: Use information in new situations
- Analyze: Draw connections among ideas
- Evaluate: Justify a stand or decision
- Create: Produce new or original work

Ranking Guidelines:
- Rank questions by their learning effectiveness, with the most impactful questions first
- Prioritize questions that test deep understanding over surface knowledge
- Place system design and tradeoff questions at the top
- Put basic recall or definition questions last
- Consider which questions will most effectively reinforce key concepts
- Order questions to build understanding progressively
- Ensure each question adds unique value to the learning experience

Progressive Improvement:
- Review any previously asked questions provided in the prompt
- Ensure your new questions are distinct from previous ones
- Build upon concepts covered in earlier sections
- Progress from basic understanding to more complex applications
- Later questions should integrate multiple concepts from earlier sections

Return the output in valid JSON format with the following structure:
{
    "rationale": {
        "focus_areas": ["List of 2-3 key areas this section focuses on"],
        "question_strategy": "Brief explanation of why these question types were chosen and how they build understanding",
        "difficulty_progression": "How the questions progress in complexity and learning effectiveness",
        "integration": "How these questions connect with previous sections (if applicable)",
        "ranking_explanation": "Explanation of why questions are ordered this way and how they support learning"
    },
    "questions": [
        {
            "question": "The question text",
            "answer": "The answer text",
            "learning_objective": "What specific understanding or skill this question tests"
        }
    ]
}

For MCQs, format the answer as "Correct option: [letter]. [explanation]"
For True/False, format as "Answer: [True/False]. [explanation]"
For other types, provide a clear, concise answer that demonstrates understanding.
"""

# --- Core Functions ---

def clean_slug(title):
    """Creates a URL-friendly slug from a title."""
    words = title.lower().split()
    keywords = [w for w in words if w not in STOPWORDS]
    slug = slugify(" ".join(keywords))
    logging.debug(f"Cleaned title '{title}' to slug '{slug}'")
    return slug

def build_user_prompt(section_title, section_text, chapter_title=None, previous_sections=None):
    """Constructs the user prompt for the language model."""
    prefix = f"This section is titled \"{section_title}\"."
    if chapter_title:
        prefix += f" It appears in the chapter \"{chapter_title}\"."
    
    previous_context = ""
    if previous_sections:
        previous_context = "\n\nPrevious sections in this chapter:\n"
        for prev_title, prev_content in previous_sections:
            # Extract any existing quizzes from previous content
            quiz_pattern = re.compile(r":::.*?title=\"Self-Check Quiz\".*?:::", re.DOTALL)
            quiz_match = quiz_pattern.search(prev_content)
            quiz_text = quiz_match.group(0) if quiz_match else "No quiz in this section"
            
            previous_context += f"\nSection: {prev_title}\n"
            previous_context += f"Content Summary: {prev_content[:200]}...\n"
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
    
    # Extract sections
    pattern = re.compile(r"^##\s+(.*)", re.MULTILINE)
    matches = list(pattern.finditer(markdown_text))
    sections = []
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        content = markdown_text[start:end].strip()
        sections.append((title, content))
        logging.debug(f"  - Extracted section: '{title}'")
    logging.info(f"Extraction complete. Found {len(sections)} sections.")
    return chapter_title, sections

def call_openai(client, system_prompt, user_prompt, model="gpt-4"):
    """Calls the OpenAI API and handles potential errors."""
    logging.info(f"Calling OpenAI API with model '{model}'...")
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
        logging.debug(f"Raw response content from OpenAI: {content}")
        
        data = json.loads(content)
        
        # Ensure we always return a dictionary with the expected structure
        if isinstance(data, list):
            # Convert list to expected dictionary format
            return {
                "rationale": {
                    "focus_areas": ["Key concepts from the section"],
                    "question_strategy": "Questions designed to test understanding of core concepts",
                    "difficulty_progression": "Questions progress from basic to advanced concepts",
                    "integration": "Questions build on fundamental concepts"
                },
                "questions": data
            }
        elif isinstance(data, dict):
            # If it's already a dict but missing rationale, add default rationale
            if "rationale" not in data:
                data["rationale"] = {
                    "focus_areas": ["Key concepts from the section"],
                    "question_strategy": "Questions designed to test understanding of core concepts",
                    "difficulty_progression": "Questions progress from basic to advanced concepts",
                    "integration": "Questions build on fundamental concepts"
                }
            # If it's missing questions but has other fields, wrap them in questions
            if "questions" not in data:
                questions = []
                for key, value in data.items():
                    if key != "rationale":
                        if isinstance(value, list):
                            questions.extend(value)
                        else:
                            questions.append(value)
                data["questions"] = questions
            return data
        else:
            logging.warning(f"Unexpected JSON structure from OpenAI: {type(data)}")
            return {
                "rationale": {
                    "focus_areas": ["Key concepts from the section"],
                    "question_strategy": "Questions designed to test understanding of core concepts",
                    "difficulty_progression": "Questions progress from basic to advanced concepts",
                    "integration": "Questions build on fundamental concepts"
                },
                "questions": []
            }

    except APIError as e:
        logging.error(f"OpenAI API error: {e}")
        return {
            "rationale": {
                "focus_areas": ["Error occurred"],
                "question_strategy": "Failed to generate questions",
                "difficulty_progression": "N/A",
                "integration": "N/A"
            },
            "questions": []
        }
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse OpenAI JSON response: {e}")
        logging.debug(f"Problematic raw content: {content}")
        return {
            "rationale": {
                "focus_areas": ["Error occurred"],
                "question_strategy": "Failed to parse response",
                "difficulty_progression": "N/A",
                "integration": "N/A"
            },
            "questions": []
        }
    except Exception as e:
        logging.error(f"An unexpected error occurred in call_openai: {e}")
        return {
            "rationale": {
                "focus_areas": ["Error occurred"],
                "question_strategy": "Unexpected error",
                "difficulty_progression": "N/A",
                "integration": "N/A"
            },
            "questions": []
        }

def format_quiz_block(qa_pairs, answer_ref):
    """Formats the questions into a Quarto callout block."""
    # Extract questions from the nested structure
    questions = qa_pairs.get("questions", qa_pairs) if isinstance(qa_pairs, dict) else qa_pairs
    
    # Format questions without learning objectives
    formatted_questions = [f"{i+1}. {qa['question']}" for i, qa in enumerate(questions)]
    
    return f"""::: {{.callout-important title="Self-Check Quiz" collapse="true"}}

{"\n\n".join(formatted_questions)}

See @{answer_ref}.
:::

"""

def format_answer_block(slug, qa_pairs):
    """Formats the answers into a Quarto callout block."""
    # Extract questions from the nested structure
    questions = qa_pairs.get("questions", qa_pairs) if isinstance(qa_pairs, dict) else qa_pairs
    
    # Format Q&A pairs without learning objectives
    lines = [f"**{i+1}. {qa['question']}**\n\n{qa['answer']}" for i, qa in enumerate(questions)]
    
    return f"""::: {{.callout-tip #answer-{slug} title="Quiz Answers" collapse="true"}}

{"\n\n".join(lines)}
:::

"""

# --- GUI Mode ---

def launch_gui_mode(client, sections, qa_by_section, filepath, model):
    """Launches an interactive Gradio GUI to review and select Q&A pairs."""
    logging.info("Preparing to launch GUI mode...")
    final_answers = []
    total_sections = len(sections)
    
    # Extract chapter title from the first section if available
    chapter_title = None
    if sections:
        # Try to find chapter title in the first section's content
        first_section_text = sections[0][1]
        chapter_pattern = re.compile(r"^#\s+(.*)", re.MULTILINE)
        chapter_match = chapter_pattern.search(first_section_text)
        if chapter_match:
            chapter_title = chapter_match.group(1).strip()
            logging.info(f"Found chapter title: {chapter_title}")

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
                if isinstance(qa_pairs, dict) and "rationale" in qa_pairs:
                    rationale = qa_pairs["rationale"]
                    rationale_text = (
                        f"Focus Areas: {', '.join(rationale['focus_areas'])}\n"
                        f"Strategy: {rationale['question_strategy']}\n"
                        f"Progression: {rationale['difficulty_progression']}\n"
                        f"Integration: {rationale['integration']}\n"
                        f"Ranking: {rationale.get('ranking_explanation', 'Questions ordered by learning effectiveness')}"
                    )
                
                # Add questions
                questions = qa_pairs.get("questions", qa_pairs) if isinstance(qa_pairs, dict) else qa_pairs
                for i, qa in enumerate(questions):
                    formatted_content = f"{i+1}. {qa['question']}\n\n{qa['answer']}"
                    if 'learning_objective' in qa:
                        formatted_content += f"\n\nLearning Objective: {qa['learning_objective']}"
                    choices.append((formatted_content, i))
            
            # Start with no questions selected
            return progress_text, title, text, rationale_text, gr.update(choices=choices, value=[]), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

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
                    qa_by_section[title] = qa_pairs
                    questions = qa_pairs.get("questions", qa_pairs) if isinstance(qa_pairs, dict) else qa_pairs
                    logging.info(f"Successfully regenerated {len(questions)} Q&A pairs for section: '{title}'")
                    
                    # Extract rationale and questions
                    rationale_text = ""
                    choices = []
                    
                    if isinstance(qa_pairs, dict) and "rationale" in qa_pairs:
                        rationale = qa_pairs["rationale"]
                        rationale_text = (
                            f"Focus Areas: {', '.join(rationale['focus_areas'])}\n"
                            f"Strategy: {rationale['question_strategy']}\n"
                            f"Progression: {rationale['difficulty_progression']}\n"
                            f"Integration: {rationale['integration']}\n"
                            f"Ranking: {rationale.get('ranking_explanation', 'Questions ordered by learning effectiveness')}"
                        )
                    
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
                    qa_by_section[title] = qa_pairs
                    questions = qa_pairs.get("questions", qa_pairs) if isinstance(qa_pairs, dict) else qa_pairs
                    logging.info(f"Successfully generated {len(questions)} Q&A pairs for section: '{title}'")
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
            if not final_answers:
                return gr.update(value="⚠️ No sections saved!")
            
            logging.info(f"Proceeding to write {len(final_answers)} updated sections to file.")
            with open(filepath, "r", encoding="utf-8") as f:
                modified_content = f.read()

            answer_blocks = []
            for title, qa_pairs in final_answers:
                logging.debug(f"Formatting and inserting quiz for section: '{title}'")
                slug = clean_slug(title)
                quiz_block = format_quiz_block(qa_pairs, f"answer-{slug}")
                answer_block = format_answer_block(slug, qa_pairs)
                
                section_pattern = re.compile(rf"(^##\s+{re.escape(title)}\s*\n)(.*?)(?=^##|\Z)", re.DOTALL | re.MULTILINE)
                modified_content = section_pattern.sub(lambda m: m.group(0).rstrip() + "\n\n" + quiz_block + "\n", modified_content, count=1)
                answer_blocks.append(answer_block)

            if answer_blocks:
                logging.info("Appending final 'Quiz Answers' block to the document.")
                if not re.search(r"^##\s+Quiz Answers", modified_content, re.MULTILINE):
                     modified_content += "\n\n## Quiz Answers\n"
                modified_content += "\n" + "\n\n".join(answer_blocks)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(modified_content)
            logging.info(f"GUI mode: Successfully updated file: {filepath}")
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
            inputs=[],
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

def process_file(client, filepath, gui=False, model="gpt-4"):
    """Orchestrates the processing of a single markdown file."""
    logging.info(f"--- Starting processing for: {filepath} ---")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        logging.info(f"Successfully read file with {len(content)} characters.")
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return

    chapter_title, sections = extract_sections(content)
    if not sections:
        logging.warning(f"No level-2 sections (## Section Title) found. Nothing to process.")
        logging.info(f"--- Finished processing for: {filepath} ---")
        return
        
    qa_by_section = {}
    
    if gui:
        # For GUI mode, only generate questions for the first section initially
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
        
        launch_gui_mode(client, sections, qa_by_section, filepath, model)
    else:
        # For non-GUI mode, process all sections
        logging.info("Running in non-GUI mode. Writing all generated quizzes to file...")
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

        for title, qa_pairs in qa_by_section.items():
            logging.debug(f"Inserting quiz for section '{title}' into content.")
            slug = clean_slug(title)
            quiz_block = format_quiz_block(qa_pairs, f"answer-{slug}")
            answer_block = format_answer_block(slug, qa_pairs)
            
            section_pattern = re.compile(rf"(^##\s+{re.escape(title)}\s*\n)(.*?)(?=^##|\Z)", re.DOTALL | re.MULTILINE)
            modified_content = section_pattern.sub(lambda m: m.group(0).rstrip() + "\n\n" + quiz_block + "\n", modified_content, count=1)
            answer_blocks.append(answer_block)

        if answer_blocks:
            logging.info("Appending final 'Quiz Answers' block.")
            if not re.search(r"^##\s+Quiz Answers", modified_content, re.MULTILINE):
                 modified_content += "\n\n## Quiz Answers\n"
            modified_content += "\n" + "\n\n".join(answer_blocks)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(modified_content)
        logging.info(f"Non-GUI mode: Successfully updated file.")
    
    logging.info(f"--- Finished processing for: {filepath} ---")

def main():
    """Parses command-line arguments and starts the processing."""
    parser = argparse.ArgumentParser(
        description="Generate self-check quizzes from Quarto markdown files using OpenAI.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-f", "--file", help="Path to a single markdown (.qmd/.md) file.")
    parser.add_argument("-d", "--dir", help="Path to a directory containing markdown files to process.")
    parser.add_argument("--gui", action="store_true", help="Run in interactive GUI mode for reviewing questions.")
    parser.add_argument("--model", default="gpt-4o", help="The OpenAI model to use (e.g., 'gpt-4', 'gpt-4-turbo').")
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
        process_file(client, Path(args.file), gui=args.gui, model=args.model)
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
                    process_file(client, filepath, gui=args.gui, model=args.model)
                else:
                    logging.debug(f"Skipping non-markdown file: {name}")
    else:
        parser.print_help()
    
    logging.info("Script finished.")

if __name__ == "__main__":
    main()