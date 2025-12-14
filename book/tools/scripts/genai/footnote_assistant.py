import argparse
import time
import os
import json
import re
import gradio as gr
import logging

# Import client libraries
from openai import OpenAI
from groq import Groq

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("footnote_assistant.log")
    ]
)

# Initialize client based on command-line choice
client = None
api_provider = None
model_name = None

# --- Parse document and extract sections and headers ---
def parse_qmd_sections(text):
    logging.info("Parsing QMD sections")
    lines = text.splitlines()
    sections = []
    headers = []
    buffer = []
    found_header = False
    prologue = ""  # Define prologue variable

    for i, line in enumerate(lines):
        if re.match(r'^#+\s+', line.strip()):
            prologue = "\n".join(lines[:i]).strip()
            lines = lines[i:]
            break

    for line in lines:
        # Match headers with regex: #+ followed by space, then any text
        if re.match(r'^#+\s+', line.strip()):
            if found_header and buffer:
                joined = "\n".join(buffer)
                sections.append(joined)
                buffer = []
            found_header = True
        if found_header:
            buffer.append(line)

    if buffer:
        joined = "\n".join(buffer)
        sections.append(joined)

    # Extract headers for the outline
    for i, section in enumerate(sections):
        lines = section.split('\n')
        if lines:
            first_line = lines[0].strip()
            # Use regex to extract header level and text properly
            header_match = re.match(r'^(#+)\s+(.*?)$', first_line)
            if header_match:
                level = len(header_match.group(1))
                header_text = header_match.group(2).strip()
                headers.append({"text": header_text, "level": level, "index": i})
            else:
                # Fallback method if regex doesn't match
                level = 0
                for char in first_line:
                    if char == '#':
                        level += 1
                    else:
                        break
                header_text = first_line[level:].strip()
                headers.append({"text": header_text, "level": level, "index": i})

    logging.info(f"Found {len(sections)} sections")
    return sections, headers, prologue  # Return prologue as well

# --- Replace section text in full file ---
def replace_section(full_text, old, new):
    # If old text isn't found, log a warning
    if old not in full_text:
        logging.warning(f"Could not find section to replace. First 50 chars of section: {old[:50]}")
        return full_text

    # Otherwise replace it and return
    return full_text.replace(old, new)

# --- Get LLM footnote suggestions ---
def get_footnote_suggestions(section_text, prompt_template):
    logging.info(f"Getting footnote suggestions from {api_provider} LLM using model {model_name}")

    # Don't use .format() at all - just concatenate the text at the end
    if "{text}" in prompt_template:
        complete_prompt = prompt_template.replace("{text}", section_text)
    else:
        complete_prompt = prompt_template + "\n\nText to analyze:\n" + section_text

    # Save the prompt to a file for debugging
    with open("last_prompt_sent.txt", "w") as f:
        f.write(complete_prompt)

    messages = [
        {"role": "system", "content": "You are an academic footnote assistant. Your response must be valid JSON only."},
        {"role": "user", "content": complete_prompt}
    ]

    logging.info(f"Sending request to {api_provider} API")
    try:
        # Use the global client that was set at startup
        if api_provider.lower() == "openai":
            response = client.chat.completions.create(
                model=model_name,
                messages=messages
            )
        elif api_provider.lower() == "groq":
            response = client.chat.completions.create(
                model=model_name,
                messages=messages
            )
        else:
            raise ValueError(f"Unsupported API provider: {api_provider}")

        content = response.choices[0].message.content

        # Save the response to a file for debugging
        with open("last_api_response.txt", "w") as f:
            f.write(content)

        logging.info(f"Received response: {content[:100]}...")
        return content
    except Exception as e:
        logging.error(f"API error: {e}")
        return json.dumps({"footnotes": []})

def show_section_with_markers(section_text, all_footnotes):
    """Show the section with colored footnote markers for all possible footnotes"""

    if not all_footnotes:
        return section_text.replace("\n", "<br>")

    # First, identify paragraphs in the text
    paragraphs = re.split(r'\n\s*\n', section_text)
    modified_paragraphs = []

    # Escape HTML characters
    for para in paragraphs:
        escaped_para = para.replace("<", "&lt;").replace(">", "&gt;")

        # Add colored markers for footnotes
        for fn in all_footnotes:
            insert_after = fn["insert_after"]
            marker = fn["marker"]

            # Skip if insert_after is empty
            if not insert_after.strip():
                continue

            # Check if the phrase is at the beginning of the paragraph
            if escaped_para.strip().startswith(insert_after):
                # Insert marker directly after the phrase
                escaped_para = escaped_para.replace(
                    insert_after,
                    f'{insert_after}<span style="color: #2E86C1; font-weight: bold; background-color: #EBF5FB; border-radius: 3px; padding: 0 2px;">{marker}</span>',
                    1
                )
                continue

            # Check for punctuation following the phrase
            pattern = re.compile(rf"({re.escape(insert_after)})([.,;:!?])")
            match = pattern.search(escaped_para)

            if match:
                # Insert the colored marker before the punctuation
                escaped_para = pattern.sub(
                    rf'\1<span style="color: #2E86C1; font-weight: bold; background-color: #EBF5FB; border-radius: 3px; padding: 0 2px;">{marker}</span>\2',
                    escaped_para,
                    count=1
                )
            else:
                # No punctuation, just add colored marker directly after the phrase
                escaped = re.escape(insert_after)
                pattern = re.compile(rf"({escaped})(?![^\n]*\[\^)")
                escaped_para = pattern.sub(
                    rf'\1<span style="color: #2E86C1; font-weight: bold; background-color: #EBF5FB; border-radius: 3px; padding: 0 2px;">{marker}</span>',
                    escaped_para,
                    count=1
                )

        modified_paragraphs.append(escaped_para)

    # Join paragraphs with proper spacing
    html_text = "<br><br>".join(modified_paragraphs)

    # Replace remaining newlines with <br> tags
    html_text = html_text.replace("\n", "<br>")

    return html_text

# --- Show preview with colored footnote markers for selected footnotes ---
def show_preview_with_markers(section_text, selected_options, all_footnotes):
    """Show the preview with colored footnote markers for selected footnotes and per-paragraph footnotes"""

    # Extract the indices from the selected checkbox text
    selected_indices = []
    for option in selected_options:
        # Extract the number from the format "1. [^fn-xxx]: text"
        match = re.match(r'^(\d+)\.', option)
        if match:
            # Adjust for 1-based indexing in display vs 0-based in code
            idx = int(match.group(1)) - 1
            selected_indices.append(str(idx))

    if not all_footnotes or not selected_indices:
        return section_text.replace("\n", "<br>")

    # First, identify paragraphs in the text
    paragraphs = re.split(r'\n\s*\n', section_text)
    modified_paragraphs = []

    # Escape HTML characters and initialize tracking
    escaped_paragraphs = []
    paragraph_footnotes = [[] for _ in paragraphs]

    for i, para in enumerate(paragraphs):
        escaped_paragraphs.append(para.replace("<", "&lt;").replace(">", "&gt;"))

    # For each selected footnote
    for idx in selected_indices:
        try:
            idx_int = int(idx)
            fn = all_footnotes[idx_int]

            # Get the insert phrase
            insert_after = fn["insert_after"]
            marker = fn["marker"]

            # Find which paragraph contains this phrase
            found = False
            for i, paragraph in enumerate(paragraphs):
                if insert_after in paragraph:
                    # Get the modified paragraph text (already HTML-escaped)
                    modified_paragraph = escaped_paragraphs[i]

                    # Handle punctuation positioning
                    pattern = re.compile(rf"({re.escape(insert_after)})([.,;:!?])")
                    match = pattern.search(modified_paragraph)

                    if match:
                        # Use HTML for colored marker
                        modified_paragraph = pattern.sub(
                            rf'\1<span style="color: #2E86C1; font-weight: bold; background-color: #EBF5FB; border-radius: 3px; padding: 0 2px;">{marker}</span>\2',
                            modified_paragraph,
                            count=1
                        )
                    else:
                        # No punctuation, use HTML for colored marker
                        escaped = re.escape(insert_after)
                        pattern = re.compile(rf"({escaped})(?![^\n]*\[\^)")
                        modified_paragraph = pattern.sub(
                            rf'\1<span style="color: #2E86C1; font-weight: bold; background-color: #EBF5FB; border-radius: 3px; padding: 0 2px;">{marker}</span>',
                            modified_paragraph,
                            count=1
                        )

                    # Update the modified paragraph
                    escaped_paragraphs[i] = modified_paragraph

                    # Add footnote to this paragraph's collection
                    paragraph_footnotes[i].append(f"<span style='color: #2E86C1; font-weight: bold;'>{marker}</span>: {fn['footnote_text']}")

                    found = True
                    break

        except Exception as e:
            logging.error(f"Error applying footnote {idx}: {e}")

    # Assemble paragraphs with their footnotes
    for i, para in enumerate(escaped_paragraphs):
        if paragraph_footnotes[i]:
            footnote_html = "<div style='padding-left: 20px; margin-top: 10px; margin-bottom: 10px; border-left: 2px solid #ccc;'>"
            footnote_html += "<br>".join(paragraph_footnotes[i])
            footnote_html += "</div>"
            modified_paragraphs.append(f"{para}{footnote_html}")
        else:
            modified_paragraphs.append(para)

    # Join all paragraphs with proper spacing
    html_text = "<br><br>".join(modified_paragraphs)

    return html_text

def apply_footnotes(section_text, selected_options, all_footnotes, global_footnote_set):
    """
    Apply selected footnotes to the section, ensuring no duplicates across the document.
    IMPORTANT: Skips footnotes that would be inserted inside ::: div blocks.
    """

    selected_indices = [int(re.match(r'^(\d+)\.', opt).group(1)) - 1 for opt in selected_options if re.match(r'^(\d+)\.', opt)]
    if not all_footnotes or not selected_indices:
        return section_text

    paragraphs = re.split(r'\n\s*\n', section_text)
    modified_paragraphs = []
    paragraph_footnotes = [[] for _ in paragraphs]

    # Track which paragraphs are inside div blocks
    lines = section_text.split('\n')
    in_div_block = False
    div_paragraph_indices = set()

    current_para_idx = 0
    empty_line_count = 0

    for line in lines:
        # Track div blocks
        if line.strip().startswith(':::'):
            in_div_block = not in_div_block

        # Track paragraph transitions (double newline)
        if not line.strip():
            empty_line_count += 1
            if empty_line_count >= 1:  # Paragraph break
                current_para_idx += 1
                empty_line_count = 0
        else:
            empty_line_count = 0

        # Mark this paragraph as being in a div block
        if in_div_block:
            div_paragraph_indices.add(current_para_idx)

    for idx in selected_indices:
        try:
            fn = all_footnotes[idx]
            insert_after = fn["insert_after"]
            marker = fn["marker"]

            # Check if the footnote marker was already added before in the document
            if marker in global_footnote_set:
                logging.info(f"Skipping duplicate footnote marker: {marker}")
                continue  # Skip adding this marker again

            # Find and modify the paragraph where this phrase appears
            found = False
            for i, paragraph in enumerate(paragraphs):
                if insert_after in paragraph:
                    # Check if this paragraph is inside a div block
                    if i in div_paragraph_indices:
                        logging.warning(f"Skipping footnote '{marker}' - would be inserted inside div block (paragraph {i+1})")
                        continue

                    modified_paragraph = paragraph

                    # Handle punctuation positioning
                    pattern = re.compile(rf"({re.escape(insert_after)})([.,;:!?])")
                    if pattern.search(paragraph):
                        modified_paragraph = pattern.sub(rf"\1{marker}\2", paragraph, count=1)
                    else:
                        word_pattern = re.compile(rf"({re.escape(insert_after)})(?![^\n]*\[\^)")
                        modified_paragraph = word_pattern.sub(rf"\1{marker}", paragraph, count=1)

                    paragraphs[i] = modified_paragraph
                    paragraph_footnotes[i].append(f"{marker}: {fn['footnote_text']}")
                    global_footnote_set.add(marker)  # Mark this footnote as used
                    found = True
                    logging.info(f"Applied footnote {idx} after '{insert_after}' in paragraph {i+1}")
                    break

            if not found:
                logging.warning(f"Could not find phrase '{insert_after}' in any paragraph (or phrase is in div block)")

        except Exception as e:
            logging.error(f"Error applying footnote {idx}: {e}")

    # Reconstruct paragraphs with their applied footnotes
    for i, para in enumerate(paragraphs):
        if paragraph_footnotes[i]:
            footnote_text = "\n".join(paragraph_footnotes[i])

            # Check if this paragraph contains an image/figure
            is_image = bool(re.search(r'!\[.*?\]\(.*?\)', para))

            # Add extra spacing after footnotes, especially for images and final paragraphs
            if is_image or i == len(paragraphs) - 1:
                modified_paragraphs.append(f"{para}\n\n{footnote_text}\n\n")
            else:
                modified_paragraphs.append(f"{para}\n\n{footnote_text}\n")
        else:
            modified_paragraphs.append(para)

    # Always ensure proper spacing between sections
    result = "\n\n".join(modified_paragraphs)

    # Ensure there's proper spacing at the end of the section
    if not result.endswith("\n\n"):
        result = result.rstrip() + "\n\n"

    return result

# --- Gradio GUI ---
def launch_gui(sections, headers, original_text, prompt_template, output_path, prologue):  # Added prologue parameter

    with gr.Blocks(css="""

    /* Aggressively target all possible spacing sources */
    .outline-btn {
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 0 8px !important;         /* Zero vertical padding */
        margin: 0 !important;
        font-size: 0.9em !important;
        background: none !important;
        border: none !important;
        box-shadow: none !important;
        height: 16px !important;           /* Extremely small height */
        min-height: 0 !important;
        color: #333 !important;
        border-radius: 0 !important;       /* Remove border radius */
        font-weight: normal !important;
        line-height: 1 !important;
        display: block !important;
    }

    /* Target absolutely everything that could add space */
    .outline-sidebar *,
    .outline-sidebar > *,
    .outline-sidebar > * > *,
    .outline-sidebar > * > * > *,
    .outline-sidebar button,
    .outline-sidebar div {
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1 !important;
    }

    /* Target Gradio's button container classes specifically */
    .outline-sidebar [class*="block"],
    .outline-sidebar [class*="Block"],
    .outline-sidebar [class*="container"],
    .outline-sidebar [class*="Container"] {
        margin: 0 !important;
        padding: 0 !important;
        display: block !important;
    }

    /* Force buttons to be butted against each other */
    .outline-sidebar button + div,
    .outline-sidebar div + button {
        margin-top: -1px !important; /* Negative margin to collapse any remnant space */
    }

    /* Remove any default button styles from Gradio */
    .outline-sidebar button {
        border: none !important;
        background-image: none !important;
        box-shadow: none !important;
        transition: none !important;
    }

    /* Target grandparent containers */
    .outline-sidebar > div > div {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }

    /* Force compact layout */
    .outline-sidebar * {
        line-height: 1 !important;
    }
        .container { width: 100%; }
        .main-container { display: flex; }

        .outline-sidebar {
            width: 120px !important; /* Even narrower */
            border-right: 1px solid #ddd;
            min-height: 500px;
            overflow-x: hidden;
        }
        .content-area { flex-grow: 1; padding: 0 15px; }
        .footnote-box { max-height: 300px; overflow-y: auto; border: 1px solid #ddd; border-radius: 5px; padding: 10px; }
        .progress-bar { margin-bottom: 15px; width: 100%; }
        .section-display { background-color: white; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
        .preview-box { background-color: white; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
        .marker { color: #2E86C1; font-weight: bold; background-color: #EBF5FB; border-radius: 3px; padding: 0 2px; }
        .footnote-select { margin-bottom: 10px; }
        .button-row { display: flex; justify-content: space-between; align-items: center; margin-top: 15px; }
        .button-row button { margin: 0 5px; }
        .status-message { color: #2E86C1; font-weight: bold; text-align: center; padding: 5px; }
        """) as demo:

        # Add this CSS to force vertical display of checkboxes
        gr.HTML("""
        <style>
            /* Force checkboxes to display as a vertical list */
            .footnote-box > div > div {
                display: flex !important;
                flex-direction: column !important;
            }

            /* Force each checkbox item to be full width */
            .footnote-box > div > div > label {
                width: 100% !important;
                margin-bottom: 8px !important;
                padding-bottom: 5px !important;
                border-bottom: 1px solid #eee !important;
            }

            /* Ensure the checkbox container doesn't use grid layout */
            .footnote-box .gr-form,
            .footnote-box .gr-form > div,
            .footnote-box .gr-panel {
                display: block !important;
            }

            /* Target any grid layouts and override them */
            .footnote-box [class*="grid"],
            .footnote-box [style*="grid"] {
                display: flex !important;
                flex-direction: column !important;
            }
        </style>
        """)

        # Global state
        current_section = gr.State(0)  # Current section index
        cached_footnotes = gr.State({})  # Cache footnotes by section
        cached_selections = gr.State({})  # Cache selected options by section
        updated_sections = gr.State(sections.copy())  # All updated sections
        applied_sections = gr.State(set())  # Track which sections have been applied

        # Hidden dropdown for JavaScript to use
        section_dropdown = gr.Dropdown(
            choices=[i for i in range(len(sections))],
            value=0,
            label="Section",
            interactive=True,
            visible=False,
            elem_id="section-dropdown"
        )

        gr.Markdown(f"## Academic Footnote Assistant (Using {api_provider} API with model {model_name})")

        # Main container with sidebar and content
        with gr.Row(elem_classes=["main-container"]):
            # Left sidebar for outline
            with gr.Column(scale=2, min_width=100, elem_classes=["outline-sidebar"]):
                gr.Markdown("### Document Outline")

                # Add custom CSS to fix button styling with minimal spacing
                gr.HTML("""
                <style>
                    /* Target the button containers to remove extra space */
                    .outline-sidebar > div,
                    .outline-sidebar > div > div {
                        margin: 0 !important;
                        padding: 0 !important;
                    }

                    /* Make outline buttons ultra-compact */
                    .outline-btn {
                        text-align: left !important;
                        justify-content: flex-start !important;
                        padding: 1px 8px !important;      /* Minimal vertical padding */
                        margin: 0 !important;             /* No margins */
                        font-size: 0.9em !important;
                        background: none !important;
                        border: none !important;
                        box-shadow: none !important;
                        height: 20px !important;          /* Fixed small height */
                        min-height: unset !important;     /* Override min-height */
                        color: #333 !important;
                        border-radius: 3px !important;
                        font-weight: normal !important;
                        line-height: 1 !important;        /* Minimal line height */
                        display: block !important;
                        overflow: hidden !important;
                        text-overflow: ellipsis !important;
                        white-space: nowrap !important;
                    }

                    .outline-btn:hover {
                        background-color: #f0f0f0 !important;
                    }

                    /* Ensure no extra space between buttons */
                    .outline-sidebar button + button,
                    .outline-sidebar div + div {
                        margin-top: 0 !important;
                    }

                    /* Compact any button container elements */
                    .outline-sidebar div[class*="container"],
                    .outline-sidebar div[class*="Container"] {
                        margin: 0 !important;
                        padding: 0 !important;
                    }

                    /* Hide any decorative elements that might add space */
                    .outline-sidebar div[class*="block"],
                    .outline-sidebar div[class*="Block"] {
                        margin: 0 !important;
                        padding: 0 !important;
                    }

                    /* Target the HTML components that add indentation styling */
                    .outline-sidebar > div > div:has(style) {
                        margin: 0 !important;
                        padding: 0 !important;
                        height: 0 !important;
                        overflow: hidden !important;
                    }
                </style>
                """)

                # Create all buttons in a single container to avoid spacing
                with gr.Column(elem_classes=["outline-buttons-container"]):
                    gr.HTML("""
                    <style>
                        .outline-buttons-container > div {
                            margin: 0 !important;
                            padding: 0 !important;
                        }
                    </style>
                    """)

                    # Create buttons for each header
                    for header in headers:
                        # Calculate indent
                        indent = 10 * (header["level"] - 1)

                        # Truncate long header text
                        display_text = header["text"]
                        if len(display_text) > 30:
                            display_text = display_text[:27] + "..."

                        # Create the button with indentation
                        btn = gr.Button(
                            display_text,
                            elem_classes=["outline-btn"],
                            elem_id=f"outline-btn-{header['index']}",
                            size="sm"
                        )

                        # Add CSS inline for this specific button
                        gr.HTML(f"""
                        <style>
                            #outline-btn-{header['index']} {{
                                margin-left: {indent}px !important;
                                width: calc(100% - {indent}px) !important;
                                margin-top: 0 !important;
                                margin-bottom: 0 !important;
                            }}
                        </style>
                        """)

                        # Set up click handler
                        def make_click_handler(idx):
                            def click_handler():
                                return idx
                            return click_handler

                        # Connect the button click to navigation
                        btn.click(
                            fn=make_click_handler(header["index"]),
                            inputs=[],
                            outputs=[current_section]
                        )

            # Right content area for the main UI
            with gr.Column(scale=9, elem_classes=["content-area"]):
                # Section info and progress
                with gr.Row(elem_classes=["container"]):
                    section_info = gr.Markdown("Section 1 of " + str(len(sections)), elem_classes=["status-message"])

                with gr.Row(elem_classes=["progress-bar", "container"]):
                    progress = gr.Slider(minimum=1, maximum=len(sections), value=1, step=1, label="Progress", interactive=False)

                # Section content
                gr.Markdown("<div class='container'><strong>Section with Suggested Footnotes:</strong></div>")
                section_html = gr.HTML(elem_classes=["container", "section-display"])

                # Hidden section text (for reference only)
                section_text = gr.Textbox(visible=False)

                # Footnote selection
                gr.Markdown("<div class='container'><strong>Select Footnotes to Apply:</strong></div>")

                with gr.Column(elem_classes=["container", "footnote-select"]):
                    checkbox_group = gr.CheckboxGroup(
                        choices=[],
                        value=[],
                        label="",
                        elem_classes=["footnote-box"]
                    )

                # Preview
                gr.Markdown("<div class='container'><strong>Preview Result:</strong></div>")
                preview_html = gr.HTML(elem_classes=["container", "preview-box"])

                # Action buttons
                with gr.Row(elem_classes=["container", "button-row"]):
                    prev_btn = gr.Button("⬅️ Previous")
                    regenerate_btn = gr.Button("🔄 Regenerate")
                    apply_btn = gr.Button("✅ Apply Section")
                    next_btn = gr.Button("Next ➡️")
                    save_btn = gr.Button("💾 Save & Exit", variant="primary", size="lg")

        # Status display
        status_display = gr.Markdown("", elem_classes=["container", "status-message"])

        # Process LLM response for a section - used by both load_section and regenerate
        def process_llm_response(section_idx, section_text_value, raw_response, cached_footnotes_dict, cached_selections_dict):
            try:
                # Try to extract JSON if it's embedded in markdown or examples
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_response, re.DOTALL)
                if json_match:
                    logging.info("Found JSON embedded in markdown, extracting...")
                    parsed_json = json_match.group(1)
                    data = json.loads(parsed_json)
                else:
                    data = json.loads(raw_response)

                footnotes = data.get("footnotes", [])
                logging.info(f"Parsed {len(footnotes)} footnotes from LLM response")

                # Verify footnote structure
                valid_footnotes = []
                for i, fn in enumerate(footnotes):
                    if all(key in fn for key in ["marker", "insert_after", "footnote_text"]):
                        valid_footnotes.append(fn)
                    else:
                        logging.warning(f"Footnote {i} missing required keys, skipping")

                # Create choices for checkbox - show marker and text
                checkbox_choices = []
                for i, fn in enumerate(valid_footnotes):
                    # Format like actual footnotes: [^marker]: text
                    checkbox_choices.append(f"{i+1}. {fn['marker']}: {fn['footnote_text']}")

                # Generate HTML with colored markers
                section_with_markers = show_section_with_markers(section_text_value, valid_footnotes)

                # Clear any previous selections for this section when regenerating
                if str(section_idx) in cached_selections_dict:
                    del cached_selections_dict[str(section_idx)]

                # Update the cache
                new_footnotes_dict = dict(cached_footnotes_dict)
                new_footnotes_dict[str(section_idx)] = valid_footnotes

                # Show default preview
                preview = section_text_value.replace("\n", "<br>")

                return {
                    'section_html': section_with_markers,
                    'checkbox_choices': checkbox_choices,
                    'checkbox_value': [],
                    'preview_html': preview,
                    'status_msg': f"Loaded {len(valid_footnotes)} footnote suggestions for section {section_idx + 1}",
                    'valid_footnotes': valid_footnotes,
                    'cached_footnotes': new_footnotes_dict
                }

            except Exception as e:
                logging.error(f"Error parsing LLM response: {e}")
                logging.error(f"Raw response: {raw_response}")

                return {
                    'section_html': f"<p>Error processing footnotes. Please check logs.</p><pre>{section_text_value}</pre>",
                    'checkbox_choices': [],
                    'checkbox_value': [],
                    'preview_html': section_text_value.replace("\n", "<br>"),
                    'status_msg': f"Error loading footnotes for section {section_idx + 1}",
                    'valid_footnotes': [],
                    'cached_footnotes': cached_footnotes_dict
                }

        # Function to load a section
        def load_section(index, cached_footnotes_dict, cached_selections_dict, applied_sections_set):
            logging.info(f"Loading section {index}")
            if index < 0:
                index = 0
            if index >= len(sections):
                return {
                    section_text: "",
                    section_html: "End of document reached.",
                    checkbox_group: gr.update(choices=[], value=[]),
                    preview_html: "",
                    progress: len(sections),
                    section_info: f"End of document reached",
                    status_display: "End of document reached. Click 'Save & Exit' to save your changes.",
                }

            section_text_value = sections[index]
            section_key = str(index)

            # Check if we already have footnotes for this section
            if section_key in cached_footnotes_dict:
                logging.info(f"Using cached footnotes for section {index}")
                valid_footnotes = cached_footnotes_dict[section_key]

                # Create choices for checkbox - show marker and text
                checkbox_choices = []
                for i, fn in enumerate(valid_footnotes):
                    # Format like actual footnotes: [^marker]: text
                    checkbox_choices.append(f"{i+1}. {fn['marker']}: {fn['footnote_text']}")

                # Generate HTML with colored markers
                section_with_markers = show_section_with_markers(section_text_value, valid_footnotes)

                # Check if we have saved selections for this section
                selected_values = []
                if section_key in cached_selections_dict:
                    selected_values = cached_selections_dict[section_key]

                # Generate preview based on saved selections
                preview = section_text_value.replace("\n", "<br>")
                if selected_values:
                    preview = show_preview_with_markers(section_text_value, selected_values, valid_footnotes)

                applied_status = ""
                if index in applied_sections_set:
                    applied_status = " (Applied)"

                return {
                    section_text: section_text_value,
                    section_html: section_with_markers,
                    checkbox_group: gr.update(choices=checkbox_choices, value=selected_values),
                    preview_html: preview,
                    progress: index + 1,
                    section_info: f"Section {index + 1} of {len(sections)}{applied_status}",
                    status_display: f"Loaded section {index + 1} with {len(valid_footnotes)} footnote suggestions",
                }
            else:
                # First time visiting this section - make the API call
                logging.info(f"First visit to section {index} - making API call")
                raw_response = get_footnote_suggestions(section_text_value, prompt_template)

                result = process_llm_response(index, section_text_value, raw_response,
                                             cached_footnotes_dict, cached_selections_dict)

                return {
                    section_text: section_text_value,
                    section_html: result['section_html'],
                    checkbox_group: gr.update(choices=result['checkbox_choices'], value=result['checkbox_value']),
                    preview_html: result['preview_html'],
                    progress: index + 1,
                    section_info: f"Section {index + 1} of {len(sections)}",
                    status_display: result['status_msg'],
                    cached_footnotes: result['cached_footnotes'],
                }

        # Function to regenerate footnotes for current section
        def regenerate_footnotes(section_idx, section_content, cached_footnotes_dict, cached_selections_dict):
            logging.info(f"Regenerating footnotes for section {section_idx}")
            raw_response = get_footnote_suggestions(section_content, prompt_template)

            result = process_llm_response(section_idx, section_content, raw_response,
                                         cached_footnotes_dict, cached_selections_dict)

            return {
                section_html: result['section_html'],
                checkbox_group: gr.update(choices=result['checkbox_choices'], value=result['checkbox_value']),
                preview_html: result['preview_html'],
                status_display: f"Regenerated footnotes for section {section_idx + 1}",
                cached_footnotes: result['cached_footnotes']
            }

        # Update preview based on checkbox selections
        def update_preview(selected_options, section_content, footnotes_dict, section_idx):
            section_key = str(section_idx)
            if section_key not in footnotes_dict:
                return section_content.replace("\n", "<br>")

            footnotes_data = footnotes_dict[section_key]

            if not selected_options:
                return section_content.replace("\n", "<br>")

            # Generate HTML with selected footnotes
            preview = show_preview_with_markers(section_content, selected_options, footnotes_data)
            return preview

        # Global set to track applied footnotes across sections
        global_footnote_set = set()

        def apply_to_section(section_idx, selected_options, footnotes_dict, updates, cached_selections_dict, applied_sections_set):
            """
            Apply selected footnotes to a section while preventing duplicate footnotes in the entire document.
            """
            section_key = str(section_idx)
            if section_key not in footnotes_dict:
                return {
                    updated_sections: updates,
                    cached_selections: cached_selections_dict,
                    applied_sections: applied_sections_set,
                    section_info: f"Section {section_idx + 1} of {len(sections)}",
                    status_display: "No footnotes available for this section"
                }

            footnotes_data = footnotes_dict[section_key]
            section_content = sections[section_idx]

            if selected_options:
                updated_text = apply_footnotes(section_content, selected_options, footnotes_data, global_footnote_set)

                # Update stored document state
                updated_sections_copy = updates.copy()
                updated_sections_copy[section_idx] = updated_text

                new_selections = dict(cached_selections_dict)
                new_selections[section_key] = selected_options

                new_applied = set(applied_sections_set)
                new_applied.add(section_idx)

                return {
                    updated_sections: updated_sections_copy,
                    cached_selections: new_selections,
                    applied_sections: new_applied,
                    section_info: f"Section {section_idx + 1} of {len(sections)} (Applied)",
                    status_display: f"Applied {len(selected_options)} footnotes to section {section_idx + 1}"
                }

            return {
                updated_sections: updates,
                cached_selections: cached_selections_dict,
                applied_sections: applied_sections_set,
                section_info: f"Section {section_idx + 1} of {len(sections)}",
                status_display: "No footnotes selected to apply"
            }


        # Move to previous section
        def prev_section(section_idx):
            prev_idx = section_idx - 1
            if prev_idx < 0:
                return {
                    status_display: "Already at the first section"
                }
            else:
                return {
                    current_section: prev_idx,
                    status_display: f"Moving to section {prev_idx + 1}"
                }

        # Move to next section
        def next_section(section_idx):
            next_idx = section_idx + 1
            if next_idx >= len(sections):
                return {
                    status_display: "End of document reached"
                }
            else:
                return {
                    current_section: next_idx,
                    status_display: f"Moving to section {next_idx + 1}"
                }

        def save_document(updates):
            try:
                # We'll rebuild the document from scratch
                rebuilt_document = []
                changes_applied = a = 0

                logging.info(f"Starting save process with {len(updates)} sections")

                # Track which sections were actually changed
                changed_sections = []

                # Go through each section in order
                for i, (original, updated) in enumerate(zip(sections, updates)):
                    # Determine if this section was modified
                    if original != updated:
                        # Add the updated version to our rebuilt document
                        rebuilt_document.append(updated)
                        changed_sections.append(i)
                        changes_applied += 1
                        logging.info(f"Section {i}: Using modified version")
                    else:
                        # Add the original version to our rebuilt document
                        rebuilt_document.append(original)
                        logging.info(f"Section {i}: Using original version")

                # Join all sections to create the full document
                full_text = prologue + "\n\n" + "\n".join(rebuilt_document)

                # Create output filename
                output_filename = output_path
                if "." in output_path:
                    base, ext = output_path.rsplit(".", 1)
                    output_filename = f"{base}-footnoted.{ext}"
                else:
                    output_filename = f"{output_path}-footnoted"

                # Write the rebuilt document
                with open(output_filename, "w", encoding="utf-8") as f:
                    f.write(full_text)

                # Write a debug file with just the changed sections for verification
                if changed_sections:
                    debug_filename = f"changes_debug_{int(time.time())}.txt"
                    with open(debug_filename, "w", encoding="utf-8") as f:
                        for idx in changed_sections:
                            f.write(f"=== SECTION {idx} ===\n")
                            f.write(f"ORIGINAL:\n{sections[idx]}\n\n")
                            f.write(f"UPDATED:\n{updates[idx]}\n\n")
                            f.write("="*50 + "\n\n")

                logging.info(f"Saved document with {changes_applied} changed sections to {output_filename}")
                return f"Document saved to {output_filename} with {changes_applied} changes"
            except Exception as e:
                import traceback
                logging.error(f"Save error: {str(e)}")
                logging.error(traceback.format_exc())
                return f"Error: {str(e)}"

        # Set up event handlers
        # Section dropdown (for outline navigation)
        section_dropdown.change(
            fn=lambda x: x,
            inputs=[section_dropdown],
            outputs=[current_section]
        )

        # Regenerate button
        regenerate_btn.click(
            fn=regenerate_footnotes,
            inputs=[current_section, section_text, cached_footnotes, cached_selections],
            outputs=[section_html, checkbox_group, preview_html, status_display, cached_footnotes]
        )

        # Auto-update preview when checkboxes change
        checkbox_group.change(
            fn=update_preview,
            inputs=[checkbox_group, section_text, cached_footnotes, current_section],
            outputs=preview_html
        )

        # Apply footnotes to current section
        apply_btn.click(
            fn=apply_to_section,
            inputs=[current_section, checkbox_group, cached_footnotes, updated_sections, cached_selections, applied_sections],
            outputs=[updated_sections, cached_selections, applied_sections, section_info, status_display]
        )

        # Previous section button
        prev_btn.click(
            fn=prev_section,
            inputs=[current_section],
            outputs=[current_section, status_display]
        )

        # Next section button
        next_btn.click(
            fn=next_section,
            inputs=[current_section],
            outputs=[current_section, status_display]
        )

        # Save button
        save_btn.click(
            fn=save_document,
            inputs=[updated_sections],
            outputs=status_display
        )

        # Handle section changes
        current_section.change(
            fn=load_section,
            inputs=[current_section, cached_footnotes, cached_selections, applied_sections],
            outputs=[section_text, section_html, checkbox_group, preview_html,
                    progress, section_info, status_display, cached_footnotes]
        )

        # Load the first section on startup
        demo.load(
            fn=lambda: load_section(0, {}, {}, set()),
            inputs=None,
            outputs=[section_text, section_html, checkbox_group, preview_html,
                    progress, section_info, status_display, cached_footnotes]
        )

    logging.info("Launching Gradio interface")
    demo.launch(share=True)
    logging.info("Gradio interface closed")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="Input file path")
    parser.add_argument("--prompt", default="prompt.txt", help="Prompt template file path (default: prompt.txt)")
    parser.add_argument("--api", default="openai", choices=["openai", "groq"], help="API provider (default: openai)")
    parser.add_argument("--model", help="Model name (defaults: gpt-4 for OpenAI, llama3-70b-8192 for Groq)")
    args = parser.parse_args()

    # Set default model based on API choice if not provided
    global api_provider, model_name, client
    api_provider = args.api.lower()

    if args.model:
        model_name = args.model
    else:
        # Default models
        if api_provider == "openai":
            model_name = "gpt-4-turbo"
        elif api_provider == "groq":
            model_name = "llama3-70b-8192"
        else:
            model_name = "gpt-4"  # Default fallback

    logging.info(f"Starting application with file: {args.file}, prompt template: {args.prompt}")
    logging.info(f"Using API provider: {api_provider} with model: {model_name}")

    # Initialize the appropriate client
    api_key_var = "OPENAI_API_KEY" if api_provider == "openai" else "GROQ_API_KEY"
    api_key = os.getenv(api_key_var)

    if not api_key:
        logging.error(f"Error: {api_key_var} environment variable is not set")
        print(f"❌ Error: {api_key_var} environment variable is not set")
        print(f"Please set it by running: export {api_key_var}=your_api_key")
        return

    # Initialize the client based on provider
    if api_provider == "openai":
        client = OpenAI(api_key=api_key)
    elif api_provider == "groq":
        client = Groq(api_key=api_key)
    else:
        logging.error(f"Unsupported API provider: {api_provider}")
        print(f"❌ Error: Unsupported API provider: {api_provider}")
        return

    # Read the input file and prompt template
    try:
        with open(args.file) as f:
            text = f.read()
        logging.info(f"Successfully read input file: {args.file}")

        with open(args.prompt) as f:
            prompt = f.read()
        logging.info(f"Successfully read prompt template: {args.prompt}")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        print(f"Error: {e}")
        return
    except Exception as e:
        logging.error(f"Error reading files: {e}")
        print(f"Error: {e}")
        return

    # Parse sections and headers
    sections, headers, prologue = parse_qmd_sections(text)  # Now properly unpacking prologue
    launch_gui(sections, headers, text, prompt, args.file, prologue)  # Pass prologue to the GUI

if __name__ == "__main__":
    main()
