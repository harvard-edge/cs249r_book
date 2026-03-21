import streamlit as st
import sys
import os
import glob
import re
import random

# Find the repository root dynamically and add to path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
interviews_dir = os.path.join(repo_root, "interviews")

# Add mlsysim to path so we can run actual physics simulations!
sys.path.insert(0, os.path.join(repo_root, "mlsysim"))
try:
    import mlsysim
    SIMULATOR_AVAILABLE = True
except ImportError:
    SIMULATOR_AVAILABLE = False

# --- Parser Logic (Inlined for standalone deployment) ---
def parse_markdown_questions(file_path: str) -> list[dict]:
    questions = []
    if not os.path.exists(file_path):
        return questions
        
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    pattern = re.compile(r'<summary><b><img src=.*?alt="([^"]+)".*?>([^<]+)</b>.*?</summary>\s*-\s*\*\*Interviewer:\*\*\s*"(.*?)"\s*<details>\s*<summary><b>🔍 Reveal Answer</b></summary>\s*(.*?)\s*</details>\s*</details>', re.DOTALL)
    matches = pattern.finditer(content)
    for match in matches:
        level = match.group(1).strip()
        title = match.group(2).replace('·', '').strip()
        prompt = match.group(3).strip()
        answer = match.group(4).strip()
        questions.append({
            "level": level, "title": title, "prompt": prompt,
            "answer": answer, "file": os.path.basename(file_path)
        })
    return questions

def get_numbers_cheat_sheet(base_dir: str) -> str:
    numbers_file = os.path.join(base_dir, "NUMBERS.md")
    try:
        with open(numbers_file, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Error: NUMBERS.md not found."

# --- Streamlit UI ---
st.set_page_config(
    page_title="MLSys Interview Simulator", 
    page_icon="⚡", layout="wide", initial_sidebar_state="expanded"
)

# Custom CSS for styling (Matching MLSysBook Ecosystem)
st.markdown("""
<style>
    .main-header { font-size: 2.8rem !important; font-weight: 800; margin-bottom: 0rem; color: #e8e8f0; letter-spacing: -0.02em; }
    .sub-header { font-size: 1.2rem; color: #9090a0; margin-bottom: 2.5rem; font-style: italic; }
    .question-card { background-color: rgba(99, 102, 241, 0.08); border-left: 5px solid #6366f1; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem; box-shadow: 0 2px 8px rgba(99, 102, 241, 0.1); }
    .expert-card { background-color: rgba(34, 197, 94, 0.05); border-left: 5px solid #22c55e; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem; }
    .level-badge { display: inline-block; padding: 0.2rem 0.6rem; font-size: 0.8rem; font-weight: bold; color: #fff; border-radius: 12px; margin-bottom: 1rem; letter-spacing: 0.05em; text-transform: uppercase; }
    .level-l3 { background-color: #22c55e; } .level-l4 { background-color: #3b82f6; } .level-l5 { background-color: #eab308; } .level-l6 { background-color: #ef4444; }
    .stTextArea textarea { background-color: #1a1a24 !important; color: #e8e8f0 !important; border: 1px solid #2a2a3a !important; border-radius: 0.5rem; }
    .stTextArea textarea:focus { border-color: #6366f1 !important; box-shadow: 0 0 0 1px #6366f1 !important; }
    code { color: #818cf8 !important; background-color: rgba(99, 102, 241, 0.1) !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">⚡ The MLSys Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">You can generate the code, but you cannot prompt your way out of a silicon bottleneck.</p>', unsafe_allow_html=True)

# Initialize session state
if 'current_question' not in st.session_state: st.session_state.current_question = None
if 'revealed' not in st.session_state: st.session_state.revealed = False
if 'streak' not in st.session_state: st.session_state.streak = 0
if 'completed_questions' not in st.session_state: st.session_state.completed_questions = set()

# Sidebar
with st.sidebar:
    st.markdown("### Select a Track")
    track = st.radio("Deployment Regime:", ["☁️ Cloud", "🤖 Edge", "📱 Mobile", "🔬 TinyML"], label_visibility="collapsed")
    clean_track = track.split(" ")[1].lower()
    level_filter = st.selectbox("Difficulty Level (Optional):", ["All Levels", "L3", "L4", "L5", "L6+"])
    
    if st.button("🎲 Draw Scenario", type="primary", use_container_width=True):
        track_dir = os.path.join(interviews_dir, clean_track)
        if os.path.exists(track_dir):
            md_files = [f for f in glob.glob(os.path.join(track_dir, "*.md")) if not f.endswith("README.md")]
            all_questions = []
            for file in md_files: all_questions.extend(parse_markdown_questions(file))
            if level_filter != "All Levels": all_questions = [q for q in all_questions if level_filter in q.get('level', '')]
            
            if all_questions:
                st.session_state.current_question = random.choice(all_questions)
                st.session_state.revealed = False
            else:
                st.error("No questions found for this criteria.")
        else:
            st.error(f"Directory {track_dir} not found.")
            
    # Streak Counter Display
    if st.session_state.streak > 0:
        st.markdown(f"""
        <div style="background-color: rgba(234, 179, 8, 0.1); border: 1px solid #eab308; border-radius: 5px; padding: 10px; text-align: center; margin-top: 10px;">
            <span style="font-size: 1.5rem;">🔥</span><br>
            <span style="font-weight: bold; color: #eab308;">{st.session_state.streak} Scenario Streak!</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("📊 View Numbers Cheat Sheet"):
        st.markdown(get_numbers_cheat_sheet(interviews_dir))

# Main content area
if st.session_state.current_question:
    q = st.session_state.current_question
    
    # Determine badge color
    level_lower = q['level'].lower()
    badge_class = "level-badge"
    if "l3" in level_lower or "junior" in level_lower: badge_class += " level-l3"
    elif "l4" in level_lower or "mid" in level_lower: badge_class += " level-l4"
    elif "l5" in level_lower or "senior" in level_lower: badge_class += " level-l5"
    elif "l6" in level_lower or "principal" in level_lower: badge_class += " level-l6"

    st.markdown(f"""
    <div class="question-card">
        <span class="{badge_class}">{q['level']}</span>
        <h3>{q['title']}</h3>
        <p><b>Interviewer:</b> "{q['prompt']}"</p>
    </div>
    """, unsafe_allow_html=True)
    
    user_notes = st.text_area("Napkin Math / Rough Notes (Optional):", height=150, placeholder="Identify the bottleneck, write the equation, calculate the overhead...")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔍 Reveal Expert Answer", use_container_width=True):
            st.session_state.revealed = True
            q_id = f"{q['file']}-{q['title']}"
            if q_id not in st.session_state.completed_questions:
                st.session_state.streak += 1
                st.session_state.completed_questions.add(q_id)
            
    if st.session_state.revealed:
        st.markdown("---")
        st.markdown("### The Architect's Breakdown")
        answer_text = q['answer']
        answer_text = answer_text.replace("**Common Mistake:**", "🚨 **Common Mistake:**")
        answer_text = answer_text.replace("**Realistic Solution:**", "✅ **Realistic Solution:**")
        answer_text = answer_text.replace("> **Napkin Math:**", "🧮 **Napkin Math:**")
        
        st.markdown(f'<div class="expert-card">\n\n{answer_text}\n\n</div>', unsafe_allow_html=True)
        
        col_source, col_share = st.columns([3, 1])
        with col_source:
            st.markdown(f"*Source: `{clean_track}/{q['file']}`*")
        with col_share:
            share_text = f"I just tackled this {q['level']} ML Systems architecture scenario:\n\n\"{q['prompt']}\"\n\nTest your mechanical sympathy: mlsysbook.ai/interviews"
            st.text_area("Share to Flex:", value=share_text, height=100)
else:
    st.markdown("""
    ### Welcome to the Assessment Engine
    
    To master ML Systems, you must practice diagnosing failures and calculating hardware constraints. 
    
    **How to use this tool:**
    1. Select a track on the left (Cloud, Edge, Mobile, or TinyML).
    2. Click **Draw Scenario** to pull a random technical prompt.
    3. Use the text box to work out your logic and 'Napkin Math'.
    4. Reveal the answer to see how an L5/L6 architect would break down the physics of the problem.
    
    👈 *Start by drawing a scenario.*
    """)
    
    # Show some quick stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Scenarios", value="240+")
    with col2:
        st.metric(label="Deployment Tracks", value="4")
    with col3:
        st.metric(label="Difficulty Levels", value="L3 - L6+")
