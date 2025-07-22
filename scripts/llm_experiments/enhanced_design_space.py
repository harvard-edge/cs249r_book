# Enhanced Design Space for LLM Cross-Reference Optimization
#
# This module defines the complete parameter space for systematic optimization
# of cross-reference explanation generation.

# 🤖 MODEL DIMENSION
AVAILABLE_MODELS = [
    "qwen2.5:7b",      # Chinese-trained, excellent reasoning
    "llama3.1:8b",     # Meta's flagship, balanced performance  
    "mistral:7b",      # European, efficient, instruction-tuned
    "gemma2:9b",       # Google's latest, strong reasoning
    "phi3:3.8b"        # Microsoft, compact but powerful
]

# 📏 LENGTH DIMENSION  
LENGTH_TARGETS = [
    {"min_words": 3, "max_words": 5, "description": "ultra_short"},
    {"min_words": 4, "max_words": 7, "description": "short"},
    {"min_words": 6, "max_words": 10, "description": "medium"},
    {"min_words": 8, "max_words": 12, "description": "standard"},
    {"min_words": 10, "max_words": 15, "description": "extended"}
]

# 🌡️ TEMPERATURE DIMENSION (Creativity)
TEMPERATURE_VALUES = [
    {"value": 0.1, "description": "conservative"},
    {"value": 0.3, "description": "balanced"},      # Current
    {"value": 0.5, "description": "creative"},
    {"value": 0.7, "description": "highly_creative"}
]

# 🎯 TOP_P DIMENSION (Nucleus Sampling)
TOP_P_VALUES = [
    {"value": 0.7, "description": "focused"},
    {"value": 0.9, "description": "diverse"},       # Current
    {"value": 0.95, "description": "very_diverse"}
]

# 📝 PROMPT STYLE DIMENSION
PROMPT_STYLES = [
    {
        "name": "natural",
        "template": "Write a natural {length_desc} explanation that completes: \"See also: {target_title} - [your explanation]\"",
        "description": "Current style - natural completion"
    },
    {
        "name": "directive", 
        "template": "Explain in {length_desc} why students should read \"{target_title}\" after \"{source_title}\":",
        "description": "Direct instruction style"
    },
    {
        "name": "summary",
        "template": "Summarize in {length_desc} how \"{target_title}\" relates to \"{source_title}\":",
        "description": "Summary-focused style"
    },
    {
        "name": "contextual",
        "template": "In {length_desc}, explain why \"{target_title}\" matters for understanding \"{source_title}\":",
        "description": "Context-focused style"
    }
]

# 📖 CONTEXT LENGTH DIMENSION
CONTEXT_LENGTHS = [
    {"chars": 200, "description": "minimal"},
    {"chars": 500, "description": "standard"},      # Current
    {"chars": 1000, "description": "extended"}
]

# 🎨 EXPLANATION FOCUS DIMENSION  
EXPLANATION_FOCUSES = [
    {
        "name": "foundational",
        "instruction": "Focus on fundamental concepts and background knowledge.",
        "examples": ["provides essential background", "covers prerequisite concepts"]
    },
    {
        "name": "practical", 
        "instruction": "Focus on implementation and practical applications.",
        "examples": ["shows practical applications", "demonstrates real-world uses"]
    },
    {
        "name": "comparative",
        "instruction": "Focus on differences, similarities, and contrasts.",
        "examples": ["contrasts different approaches", "compares alternative methods"]
    },
    {
        "name": "contextual",
        "instruction": "Focus on why this connection matters and its significance.", 
        "examples": ["explains why this matters", "provides crucial context"]
    }
]

# 🧪 EXPERIMENT CONFIGURATIONS
EXPERIMENT_CONFIGS = {
    "quick_model_comparison": {
        "models": AVAILABLE_MODELS,
        "length_targets": [LENGTH_TARGETS[1]],  # Just "short"
        "temperatures": [TEMPERATURE_VALUES[1]],  # Just 0.3
        "top_ps": [TOP_P_VALUES[1]],  # Just 0.9
        "prompt_styles": [PROMPT_STYLES[0]],  # Just "natural"
        "context_lengths": [CONTEXT_LENGTHS[1]],  # Just 500 chars
        "focuses": [EXPLANATION_FOCUSES[0]]  # Just "foundational"
    },
    
    "comprehensive_sweep": {
        "models": AVAILABLE_MODELS,
        "length_targets": LENGTH_TARGETS,
        "temperatures": TEMPERATURE_VALUES,
        "top_ps": TOP_P_VALUES,
        "prompt_styles": PROMPT_STYLES,
        "context_lengths": CONTEXT_LENGTHS,
        "focuses": EXPLANATION_FOCUSES
    },
    
    "length_optimization": {
        "models": ["qwen2.5:7b"],  # Use best model from quick comparison
        "length_targets": LENGTH_TARGETS,
        "temperatures": [TEMPERATURE_VALUES[1]],  # Fixed params
        "top_ps": [TOP_P_VALUES[1]],
        "prompt_styles": [PROMPT_STYLES[0]],
        "context_lengths": [CONTEXT_LENGTHS[1]],
        "focuses": [EXPLANATION_FOCUSES[0]]
    },
    
    "prompt_optimization": {
        "models": ["qwen2.5:7b"],  # Use best model
        "length_targets": [LENGTH_TARGETS[1]],  # Use best length
        "temperatures": TEMPERATURE_VALUES,
        "top_ps": TOP_P_VALUES, 
        "prompt_styles": PROMPT_STYLES,
        "context_lengths": CONTEXT_LENGTHS,
        "focuses": EXPLANATION_FOCUSES
    }
}

def get_total_experiments(config_name: str) -> int:
    """Calculate total number of experiments for a given configuration."""
    config = EXPERIMENT_CONFIGS[config_name]
    total = 1
    for dimension in config.values():
        total *= len(dimension)
    return total

def get_experiment_summary():
    """Print a summary of all available experiment configurations."""
    print("🧪 EXPERIMENT CONFIGURATION SUMMARY")
    print("=" * 50)
    
    for name, config in EXPERIMENT_CONFIGS.items():
        total = get_total_experiments(name)
        print(f"\n📊 {name.upper()}:")
        print(f"   Total combinations: {total:,}")
        print(f"   Models: {len(config['models'])}")
        print(f"   Length targets: {len(config['length_targets'])}")
        print(f"   Temperature values: {len(config['temperatures'])}")
        print(f"   Top-p values: {len(config['top_ps'])}")
        print(f"   Prompt styles: {len(config['prompt_styles'])}")
        print(f"   Context lengths: {len(config['context_lengths'])}")
        print(f"   Focus types: {len(config['focuses'])}")
        
        # Estimate time (assuming 3 seconds per combination with 5 test cases)
        estimated_time = (total * 5 * 3) / 60  # Convert to minutes
        print(f"   Estimated time: {estimated_time:.1f} minutes")

if __name__ == "__main__":
    get_experiment_summary() 