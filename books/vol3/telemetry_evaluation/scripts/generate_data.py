import pandas as pd
import os

def generate_static_vs_agentic():
    """
    Generates the static_vs_agentic.csv dataset which tracks model performance 
    on static benchmarks (MMLU) versus agentic benchmarks (SWE_bench_Lite) over time.
    """
    
    # Data sourced from official model technical reports, release blog posts, and the SWE-bench leaderboard.
    # - MMLU: Measuring Massive Multitask Language Understanding (Static benchmark)
    # - SWE_bench_Lite: Software Engineering benchmark resolving real GitHub issues (Agentic benchmark)
    data = [
        {
            "Date": "2023-03-14",
            "Model": "GPT-4",
            "MMLU": 86.4,
            "SWE_bench_Lite": 1.7,
            # Source: OpenAI GPT-4 Technical Report / Initial SWE-bench paper baselines
        },
        {
            "Date": "2024-03-04",
            "Model": "Claude 3 Opus",
            "MMLU": 86.8,
            "SWE_bench_Lite": 11.7,
            # Source: Anthropic Claude 3 release post / SWE-bench Leaderboard
        },
        {
            "Date": "2024-05-13",
            "Model": "GPT-4o",
            "MMLU": 88.7,
            "SWE_bench_Lite": 16.0,
            # Source: OpenAI GPT-4o release announcement / SWE-bench Leaderboard
        },
        {
            "Date": "2024-06-20",
            "Model": "Claude 3.5 Sonnet",
            "MMLU": 88.3,
            "SWE_bench_Lite": 26.6,
            # Source: Anthropic Claude 3.5 Sonnet release post / SWE-bench Leaderboard
        },
        {
            "Date": "2024-09-12",
            "Model": "o1-preview",
            "MMLU": 90.8,
            "SWE_bench_Lite": 40.8,
            # Source: OpenAI o1 release post / SWE-bench Leaderboard
        },
        {
            "Date": "2024-10-22",
            "Model": "Claude 3.5 Sonnet (v2)",
            "MMLU": 88.7,
            "SWE_bench_Lite": 49.0,
            # Source: Anthropic upgraded Claude 3.5 Sonnet announcement / SWE-bench Leaderboard
        }
    ]
    
    df = pd.DataFrame(data)
    
    # Ensure the output directory exists
    output_dir = "../data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Write to CSV
    output_path = os.path.join(output_dir, "static_vs_agentic.csv")
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    # Ensure script is run from the scripts directory so relative paths work
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    generate_static_vs_agentic()
