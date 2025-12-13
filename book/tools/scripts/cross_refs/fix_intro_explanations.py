#!/usr/bin/env python3
"""
Fix introduction chapter explanations with proper academic style.
No marketing language, just technical descriptions.
"""

import json
from pathlib import Path

def get_proper_explanation(target_chapter: str, connection_type: str, section_context: str = "") -> str:
    """Generate proper academic explanations based on chapter and connection type"""

    # Academic-style explanations by chapter and type
    explanations = {
        # ML Systems - core architecture
        ("ml_systems", "foundation"): "System architecture design patterns",
        ("ml_systems", "prerequisite"): "Core ML systems concepts",
        ("ml_systems", "extends"): "Advanced systems engineering",

        # DL Primer - fundamentals
        ("dl_primer", "foundation"): "Neural network fundamentals",
        ("dl_primer", "prerequisite"): "Deep learning mathematics",
        ("dl_primer", "extends"): "Advanced network architectures",

        # Workflow - pipelines
        ("workflow", "complements"): "Pipeline orchestration patterns",
        ("workflow", "extends"): "Workflow automation strategies",
        ("workflow", "foundation"): "ML pipeline components",

        # Data Engineering
        ("data_engineering", "foundation"): "Data pipeline architecture",
        ("data_engineering", "extends"): "Scalable data processing",
        ("data_engineering", "complements"): "ETL design patterns",

        # Frameworks
        ("frameworks", "foundation"): "Framework architecture comparison",
        ("frameworks", "extends"): "Production framework deployment",
        ("frameworks", "complements"): "Framework selection criteria",

        # Training
        ("training", "foundation"): "Optimization algorithms",
        ("training", "extends"): "Distributed training methods",
        ("training", "prerequisite"): "Gradient descent theory",

        # Efficient AI
        ("efficient_ai", "extends"): "Model compression techniques",
        ("efficient_ai", "optimizes"): "Inference optimization strategies",
        ("efficient_ai", "complements"): "Efficiency-accuracy tradeoffs",

        # Optimizations
        ("optimizations", "extends"): "Performance tuning methods",
        ("optimizations", "optimizes"): "System-level optimizations",
        ("optimizations", "complements"): "Optimization tradeoffs",

        # Hardware Acceleration
        ("hw_acceleration", "extends"): "GPU/TPU programming",
        ("hw_acceleration", "optimizes"): "Hardware-specific optimizations",
        ("hw_acceleration", "complements"): "Accelerator architectures",

        # Benchmarking
        ("benchmarking", "extends"): "Performance evaluation metrics",
        ("benchmarking", "optimizes"): "Benchmark design principles",
        ("benchmarking", "complements"): "Comparative analysis methods",

        # Ops
        ("ops", "applies"): "Production deployment patterns",
        ("ops", "extends"): "MLOps best practices",
        ("ops", "complements"): "Monitoring and debugging",

        # On-device Learning
        ("ondevice_learning", "applies"): "Edge deployment strategies",
        ("ondevice_learning", "extends"): "Mobile optimization techniques",
        ("ondevice_learning", "optimizes"): "Resource-constrained inference",

        # Privacy & Security
        ("privacy_security", "considers"): "Privacy-preserving techniques",
        ("privacy_security", "explores"): "Security threat models",
        ("privacy_security", "extends"): "Federated learning protocols",

        # Responsible AI
        ("responsible_ai", "considers"): "Ethical ML frameworks",
        ("responsible_ai", "explores"): "Bias detection methods",
        ("responsible_ai", "extends"): "Fairness constraints",

        # Robust AI
        ("robust_ai", "considers"): "Adversarial defense strategies",
        ("robust_ai", "explores"): "Robustness verification methods",
        ("robust_ai", "extends"): "Certified defense techniques",

        # Generative AI
        ("generative_ai", "specializes"): "Generative model architectures",
        ("generative_ai", "extends"): "Advanced generation techniques",
        ("generative_ai", "anticipates"): "Emerging generative methods",

        # Sustainable AI
        ("sustainable_ai", "specializes"): "Energy-efficient architectures",
        ("sustainable_ai", "considers"): "Carbon footprint analysis",
        ("sustainable_ai", "extends"): "Green computing strategies",

        # AI for Good
        ("ai_for_good", "specializes"): "Social impact applications",
        ("ai_for_good", "considers"): "Humanitarian use cases",
        ("ai_for_good", "extends"): "Impact measurement frameworks",

        # Frontiers
        ("frontiers", "anticipates"): "Emerging research directions",
        ("frontiers", "explores"): "Next-generation architectures",
        ("frontiers", "extends"): "Cutting-edge techniques",

        # Emerging Topics
        ("emerging_topics", "anticipates"): "Latest ML developments",
        ("emerging_topics", "explores"): "Research trends analysis",
        ("emerging_topics", "extends"): "Future system designs"
    }

    # Get specific explanation or fall back to generic
    key = (target_chapter, connection_type)
    if key in explanations:
        return explanations[key]

    # Generic fallbacks by connection type
    fallbacks = {
        "foundation": f"{target_chapter.replace('_', ' ').title()} fundamentals",
        "prerequisite": f"Required {target_chapter.replace('_', ' ')} concepts",
        "extends": f"Advanced {target_chapter.replace('_', ' ')} techniques",
        "complements": f"Alternative {target_chapter.replace('_', ' ')} approaches",
        "applies": f"{target_chapter.replace('_', ' ').title()} implementation",
        "optimizes": f"{target_chapter.replace('_', ' ').title()} optimization",
        "considers": f"{target_chapter.replace('_', ' ').title()} considerations",
        "explores": f"Deeper {target_chapter.replace('_', ' ')} analysis",
        "anticipates": f"Future {target_chapter.replace('_', ' ')} directions",
        "specializes": f"Specialized {target_chapter.replace('_', ' ')} applications"
    }

    return fallbacks.get(connection_type, f"{target_chapter.replace('_', ' ').title()} concepts")

def fix_introduction_explanations():
    """Fix all explanations in introduction chapter"""

    xref_path = Path("/Users/VJ/GitHub/MLSysBook/quarto/contents/core/introduction/introduction_xrefs.json")

    with open(xref_path) as f:
        data = json.load(f)

    fixes = 0

    for section_id, refs in data.get('cross_references', {}).items():
        for ref in refs:
            old = ref.get('explanation', '')

            # Generate proper academic explanation
            new = get_proper_explanation(
                ref.get('target_chapter'),
                ref.get('connection_type', 'related')
            )

            if new != old:
                ref['explanation'] = new
                fixes += 1

    # Save updated file
    with open(xref_path, 'w') as f:
        json.dump(data, f, indent=2)

    return fixes

def main():
    print("Fixing introduction chapter explanations...")
    print("="*50)

    fixes = fix_introduction_explanations()

    print(f"✅ Fixed {fixes} explanations")
    print("\nNew style:")
    print("  • Technical descriptions (8-10 words)")
    print("  • No marketing language")
    print("  • Academic tone")
    print("\nExamples:")
    print('  • "Neural network fundamentals"')
    print('  • "Distributed training methods"')
    print('  • "Pipeline orchestration patterns"')

if __name__ == "__main__":
    main()
