#!/usr/bin/env python3
"""
Knowledge Map: Track and build upon learned concepts throughout the textbook
===========================================================================

This module maintains a progressive knowledge map that tracks:
1. What concepts have been introduced in previous chapters
2. Prerequisites for current concepts
3. How to scaffold questions based on prior knowledge
"""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json

@dataclass
class Concept:
    """Represents a single concept in the knowledge map."""
    id: str
    name: str
    chapter: str
    section: str
    bloom_level: str  # The level at which this was introduced
    prerequisites: List[str] = field(default_factory=list)
    enables: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    
class KnowledgeMap:
    """
    Maintains the knowledge progression throughout the ML Systems textbook.
    """
    
    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.chapter_order = [
            "introduction",
            "ml_systems",
            "dl_primer", 
            "data_engineering",
            "dnn_architectures",
            "frameworks",
            "training",
            "efficient_ai",
            "optimizations",      # Great for CALC questions
            "hw_acceleration",
            "benchmarking",
            "ops",
            "ondevice_learning",
            "robust_ai",
            "privacy_security",
            "responsible_ai",
            "sustainable_ai",
            "ai_for_good",
            "workflow",
            "conclusion"
        ]
        self._initialize_knowledge_map()
    
    def _initialize_knowledge_map(self):
        """Initialize the knowledge map with ML Systems concepts."""
        
        # Introduction chapter concepts
        self.add_concept(Concept(
            id="ml_basics",
            name="Machine Learning Basics",
            chapter="introduction",
            section="ml_fundamentals",
            bloom_level="understand",
            enables=["supervised_learning", "unsupervised_learning", "model_training"]
        ))
        
        # ML Systems chapter
        self.add_concept(Concept(
            id="system_stack",
            name="ML System Stack",
            chapter="ml_systems", 
            section="system_architecture",
            bloom_level="understand",
            prerequisites=["ml_basics"],
            enables=["hardware_software_interface", "deployment_strategies"]
        ))
        
        # Training chapter
        self.add_concept(Concept(
            id="gradient_descent",
            name="Gradient Descent",
            chapter="training",
            section="optimization_algorithms",
            bloom_level="apply",
            prerequisites=["ml_basics"],
            enables=["sgd", "adam", "learning_rate_scheduling"],
            examples=["Calculate gradient update: w_new = w_old - lr * gradient"]
        ))
        
        self.add_concept(Concept(
            id="batch_size_tradeoffs",
            name="Batch Size Tradeoffs",
            chapter="training",
            section="hyperparameters",
            bloom_level="analyze",
            prerequisites=["gradient_descent"],
            enables=["memory_optimization", "convergence_analysis"],
            examples=["Memory usage = model_size + batch_size * activation_memory"]
        ))
        
        # Optimizations chapter - EXCELLENT for CALC questions
        self.add_concept(Concept(
            id="quantization",
            name="Model Quantization",
            chapter="optimizations",
            section="compression_techniques",
            bloom_level="apply",
            prerequisites=["model_training", "numerical_precision"],
            enables=["int8_inference", "mixed_precision"],
            examples=[
                "Memory savings = (1 - new_bits/original_bits) * 100%",
                "INT8 quantization: 32/8 = 4x memory reduction"
            ]
        ))
        
        self.add_concept(Concept(
            id="pruning",
            name="Model Pruning",
            chapter="optimizations",
            section="sparsity",
            bloom_level="analyze",
            prerequisites=["neural_networks", "model_training"],
            enables=["structured_pruning", "unstructured_pruning"],
            examples=[
                "Sparsity = pruned_params / total_params",
                "FLOPs reduction ≈ sparsity² for unstructured pruning"
            ]
        ))
        
        self.add_concept(Concept(
            id="knowledge_distillation",
            name="Knowledge Distillation",
            chapter="optimizations",
            section="model_compression",
            bloom_level="evaluate",
            prerequisites=["model_training", "loss_functions"],
            enables=["teacher_student", "ensemble_distillation"],
            examples=[
                "Compression ratio = teacher_params / student_params",
                "Speedup = teacher_latency / student_latency"
            ]
        ))
        
        # Architecture chapter - Good for system design calculations
        self.add_concept(Concept(
            id="transformer_complexity",
            name="Transformer Computational Complexity",
            chapter="dnn_architectures",
            section="transformers",
            bloom_level="analyze",
            prerequisites=["attention_mechanism", "matrix_operations"],
            enables=["efficient_attention", "model_scaling"],
            examples=[
                "Self-attention complexity: O(n²d) where n=sequence_length, d=dimension",
                "Memory requirement: O(n² + nd)"
            ]
        ))
        
        self.add_concept(Concept(
            id="cnn_operations",
            name="CNN Operations Analysis",
            chapter="dnn_architectures",
            section="convolutional_networks",
            bloom_level="apply",
            prerequisites=["convolution", "pooling"],
            enables=["receptive_field", "parameter_sharing"],
            examples=[
                "Conv layer params = (kernel_h * kernel_w * in_channels + 1) * out_channels",
                "Output size = (input_size - kernel_size + 2*padding) / stride + 1"
            ]
        ))
        
        # Hardware Acceleration
        self.add_concept(Concept(
            id="roofline_model",
            name="Roofline Performance Model",
            chapter="hw_acceleration",
            section="performance_analysis",
            bloom_level="evaluate",
            prerequisites=["flops", "memory_bandwidth"],
            enables=["bottleneck_analysis", "optimization_targets"],
            examples=[
                "Arithmetic intensity = FLOPs / bytes_accessed",
                "Peak performance = min(peak_flops, peak_bandwidth * arithmetic_intensity)"
            ]
        ))
        
        # Benchmarking
        self.add_concept(Concept(
            id="latency_throughput",
            name="Latency vs Throughput",
            chapter="benchmarking",
            section="metrics",
            bloom_level="analyze",
            prerequisites=["system_performance"],
            enables=["batch_processing", "pipeline_optimization"],
            examples=[
                "Throughput = batch_size / latency",
                "Effective throughput = (1 - error_rate) * raw_throughput"
            ]
        ))
    
    def add_concept(self, concept: Concept):
        """Add a concept to the knowledge map."""
        self.concepts[concept.id] = concept
    
    def get_prerequisites_for_chapter(self, chapter: str) -> Set[str]:
        """Get all concepts that should be known before this chapter."""
        chapter_idx = self.chapter_order.index(chapter)
        prior_chapters = self.chapter_order[:chapter_idx]
        
        prerequisites = set()
        for concept in self.concepts.values():
            if concept.chapter in prior_chapters:
                prerequisites.add(concept.id)
        
        return prerequisites
    
    def get_available_concepts_for_question(self, 
                                           current_chapter: str,
                                           current_section: str,
                                           include_current: bool = True) -> List[Concept]:
        """Get concepts available for use in questions at this point."""
        chapter_idx = self.chapter_order.index(current_chapter)
        available = []
        
        # Add all concepts from previous chapters
        for concept in self.concepts.values():
            concept_chapter_idx = self.chapter_order.index(concept.chapter)
            
            if concept_chapter_idx < chapter_idx:
                available.append(concept)
            elif concept_chapter_idx == chapter_idx and include_current:
                # Include if it's from an earlier section (simplified - would need section order)
                available.append(concept)
        
        return available
    
    def suggest_calc_questions(self, chapter: str) -> List[Dict]:
        """Suggest CALC questions based on available concepts."""
        available_concepts = self.get_available_concepts_for_question(chapter, "", True)
        
        calc_suggestions = []
        for concept in available_concepts:
            if concept.examples:  # Has calculation examples
                for example in concept.examples:
                    if any(op in example for op in ['=', '*', '/', '+', '-', 'O(']):
                        calc_suggestions.append({
                            "concept": concept.name,
                            "chapter": concept.chapter,
                            "formula": example,
                            "prerequisites": concept.prerequisites,
                            "bloom_level": "apply",  # CALC questions are typically Apply level
                            "suggested_question": self._generate_calc_question(concept, example)
                        })
        
        return calc_suggestions
    
    def _generate_calc_question(self, concept: Concept, formula: str) -> str:
        """Generate a CALC question suggestion based on concept and formula."""
        
        templates = {
            "quantization": "A model with {param_count} float32 parameters is quantized to INT8. Calculate the memory savings and the new model size.",
            "pruning": "After pruning {percent}% of parameters from a {size}MB model, calculate the compressed size and theoretical speedup.",
            "distillation": "A teacher model with {teacher_params}M parameters is distilled to a student with {student_params}M parameters. Calculate the compression ratio and expected latency improvement.",
            "transformer_complexity": "For a transformer processing sequences of length {seq_len} with hidden dimension {dim}, calculate the self-attention memory requirements.",
            "cnn_operations": "Calculate the number of parameters in a Conv2D layer with {in_ch} input channels, {out_ch} output channels, and {k}x{k} kernels.",
            "roofline_model": "Given a kernel with {flops} FLOPs accessing {bytes} bytes of data, and hardware with {peak_flops} GFLOPS and {bandwidth} GB/s bandwidth, determine if it's compute or memory bound.",
            "latency_throughput": "With a model latency of {latency}ms and batch size {batch}, calculate the maximum throughput in samples/second."
        }
        
        # Match concept to template
        for key, template in templates.items():
            if key in concept.id:
                return template
        
        # Default template
        return f"Using the formula: {formula}, calculate the result for a typical ML system scenario."
    
    def generate_scaffolded_questions(self, 
                                     chapter: str,
                                     section: str,
                                     num_questions: int = 5) -> List[Dict]:
        """Generate questions that build on prior knowledge."""
        
        available_concepts = self.get_available_concepts_for_question(chapter, section)
        current_chapter_concepts = [c for c in available_concepts if c.chapter == chapter]
        prior_concepts = [c for c in available_concepts if c.chapter != chapter]
        
        questions = []
        
        # Start with review of prerequisites (Remember/Understand)
        if prior_concepts and num_questions > 3:
            questions.append({
                "type": "prerequisite_review",
                "bloom_level": "remember",
                "concepts": [c.id for c in prior_concepts[:2]],
                "prompt": "Review question connecting to prior knowledge"
            })
        
        # Build up complexity using current chapter concepts
        for i, concept in enumerate(current_chapter_concepts[:num_questions-1]):
            bloom_progression = ["understand", "apply", "analyze", "evaluate", "create"]
            level = bloom_progression[min(i, len(bloom_progression)-1)]
            
            questions.append({
                "type": "progressive",
                "bloom_level": level,
                "concept": concept.id,
                "prerequisites": concept.prerequisites,
                "prompt": f"Question at {level} level for {concept.name}"
            })
        
        # End with synthesis question if appropriate
        if len(questions) < num_questions and len(available_concepts) > 2:
            questions.append({
                "type": "synthesis",
                "bloom_level": "evaluate",
                "concepts": [c.id for c in available_concepts[-3:]],
                "prompt": "Synthesis question combining multiple concepts"
            })
        
        return questions

    def export_for_prompt(self, chapter: str) -> str:
        """Export knowledge map context for LLM prompt."""
        prerequisites = self.get_prerequisites_for_chapter(chapter)
        available = self.get_available_concepts_for_question(chapter, "")
        
        prompt = f"""
## Knowledge Map Context

### Prior Knowledge (from previous chapters):
Students have already learned:
"""
        for concept_id in prerequisites:
            concept = self.concepts[concept_id]
            prompt += f"- {concept.name} ({concept.chapter}): Bloom's level - {concept.bloom_level}\n"
            if concept.examples:
                prompt += f"  Examples: {concept.examples[0]}\n"
        
        prompt += """

### Available for Integration:
These concepts can be referenced or built upon:
"""
        for concept in available[:10]:  # Limit to avoid prompt bloat
            if concept.examples and any(op in concept.examples[0] for op in ['=', '*', '/', '+', '-']):
                prompt += f"- {concept.name}: {concept.examples[0]}\n"
        
        prompt += """

### Question Generation Guidelines:
1. Review questions can reference prior knowledge
2. New concepts should build on prerequisites
3. Synthesis questions should integrate multiple concepts
4. CALC questions should use formulas from the knowledge map when possible
"""
        
        return prompt


# Example usage for generating prompts
def enhance_prompt_with_knowledge_map(base_prompt: str, 
                                     chapter: str,
                                     section: str) -> str:
    """Enhance a quiz generation prompt with knowledge map context."""
    
    km = KnowledgeMap()
    
    # Add knowledge map context
    knowledge_context = km.export_for_prompt(chapter)
    
    # Get CALC suggestions for technical chapters
    calc_suggestions = ""
    if chapter in ["optimizations", "hw_acceleration", "benchmarking", "training"]:
        suggestions = km.suggest_calc_questions(chapter)
        if suggestions:
            calc_suggestions = "\n### Suggested CALC Questions:\n"
            for s in suggestions[:3]:  # Top 3 suggestions
                calc_suggestions += f"- {s['suggested_question']}\n"
                calc_suggestions += f"  Formula: {s['formula']}\n"
    
    # Get scaffolding structure
    scaffolding = km.generate_scaffolded_questions(chapter, section)
    scaffolding_guide = "\n### Question Progression:\n"
    for i, q in enumerate(scaffolding, 1):
        scaffolding_guide += f"{i}. {q['bloom_level'].capitalize()} level - {q['type']}\n"
    
    # Combine everything
    enhanced_prompt = f"""{base_prompt}

{knowledge_context}

{calc_suggestions}

{scaffolding_guide}

Remember: Students have progressively built knowledge from chapter 1 to {chapter}.
Questions should acknowledge and build upon this foundation.
"""
    
    return enhanced_prompt


if __name__ == "__main__":
    # Test the knowledge map
    km = KnowledgeMap()
    
    # Test for Optimizations chapter
    print("=== Optimizations Chapter ===")
    print("\nPrerequisites:")
    prereqs = km.get_prerequisites_for_chapter("optimizations")
    for p in list(prereqs)[:5]:
        print(f"  - {p}")
    
    print("\nCALC Question Suggestions:")
    calc_questions = km.suggest_calc_questions("optimizations")
    for q in calc_questions[:3]:
        print(f"  - {q['suggested_question']}")
        print(f"    Formula: {q['formula']}")
    
    print("\nScaffolded Question Structure:")
    scaffolding = km.generate_scaffolded_questions("optimizations", "quantization")
    for i, s in enumerate(scaffolding, 1):
        print(f"  {i}. {s['bloom_level']} - {s['type']}")
    
    # Test for Architecture chapter  
    print("\n=== Architecture Chapter ===")
    calc_questions = km.suggest_calc_questions("dnn_architectures")
    for q in calc_questions[:2]:
        print(f"  - {q['suggested_question']}")