# Test Cases for LLM Cross-Reference Explanation Optimization
# 
# This module contains realistic test cases extracted from the ML Systems book
# to systematically evaluate different LLM models and explanation lengths.

TEST_CASES = [
    # Case 1: Introduction to Deep Learning Primer
    {
        "id": "intro_to_dl_primer",
        "source_title": "AI Pervasiveness", 
        "source_content": "Artificial Intelligence (AI) has emerged as one of the most transformative forces in human history. From the moment we wake up to when we go to sleep, AI systems invisibly shape our world. They manage traffic flows in our cities, optimize power distribution across electrical grids, and enable billions of wireless devices to communicate seamlessly.",
        "target_title": "Biological to Artificial Neurons",
        "target_content": "The human brain contains approximately 86 billion neurons, each forming thousands of connections with other neurons. These biological neural networks process information through electrical and chemical signals, enabling everything from basic reflexes to complex reasoning. Understanding how biological neurons work provides crucial insights for designing artificial neural networks.",
        "connection_type": "Preview",
        "domain": "neural_networks",
        "difficulty": "introductory"
    },
    
    # Case 2: Training to Hardware Acceleration  
    {
        "id": "training_to_hw_accel",
        "source_title": "Distributed Training",
        "source_content": "Training large neural networks requires distributing computation across multiple devices and machines. Data parallelism splits the training data across workers, while model parallelism splits the model itself. Pipeline parallelism divides the model into stages that process different parts of the input simultaneously.",
        "target_title": "GPU Architecture and Optimization", 
        "target_content": "Graphics Processing Units (GPUs) excel at parallel computation through thousands of cores designed for simultaneous execution. Modern GPUs feature specialized tensor cores for AI workloads, high-bandwidth memory systems, and sophisticated caching hierarchies optimized for the matrix operations common in neural networks.",
        "connection_type": "Preview",
        "domain": "systems_architecture", 
        "difficulty": "intermediate"
    },
    
    # Case 3: Robustness to Privacy
    {
        "id": "robustness_to_privacy", 
        "source_title": "Adversarial Attacks",
        "source_content": "Adversarial examples are inputs specifically crafted to fool machine learning models into making incorrect predictions. These attacks exploit the high-dimensional nature of input spaces and the complex decision boundaries learned by neural networks. Even imperceptible perturbations can cause dramatic misclassifications.",
        "target_title": "Differential Privacy in ML",
        "target_content": "Differential privacy provides mathematical guarantees about the privacy of individual data points in a dataset. By adding carefully calibrated noise to training processes or model outputs, differential privacy ensures that the presence or absence of any single individual's data cannot be reliably detected from the model's behavior.",
        "connection_type": "Preview",
        "domain": "ml_safety",
        "difficulty": "advanced"  
    },
    
    # Case 4: Frameworks to Deployment
    {
        "id": "frameworks_to_deployment",
        "source_title": "PyTorch vs TensorFlow",
        "source_content": "PyTorch and TensorFlow represent two dominant paradigms in deep learning frameworks. PyTorch emphasizes dynamic computation graphs and intuitive debugging, making it popular for research. TensorFlow focuses on production deployment with static graphs and comprehensive ecosystem tools.",
        "target_title": "Model Serving and MLOps",
        "target_content": "Deploying machine learning models in production requires careful consideration of latency, throughput, scalability, and reliability. Model serving systems must handle version management, A/B testing, monitoring, and rollback capabilities while maintaining consistent performance under varying loads.",
        "connection_type": "Preview", 
        "domain": "ml_engineering",
        "difficulty": "intermediate"
    },
    
    # Case 5: Backward Reference - Optimization to Training
    {
        "id": "optimization_to_training",
        "source_title": "Model Compression Techniques",
        "source_content": "Model compression reduces the memory footprint and computational requirements of neural networks through techniques like pruning, quantization, and knowledge distillation. These methods enable deployment on resource-constrained devices while maintaining acceptable accuracy.",
        "target_title": "Training Fundamentals", 
        "target_content": "Neural network training involves iteratively adjusting model parameters to minimize a loss function. The process requires careful management of learning rates, batch sizes, regularization techniques, and optimization algorithms like SGD or Adam to achieve convergence on training data while generalizing to unseen examples.",
        "connection_type": "Background",
        "domain": "ml_fundamentals",
        "difficulty": "foundational"
    },
    
    # Case 6: Complex Technical Connection
    {
        "id": "complex_technical",
        "source_title": "Transformer Architecture Details",
        "source_content": "The Transformer architecture revolutionized natural language processing through self-attention mechanisms that capture long-range dependencies without recurrent connections. Multi-head attention allows the model to attend to different representation subspaces simultaneously, while positional encodings provide sequence order information.",
        "target_title": "Efficient Attention Mechanisms",
        "target_content": "Standard attention has quadratic complexity in sequence length, creating computational bottlenecks for long sequences. Efficient attention variants like linear attention, sparse attention, and sliding window attention reduce this complexity while attempting to preserve the representational power of full attention.",
        "connection_type": "Preview",
        "domain": "deep_learning_architectures", 
        "difficulty": "advanced"
    },
    
    # Case 7: Practical Application Connection
    {
        "id": "practical_application", 
        "source_title": "Edge Computing Constraints",
        "source_content": "Edge devices face severe constraints in memory, computational power, and energy consumption. These limitations require specialized techniques for model deployment including quantization, pruning, and efficient neural architecture design tailored for mobile and embedded processors.",
        "target_title": "Real-world Deployment Challenges",
        "target_content": "Deploying ML systems in production environments involves handling data drift, monitoring model performance, managing infrastructure costs, ensuring regulatory compliance, and maintaining system reliability. These challenges require comprehensive MLOps practices and continuous monitoring systems.",
        "connection_type": "Preview",
        "domain": "practical_ml",
        "difficulty": "intermediate"
    },
    
    # Case 8: Short content test
    {
        "id": "short_content_test",
        "source_title": "CNN Basics",
        "source_content": "Convolutional Neural Networks use filters to detect local patterns in images through convolution operations.",
        "target_title": "Image Classification Applications", 
        "target_content": "Image classification systems identify and categorize objects in images using trained neural networks.",
        "connection_type": "Preview",
        "domain": "computer_vision",
        "difficulty": "introductory"
    }
]

# Evaluation criteria for explanations
EVALUATION_CRITERIA = {
    "relevance": "How well does the explanation capture the actual relationship between the sections?",
    "clarity": "Is the explanation clear and easy to understand for the target audience?", 
    "conciseness": "Is the explanation appropriately concise without being too brief or verbose?",
    "usefulness": "Would this explanation actually help a reader decide to follow the cross-reference?",
    "accuracy": "Is the explanation factually correct about the content domains?",
    "uniqueness": "Does the explanation add value beyond just restating the section titles?"
}

# Target explanation lengths to test
LENGTH_TARGETS = [
    {"min_words": 3, "max_words": 5, "description": "ultra_short"},
    {"min_words": 4, "max_words": 7, "description": "short"}, 
    {"min_words": 6, "max_words": 10, "description": "medium"},
    {"min_words": 8, "max_words": 12, "description": "standard"},
    {"min_words": 10, "max_words": 15, "description": "extended"}
]

def get_test_case_by_id(test_id: str):
    """Get a specific test case by ID"""
    for case in TEST_CASES:
        if case["id"] == test_id:
            return case
    return None

def get_test_cases_by_domain(domain: str):
    """Get all test cases for a specific domain"""
    return [case for case in TEST_CASES if case["domain"] == domain]

def get_test_cases_by_difficulty(difficulty: str):
    """Get all test cases for a specific difficulty level"""
    return [case for case in TEST_CASES if case["difficulty"] == difficulty] 