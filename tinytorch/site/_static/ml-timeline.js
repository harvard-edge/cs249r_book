/**
 * Interactive ML History Timeline
 * Handles popup functionality for milestone cards
 */

document.addEventListener('DOMContentLoaded', function() {
    const timelineData = {
        perceptron: {
            year: "1957",
            title: "The Perceptron",
            researcher: "Frank Rosenblatt",
            subtitle: "The first trainable neural network proves machines can learn from data",
            achievement: "Binary classification with gradient descent",
            architecture: "Input → Linear → Sigmoid → Output",
            whatYouBuild: [
                "Binary classification with gradient descent",
                "Simple but revolutionary architecture",
                "YOUR Linear layer recreates history"
            ],
            systemsInsights: [
                "Memory: O(n) parameters",
                "Compute: O(n) operations",
                "Limitation: Only linearly separable problems"
            ],
            modules: "After Modules 02-04",
            expectedResults: "~50% (untrained) → 95%+ (trained) accuracy",
            commands: [
                "cd milestones/01_1957_perceptron",
                "python 01_rosenblatt_forward.py   # See the problem (random weights)",
                "python 02_rosenblatt_trained.py   # See the solution (trained)"
            ]
        },
        xor: {
            year: "1969",
            title: "The XOR Crisis",
            researcher: "Minsky & Papert",
            subtitle: "Hidden layers solve non-linear problems that nearly ended AI research",
            achievement: "Non-linear learning through hidden representations",
            architecture: "Input → Linear → ReLU → Linear → Output",
            whatYouBuild: [
                "Hidden layers enable non-linear solutions",
                "Multi-layer networks break through limitations",
                "YOUR autograd makes it possible"
            ],
            systemsInsights: [
                "Memory: O(n²) with hidden layers",
                "Compute: O(n²) operations",
                "Breakthrough: Hidden representations"
            ],
            modules: "After Modules 02-06",
            expectedResults: "50% (single layer) → 100% (multi-layer) on XOR",
            commands: [
                "cd milestones/02_1969_xor",
                "python 01_xor_crisis.py   # Watch it fail (loss stuck at 0.69)",
                "python 02_xor_solved.py   # Hidden layers solve it!"
            ]
        },
        mlp: {
            year: "1986",
            title: "MLP Revival",
            researcher: "Backpropagation Era",
            subtitle: "Backpropagation enables training deep networks on real datasets",
            achievement: "Multi-class digit recognition",
            architecture: "Images → Flatten → Linear → ReLU → Linear → ReLU → Linear → Classes",
            whatYouBuild: [
                "Multi-class digit recognition",
                "Complete training pipelines",
                "YOUR optimizers achieve 95%+ accuracy"
            ],
            systemsInsights: [
                "Memory: ~100K parameters for MNIST",
                "Compute: Dense matrix operations",
                "Architecture: Multi-layer feature learning"
            ],
            modules: "After Modules 02-08",
            expectedResults: "95%+ accuracy on MNIST",
            commands: [
                "cd milestones/03_1986_mlp",
                "python 01_rumelhart_tinydigits.py  # 8x8 digits (quick)",
                "python 02_rumelhart_mnist.py       # Full MNIST"
            ]
        },
        cnn: {
            year: "1998",
            title: "CNN Revolution",
            researcher: "Yann LeCun",
            subtitle: "CNNs exploit spatial structure for computer vision—enabling modern AI",
            achievement: "Spatial intelligence for computer vision",
            architecture: "Images → Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → Linear → Classes",
            whatYouBuild: [
                "Convolutional feature extraction",
                "Natural image classification (CIFAR-10)",
                "YOUR Conv2d + MaxPool2d unlock spatial intelligence"
            ],
            systemsInsights: [
                "Memory: ~1M parameters (weight sharing reduces vs dense)",
                "Compute: Convolution is intensive but parallelizable",
                "Architecture: Local connectivity + translation invariance"
            ],
            modules: "After Modules 02-09",
            expectedResults: "75%+ accuracy on CIFAR-10 ✨",
            commands: [
                "cd milestones/04_1998_cnn",
                "python 01_lecun_tinydigits.py  # Spatial features on digits",
                "python 02_lecun_cifar10.py     # CIFAR-10 @ 75%+ accuracy"
            ],
            northStar: true
        },
        transformer: {
            year: "2017",
            title: "Transformer Era",
            researcher: "Vaswani et al.",
            subtitle: "Attention mechanism launches the LLM revolution (GPT, BERT, ChatGPT)",
            achievement: "Self-attention for language understanding",
            architecture: "Tokens → Embeddings → Attention → FFN → ... → Attention → Output",
            whatYouBuild: [
                "Self-attention mechanisms",
                "Autoregressive text generation",
                "YOUR attention implementation generates language"
            ],
            systemsInsights: [
                "Memory: O(n²) attention requires careful management",
                "Compute: Highly parallelizable",
                "Architecture: Long-range dependencies"
            ],
            modules: "After Modules 02-13",
            expectedResults: "Loss < 1.5, coherent responses to questions",
            commands: [
                "cd milestones/05_2017_transformer",
                "python 01_vaswani_generation.py  # Q&A generation with TinyTalks",
                "python 02_vaswani_dialogue.py    # Multi-turn dialogue"
            ]
        },
        olympics: {
            year: "2018",
            title: "MLPerf Torch Olympics",
            researcher: "MLCommons (founded 2018)",
            subtitle: "Systematic optimization becomes essential as models grow larger",
            achievement: "Production-ready optimization",
            architecture: "Profile → Compress → Accelerate",
            whatYouBuild: [
                "Performance profiling and bottleneck analysis",
                "Model compression (quantization + pruning)",
                "Inference acceleration (KV-cache + batching)"
            ],
            systemsInsights: [
                "Memory: 4-16× compression through quantization/pruning",
                "Speed: 12-40× faster generation with KV-cache + batching",
                "Workflow: Systematic 'measure → optimize → validate' methodology"
            ],
            modules: "After Modules 14-18",
            expectedResults: "8-16× smaller models, 12-40× faster inference",
            commands: [
                "cd milestones/06_2018_mlperf",
                "python 01_baseline_profile.py   # Find bottlenecks",
                "python 02_compression.py         # Reduce size (quantize + prune)",
                "python 03_generation_opts.py    # Speed up inference (cache + batch)"
            ]
        }
    };

    // Create popup HTML if not exists
    let popup = document.getElementById('ml-timeline-popup');
    if (!popup) {
        popup = document.createElement('div');
        popup.id = 'ml-timeline-popup';
        popup.className = 'ml-timeline-popup';
        popup.innerHTML = '<div class="ml-timeline-popup-content"></div>';
        document.body.appendChild(popup);
    }

    // Handle clicks on timeline items
    document.querySelectorAll('.ml-timeline-content').forEach(card => {
        card.addEventListener('click', function(e) {
            const item = this.closest('.ml-timeline-item');
            const milestoneType = item.classList[1]; // Get the milestone class (perceptron, xor, etc.)
            const data = timelineData[milestoneType];

            if (!data) return;

            const popupContent = popup.querySelector('.ml-timeline-popup-content');
            popupContent.innerHTML = `
                <button class="ml-timeline-popup-close" aria-label="Close">&times;</button>

                <h3>
                    <span class="ml-timeline-popup-year">${data.year}</span>
                    ${data.title}
                </h3>

                <p class="ml-timeline-popup-subtitle">${data.subtitle}</p>

                ${data.northStar ? '<div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; border-left: 4px solid #f59e0b;"><strong>🎯 North Star Achievement</strong> — This is a major milestone in your TinyTorch journey!</div>' : ''}

                <div class="ml-timeline-popup-section">
                    <h4>The ${data.researcher} Breakthrough</h4>
                    <p><strong>${data.achievement}</strong></p>
                </div>

                <div class="ml-timeline-popup-section">
                    <h4>Architecture</h4>
                    <div class="ml-timeline-popup-code">${data.architecture}</div>
                </div>

                <div class="ml-timeline-popup-section">
                    <h4>What You'll Build</h4>
                    <ul>
                        ${data.whatYouBuild.map(item => `<li>${item}</li>`).join('')}
                    </ul>
                </div>

                <div class="ml-timeline-popup-section">
                    <h4>Systems Insights</h4>
                    <ul>
                        ${data.systemsInsights.map(item => `<li>${item}</li>`).join('')}
                    </ul>
                </div>

                <div class="ml-timeline-popup-metrics">
                    <div class="ml-timeline-popup-metric">
                        <div class="ml-timeline-popup-metric-label">Prerequisites</div>
                        <div class="ml-timeline-popup-metric-value">${data.modules}</div>
                    </div>
                    <div class="ml-timeline-popup-metric">
                        <div class="ml-timeline-popup-metric-label">Expected Results</div>
                        <div class="ml-timeline-popup-metric-value">${data.expectedResults}</div>
                    </div>
                </div>

                <div class="ml-timeline-popup-section">
                    <h4>Try It Yourself</h4>
                    <div class="ml-timeline-popup-code">${data.commands.join('\n')}</div>
                </div>
            `;

            popup.classList.add('active');

            // Close button handler
            const closeBtn = popupContent.querySelector('.ml-timeline-popup-close');
            closeBtn.addEventListener('click', function() {
                popup.classList.remove('active');
            });
        });
    });

    // Close popup on background click
    popup.addEventListener('click', function(e) {
        if (e.target === popup) {
            popup.classList.remove('active');
        }
    });

    // Close popup on ESC key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && popup.classList.contains('active')) {
            popup.classList.remove('active');
        }
    });
});
