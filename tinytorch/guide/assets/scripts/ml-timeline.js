<script>
/**
 * Interactive ML History Timeline (Quarto version)
 * Handles popup functionality for milestone cards
 */
document.addEventListener('DOMContentLoaded', function() {
    const timelineData = {
        perceptron: {
            year: "1958", title: "The Perceptron", researcher: "Frank Rosenblatt",
            subtitle: "The first neural network architecture starts the field",
            achievement: "Binary classification with a forward pass",
            architecture: "Input → Linear → Sigmoid → Output",
            whatYouBuild: ["Binary classification with random weights", "Simple but revolutionary architecture", "YOUR Linear layer recreates history"],
            systemsInsights: ["Memory: O(n) parameters", "Compute: O(n) operations", "Limitation: Only linearly separable problems"],
            modules: "After Modules 01-03", expectedResults: "~50% with untrained random weights",
            commands: ["tito milestone run perceptron"],
            url: "milestones/01_perceptron.html"
        },
        xor: {
            year: "1969", title: "The XOR Crisis", researcher: "Minsky & Papert",
            subtitle: "Hidden layers solve non-linear problems that nearly ended AI research",
            achievement: "Non-linear learning through hidden representations",
            architecture: "Input → Linear → ReLU → Linear → Output",
            whatYouBuild: ["Hidden layers enable non-linear solutions", "Multi-layer networks break through limitations", "YOUR autograd makes it possible"],
            systemsInsights: ["Memory: O(n²) with hidden layers", "Compute: O(n²) operations", "Breakthrough: Hidden representations"],
            modules: "After Modules 02-06", expectedResults: "50% (single layer) → 100% (multi-layer) on XOR",
            commands: ["tito milestone run xor"],
            url: "milestones/02_xor.html"
        },
        mlp: {
            year: "1986", title: "MLP Revival", researcher: "Backpropagation Era",
            subtitle: "Backpropagation enables training deep networks on real datasets",
            achievement: "Multi-class digit recognition",
            architecture: "Images → Flatten → Linear → ReLU → Linear → ReLU → Linear → Classes",
            whatYouBuild: ["XOR solved with hidden layers", "TinyDigits classification", "YOUR training pipeline learns real images"],
            systemsInsights: ["Memory: ~2.4K parameters for TinyDigits", "Compute: Dense matrix operations", "Architecture: Multi-layer feature learning"],
            modules: "After Modules 01-08", expectedResults: "Solves XOR, 85%+ on TinyDigits",
            commands: ["tito milestone run mlp"],
            url: "milestones/03_mlp.html"
        },
        cnn: {
            year: "1998", title: "CNN Revolution", researcher: "Yann LeCun",
            subtitle: "CNNs exploit spatial structure for computer vision—enabling modern AI",
            achievement: "Spatial intelligence for computer vision",
            architecture: "Images → Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → Linear → Classes",
            whatYouBuild: ["Convolutional feature extraction", "Natural image classification (CIFAR-10)", "YOUR Conv2d + MaxPool2d unlock spatial intelligence"],
            systemsInsights: ["Memory: ~1M parameters (weight sharing reduces vs dense)", "Compute: Convolution is intensive but parallelizable", "Architecture: Local connectivity + translation invariance"],
            modules: "After Modules 02-09", expectedResults: "75%+ accuracy on CIFAR-10 ✨",
            commands: ["tito milestone run cnn"], northStar: true,
            url: "milestones/04_cnn.html"
        },
        transformer: {
            year: "2017–2022", title: "Transformer Era (TinyGPT)", researcher: "Vaswani et al. & Brown et al.",
            subtitle: "Self-attention and autoregressive language modeling power modern generative AI (ChatGPT)",
            achievement: "Generative LLM trained from scratch on Shakespeare",
            architecture: "Tokens → Embeddings → Causal MHA + MLP → ... → LM Head → Autoregressive Sampling",
            whatYouBuild: ["Causal multi-head self-attention with lower-triangular masking", "Pre-LN transformer decoder blocks", "YOUR complete stack trains from scratch and writes Shakespeare"],
            systemsInsights: ["Pretraining: Teacher forcing allows O(1) parallel sequence loss calculation", "Inference: Generation is memory-bandwidth bound, demanding KV caching (Module 18)", "Convergence: AdamW with weight decay enables stable deep transformer training"],
            modules: "After Modules 01-08, 10-13", expectedResults: "Cross-entropy loss < 1.0, coherent Shakespeare text generated",
            commands: ["tito milestone run 05", "tito milestone run transformer", "tito milestone run tinygpt"],
            url: "milestones/05_transformer.html"
        },
        olympics: {
            year: "2018–Present", title: "MLPerf Benchmarks", researcher: "MLCommons (founded 2018)",
            subtitle: "Systematic optimization becomes essential as models grow larger",
            achievement: "Production-ready optimization & KV-cache acceleration",
            architecture: "Profile → Compress → Accelerate",
            whatYouBuild: ["Performance profiling and bottleneck analysis", "Model compression (quantization + pruning)", "Inference acceleration (KV-cache + batching)"],
            systemsInsights: ["Memory: quantization packs weights up to 4× smaller; pruning zeroes weights without shrinking the array", "Speed: KV-cache speeds up autoregressive generation by eliminating O(S²) prefix recomputation", "Workflow: Systematic 'measure → optimize → validate' methodology"],
            modules: "After Modules 14-19", expectedResults: "Quantization 4× compression, measurable KV-cache speedup",
            commands: ["tito milestone run 06", "tito milestone run mlperf"],
            url: "milestones/06_mlperf.html"
        }
    };

    // Nothing to pop up on a page with no timeline, and appending the overlay
    // anyway put a full-viewport element on 52 of the 62 pages.
    const timelineCards = document.querySelectorAll('.ml-timeline-content');
    if (timelineCards.length === 0) return;

    // Create popup element
    let popup = document.getElementById('ml-timeline-popup');
    if (!popup) {
        popup = document.createElement('div');
        popup.id = 'ml-timeline-popup';
        popup.className = 'ml-timeline-popup-overlay';
        popup.innerHTML = '<div class="ml-timeline-popup"></div>';
        document.body.appendChild(popup);
    }

    // Handle clicks on timeline items
    timelineCards.forEach(card => {
        card.style.cursor = 'pointer';
        card.addEventListener('click', function() {
            const item = this.closest('.ml-timeline-item');
            // Get milestone type from class list (second class after 'ml-timeline-item' and 'left'/'right')
            const classes = Array.from(item.classList);
            const milestoneType = classes.find(c => !['ml-timeline-item', 'left', 'right'].includes(c));
            const data = timelineData[milestoneType];
            if (!data) return;

            const popupInner = popup.querySelector('.ml-timeline-popup');
            popupInner.innerHTML = `
                <button class="ml-timeline-popup-close" aria-label="Close">&times;</button>
                <h3><span style="color: #f97316; font-size: 0.85rem; display: block; margin-bottom: 0.25rem;">${data.year}</span>${data.title}</h3>
                <p style="color: #64748b; margin: 0.5rem 0 1rem 0;">${data.subtitle}</p>
                ${data.northStar ? '<div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; border-left: 4px solid #f59e0b;"><strong>🎯 North Star Achievement</strong> — This is a major milestone in your TinyTorch journey!</div>' : ''}
                <div style="margin: 1rem 0;"><h4 style="font-size: 0.95rem; margin: 0 0 0.5rem 0;">The ${data.researcher} Breakthrough</h4><p><strong>${data.achievement}</strong></p></div>
                <div style="margin: 1rem 0;"><h4 style="font-size: 0.95rem; margin: 0 0 0.5rem 0;">Architecture</h4><div style="background: #f8fafc; padding: 0.75rem; border-radius: 0.375rem; font-family: monospace; font-size: 0.85rem;">${data.architecture}</div></div>
                <div style="margin: 1rem 0;"><h4 style="font-size: 0.95rem; margin: 0 0 0.5rem 0;">What You'll Build</h4><ul>${data.whatYouBuild.map(i => `<li>${i}</li>`).join('')}</ul></div>
                <div style="margin: 1rem 0;"><h4 style="font-size: 0.95rem; margin: 0 0 0.5rem 0;">Systems Insights</h4><ul>${data.systemsInsights.map(i => `<li>${i}</li>`).join('')}</ul></div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0;">
                    <div style="background: #f8fafc; padding: 0.75rem; border-radius: 0.375rem;"><div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.25rem;">Prerequisites</div><div style="font-weight: 600;">${data.modules}</div></div>
                    <div style="background: #f8fafc; padding: 0.75rem; border-radius: 0.375rem;"><div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.25rem;">Expected Results</div><div style="font-weight: 600;">${data.expectedResults}</div></div>
                </div>
                <div style="margin: 1rem 0;"><h4 style="font-size: 0.95rem; margin: 0 0 0.5rem 0;">Try It Yourself</h4><div style="background: #f8fafc; padding: 0.75rem; border-radius: 0.375rem; font-family: monospace; font-size: 0.85rem;">${data.commands.join('\n')}</div></div>
                ${data.url ? `<div style="margin: 1.25rem 0 0.25rem 0; text-align: center;"><a href="${data.url}" class="btn-tinytorch" style="display: inline-block; padding: 0.6rem 1.5rem; text-decoration: none; font-size: 0.9rem; font-weight: 600;">Read Milestone Guide →</a></div>` : ''}
            `;

            popup.classList.add('active');
            popup.style.display = 'flex';

            popup.querySelector('.ml-timeline-popup-close').addEventListener('click', () => {
                popup.classList.remove('active');
                popup.style.display = 'none';
            });
        });
    });

    // Close on background click
    popup.addEventListener('click', function(e) { if (e.target === popup) { popup.classList.remove('active'); popup.style.display = 'none'; } });
    // Close on ESC
    document.addEventListener('keydown', function(e) { if (e.key === 'Escape' && popup.classList.contains('active')) { popup.classList.remove('active'); popup.style.display = 'none'; } });
});
</script>
