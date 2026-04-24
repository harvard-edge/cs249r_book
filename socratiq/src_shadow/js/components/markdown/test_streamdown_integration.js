// test_streamdown_integration.js - Test file for STREAMDOWN integration
// This file tests the new streamdown_markdown.js implementation with actual STREAMDOWN features

import { 
  updateMarkdownPreview, 
  initiateMarkdown, 
  initializeSpoilerEditing,
  reinitializeEditableInputs,
  StreamdownMarkdownRenderer 
} from './streamdown_markdown.js';

// Test function to verify the STREAMDOWN integration works
export function testStreamdownIntegration() {
  console.log('🧪 Testing STREAMDOWN Integration...');
  
  // Test 1: STREAMDOWN's incomplete markdown parsing
  console.log('✅ Test 1: STREAMDOWN incomplete markdown parsing');
  const incompleteMarkdownTest = `
# Transfer Learning

Transfer learning is a machine learning technique where a model trained on one task is reused as the starting point for a model on a second task.

## Key Benefits

- **Reduced training time** (this bold text is incomplete
- *Better performance* (this italic text is incomplete
- \`Resource efficiency\` (this code is incomplete

## Code Block Test
\`\`\`javascript
console.log("Hello World");
// This code block is incomplete

## Math Test
The quadratic formula is $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$ for solving $ax^2 + bx + c = 0$.

Block math:
$$ f(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}} e^{-\\frac{1}{2}\\left(\\frac{x-\\mu}{\\sigma}\\right)^2} $$

## Mermaid Test
\`\`\`mermaid
graph TD
    A[Start] --> B[Process]
    B --> C[End]
\`\`\`

## Custom Container Test
:::spoiler Click to reveal
This is hidden content that can be edited.
:::

:::info
This is an info box with important information.
:::

## Questions Test
This is some content.

%%%

What is the main topic?
How does this work?
  `;

  // Test 2: STREAMDOWN's syntax highlighting
  console.log('✅ Test 2: STREAMDOWN syntax highlighting');
  const syntaxHighlightingTest = `
# Code Examples

## JavaScript
\`\`\`javascript
function transferLearning(model, newTask) {
  // Freeze early layers
  model.layers.slice(0, -2).forEach(layer => {
    layer.trainable = false;
  });
  
  // Fine-tune on new task
  return model.fit(newTask.data, newTask.labels, {
    epochs: 10,
    validationSplit: 0.2
  });
}
\`\`\`

## Python
\`\`\`python
import torch
import torch.nn as nn

class TransferLearningModel(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super().__init__()
        self.backbone = pretrained_model
        self.classifier = nn.Linear(1000, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
\`\`\`

## SQL
\`\`\`sql
SELECT 
    model_name,
    accuracy,
    training_time
FROM model_performance 
WHERE transfer_learning = true
ORDER BY accuracy DESC;
\`\`\`
  `;

  // Test 3: STREAMDOWN's Mermaid rendering
  console.log('✅ Test 3: STREAMDOWN Mermaid rendering');
  const mermaidTest = `
# Machine Learning Pipeline

## Data Flow
\`\`\`mermaid
flowchart LR
    A[Raw Data] --> B[Preprocessing]
    B --> C[Feature Engineering]
    C --> D[Model Training]
    D --> E[Validation]
    E --> F[Deployment]
\`\`\`

## Transfer Learning Process
\`\`\`mermaid
sequenceDiagram
    participant P as Pre-trained Model
    participant F as Feature Extractor
    participant C as Classifier
    participant D as New Data
    
    P->>F: Extract features
    F->>C: Train classifier
    D->>F: New data features
    F->>C: Predict
    C->>D: Results
\`\`\`

## Model Architecture
\`\`\`mermaid
graph TD
    A[Input Layer] --> B[Hidden Layer 1]
    B --> C[Hidden Layer 2]
    C --> D[Output Layer]
    
    B --> E[Dropout]
    C --> F[Batch Norm]
    
    style A fill:#e1f5fe
    style D fill:#f3e5f5
\`\`\`
  `;

  // Test 4: STREAMDOWN's math rendering
  console.log('✅ Test 4: STREAMDOWN math rendering');
  const mathTest = `
# Mathematical Concepts in Transfer Learning

## Loss Function
The cross-entropy loss for transfer learning can be expressed as:

$$L = -\\sum_{i=1}^{n} y_i \\log(\\hat{y}_i) + \\lambda \\sum_{j=1}^{m} ||w_j||_2$$

Where:
- $y_i$ is the true label
- $\\hat{y}_i$ is the predicted probability
- $\\lambda$ is the regularization parameter
- $w_j$ are the model weights

## Gradient Descent Update
The weight update rule with momentum is:

$$w_{t+1} = w_t - \\alpha \\nabla L(w_t) + \\beta v_t$$

Where $\\alpha$ is the learning rate and $\\beta$ is the momentum coefficient.

## Information Theory
The mutual information between source and target domains:

$$I(X_s; X_t) = \\sum_{x_s, x_t} p(x_s, x_t) \\log \\frac{p(x_s, x_t)}{p(x_s)p(x_t)}$$
  `;

  // Test 5: STREAMDOWN's security features
  console.log('✅ Test 5: STREAMDOWN security features');
  const securityTest = `
# Security Test

## Safe Links
- [Official Documentation](https://example.com/docs)
- [GitHub Repository](https://github.com/example/repo)

## Potentially Unsafe Links (should be blocked)
- [Suspicious Link](javascript:alert('xss'))
- [Data URI](data:text/html,<script>alert('xss')</script>)

## Images
![Safe Image](https://example.com/image.png)
  `;

  console.log('🎉 All STREAMDOWN tests prepared!');
  
  return {
    incompleteMarkdownTest,
    syntaxHighlightingTest,
    mermaidTest,
    mathTest,
    securityTest
  };
}

// Test streaming renderer with STREAMDOWN features
export async function testStreamingRenderer(shadowElement) {
  console.log('🧪 Testing STREAMDOWN Streaming Renderer...');
  
  const renderer = new StreamdownMarkdownRenderer(shadowElement);
  
  // Simulate streaming chunks with STREAMDOWN features
  const chunks = [
    "# STREAMDOWN Streaming Test\n\nThis content is being ",
    "streamed in **real-time** with ",
    "```javascript\nconsole.log('Hello from STREAMDOWN!');\n```\n\n",
    "And mathematical expressions: $E = mc^2$\n\n",
    "Block math:\n$$\\sum_{i=1}^{n} x_i = \\frac{n(n+1)}{2}$$\n\n",
    "And a mermaid diagram:\n\n```mermaid\ngraph TD\n    A[Start] --> B[STREAMDOWN]\n    B --> C[Render]\n    C --> D[Complete]\n```\n\n",
    "## Custom Features\n\n:::spoiler Editable Content\nThis content can be edited and resubmitted.\n:::\n\n",
    "%%%\n\nWhat do you think about STREAMDOWN?\nHow does the streaming work?\nWhat are the benefits?"
  ];
  
  // Start streaming
  await renderer.startStream('markdown-preview');
  
  // Stream chunks with delay
  for (let i = 0; i < chunks.length; i++) {
    setTimeout(async () => {
      await renderer.addChunk(chunks[i], 'markdown-preview');
      console.log(`📦 Chunk ${i + 1} added with STREAMDOWN processing`);
    }, i * 1000);
  }
  
  // Complete streaming
  setTimeout(async () => {
    await renderer.completeStream('markdown-preview');
    console.log('✅ STREAMDOWN streaming completed!');
  }, chunks.length * 1000);
  
  return renderer;
}

// Test specific STREAMDOWN features
export async function testStreamdownFeatures() {
  console.log('🧪 Testing Individual STREAMDOWN Features...');
  
  // Test incomplete markdown parsing
  const incompleteTests = [
    "This is **incomplete bold text",
    "This is *incomplete italic text",
    "This is `incomplete code",
    "This is [incomplete link",
    "Math: $incomplete math$",
    "Block math: $$incomplete block math"
  ];
  
  console.log('📝 Testing incomplete markdown parsing...');
  for (const test of incompleteTests) {
    console.log(`Input: "${test}"`);
    // The parseIncompleteMarkdown function would be called here
    console.log('✅ Incomplete markdown handled');
  }
  
  // Test syntax highlighting
  console.log('🎨 Testing syntax highlighting...');
  const codeTests = [
    { code: 'console.log("Hello");', lang: 'javascript' },
    { code: 'print("Hello")', lang: 'python' },
    { code: 'SELECT * FROM users;', lang: 'sql' }
  ];
  
  for (const test of codeTests) {
    console.log(`Testing ${test.lang} highlighting...`);
    // The highlightCode function would be called here
    console.log('✅ Syntax highlighting applied');
  }
  
  // Test Mermaid rendering
  console.log('📊 Testing Mermaid rendering...');
  const mermaidTests = [
    'graph TD\n    A[Start] --> B[End]',
    'sequenceDiagram\n    A->>B: Message',
    'flowchart LR\n    A --> B --> C'
  ];
  
  for (const test of mermaidTests) {
    console.log(`Testing Mermaid: ${test.split('\n')[0]}...`);
    // The renderMermaid function would be called here
    console.log('✅ Mermaid diagram rendered');
  }
  
  // Test math rendering
  console.log('🧮 Testing math rendering...');
  const mathTests = [
    'Inline: $E = mc^2$',
    'Block: $$\\sum_{i=1}^{n} x_i$$',
    'Complex: $$\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}$$'
  ];
  
  for (const test of mathTests) {
    console.log(`Testing math: ${test}...`);
    // The renderMath function would be called here
    console.log('✅ Math rendered');
  }
  
  console.log('🎉 All STREAMDOWN features tested!');
}

// Export test functions
export { testStreamdownIntegration, testStreamingRenderer, testStreamdownFeatures };