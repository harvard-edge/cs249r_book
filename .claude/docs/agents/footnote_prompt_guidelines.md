# Footnote Agent Prompt Guidelines

## Based on User Feedback from Introduction Chapter

### What to Footnote

#### ✅ GOOD Footnotes (User Kept These):

1. **Historical Context with Interesting Details**
   - Dartmouth Conference (1956) - birthplace of AI term
   - Claude Shannon - father of information theory
   - MYCIN - Stanford expert system with performance stats
   - Perceptron (1957) - first neural network with historical impact
   - AlexNet - breakthrough that started deep learning revolution

2. **Technical Concepts with Clear Explanations**
   - Moore's Law - computing power doubling observation
   - Support Vector Machines - with kernel trick explanation
   - Precision/Recall - fundamental ML metrics with tradeoffs
   - Backpropagation - how neural networks learn
   - Top-5 error rate - evaluation metric explanation

3. **Modern ML Terms**
   - Foundation models - large-scale general-purpose models
   - GPT-3 - with parameter count and capabilities
   - ImageNet - massive dataset that drove CV advances

4. **Specialized Hardware/Systems**
   - TPUs - Google's custom ML chips
   - LiDAR - 3D sensing technology for autonomous vehicles
   - TinyML - ML on microcontrollers with memory constraints

#### ❌ AVOID Footnoting (User Removed These):

1. **Basic CS Terms Without Enrichment**
   - Computer Engineering (unless adding historical nugget)
   - IoT (unless adding "50+ billion devices by 2030" type fact)
   - Electrical grids (too general without specific AI context)

2. **General Computing Terms**
   - Latency (CS students know this)
   - Particle accelerators (unless ML-specific detail like "LHC produces 50 petabytes/year")
   - AI definition (too broad/obvious for this audience)

### Key Insights from User Feedback

1. **Add Value, Not Just Definitions**
   - Bad: "IoT: Internet of Things, connected devices"
   - Good: "IoT: Network of 50+ billion connected devices by 2030, generating 79 zettabytes of data annually"

2. **Historical Nuggets Make It Memorable**
   - Include founding dates, key people, interesting backstories
   - Example: "Claude Shannon not only founded information theory but also built the first machine learning device - a mechanical mouse that could learn mazes"

3. **Quantify When Possible**
   - "GPT-3 has 175 billion parameters"
   - "ImageNet contains 14 million images"
   - "AlexNet reduced error rate from 25% to 15.3%"

4. **Target Audience: CS/Engineering Students**
   - They know basic CS terms (latency, APIs, databases)
   - They may not know ML-specific history or terminology
   - Bridge their CS knowledge to ML concepts

### Footnote Selection Criteria

Ask these questions before adding a footnote:

1. **Is this term ML/AI-specific?** → Likely needs footnote
2. **Would a CS student already know this?** → Probably skip
3. **Can I add a fascinating fact/number?** → Include it
4. **Does it have interesting history?** → Include it
5. **Is it just a basic definition?** → Skip or enrich

### Examples of Enriched Footnotes

**Instead of:**
```markdown
[^fn-iot]: **Internet of Things**: Network of connected devices.
```

**Write:**
```markdown
[^fn-iot]: **Internet of Things**: Expected to reach 75 billion connected devices by 2025, generating more data in one year than all of human history combined before 2020.
```

**Instead of:**
```markdown
[^fn-computer-engineering]: **Computer Engineering**: Field combining hardware and software.
```

**Write:**
```markdown
[^fn-computer-engineering]: **Computer Engineering**: Emerged in 1971 when Case Western Reserve University created the first accredited program, responding to the Apollo program's need for integrated hardware-software systems design.
```

### Quantity Guidelines

- **Target: 20-30 footnotes per chapter**
- Focus on quality over quantity
- Every footnote should teach something interesting
- Balance across sections (don't cluster all in one area)

### Style Consistency

1. **Format**: `[^fn-descriptive-name]`
2. **Definition**: `**Term**: Explanation. Interesting fact or analogy.`
3. **Length**: 1-2 sentences (200-400 characters ideal)
4. **Placement**: Where term first appears meaningfully
5. **Tone**: Educational but engaging, not dry

### Final Checklist

Before adding a footnote, ensure it:
- [ ] Adds value beyond basic definition
- [ ] Includes a number, date, or interesting fact when possible
- [ ] Helps CS students understand ML concepts
- [ ] Uses consistent formatting
- [ ] Doesn't duplicate existing footnotes