# Datasheet for TinyTalks Dataset

*Following "Datasheets for Datasets" by Gebru et al. (2018)*

---

## Motivation

### 1. For what purpose was the dataset created?

TinyTalks was created to provide an educational, lightweight conversational Q&A dataset specifically designed for teaching transformer architectures. The primary goal is to enable students to train their first transformer model and see meaningful learning in under 5 minutes, creating an "aha!" moment that demonstrates how transformers learn patterns.

### 2. Who created the dataset and on behalf of which entity?

TinyTalks was created by the TinyTorch Contributors as part of the TinyTorch educational deep learning framework. It was developed specifically for the Transformer milestone (Module 13 / Milestone 05) of the TinyTorch curriculum.

### 3. Who funded the creation of the dataset?

This dataset was created as an open-source educational resource without specific funding. It is part of the broader TinyTorch project.

---

## Composition

### 4. What do the instances that comprise the dataset represent?

Each instance represents a question-answer pair in natural language. Questions are conversational or factual queries, and answers are appropriate responses that an AI assistant might provide.

### 5. How many instances are there in total?

**350 question-answer pairs** distributed across 5 difficulty levels:
- Level 1 (Greetings & Identity): 50 pairs
- Level 2 (Simple Facts): 100 pairs
- Level 3 (Basic Math): 50 pairs
- Level 4 (Common Sense Reasoning): 100 pairs
- Level 5 (Multi-turn Context): 50 pairs

### 6. Does the dataset contain all possible instances or is it a sample?

This is a curated sample. It represents a pedagogically-designed subset of possible conversational Q&A pairs, specifically selected for educational value and training efficiency.

### 7. What data does each instance consist of?

Each instance consists of:
- **Question (Q:)**: A natural language question (5-20 words typically)
- **Answer (A:)**: A natural language response (5-25 words typically)
- **Format**: Plain text with clear delimiters

Example:
```
Q: What color is the sky?
A: The sky is blue during the day.
```

### 8. Is there a label or target associated with each instance?

Yes. In a Q&A format, the question serves as input and the answer serves as the target label for supervised learning. For autoregressive language modeling, the entire text sequence serves as both input and target (shifted by one token).

### 9. Is any information missing from individual instances?

No. Each Q&A pair is complete. However, the dataset intentionally excludes:
- Timestamps
- User demographics
- Conversation metadata
- Multi-modal information (images, audio)

This is by design to keep the dataset simple and focused.

### 10. Are relationships between individual instances made explicit?

Partially. Level 5 (Multi-turn Context) contains sequential Q&A pairs where the answer to one question sets up context for the next. However, most Q&A pairs (Levels 1-4) are independent.

### 11. Are there recommended data splits?

Yes, we provide:
- **Training set**: 245 pairs (70%)
- **Validation set**: 53 pairs (15%)
- **Test set**: 52 pairs (15%)

The splits maintain proportional representation of all 5 difficulty levels and are deterministic (same split every time).

### 12. Are there any errors, sources of noise, or redundancies in the dataset?

- **Errors**: Minimal. All pairs were manually reviewed for grammatical and factual accuracy.
- **Noise**: None intentionally introduced.
- **Redundancies**: Some intentional near-duplicates exist to reinforce patterns (e.g., multiple arithmetic questions with different numbers).

### 13. Is the dataset self-contained, or does it link to or otherwise rely on external resources?

Fully self-contained. No external resources, URLs, or references required.

### 14. Does the dataset contain data that might be considered confidential?

No. All data is original or public-domain factual knowledge. No confidential, proprietary, or sensitive information is included.

### 15. Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?

No. The dataset was explicitly designed to be:
- Appropriate for all ages (G-rated)
- Culturally neutral
- Free of offensive, biased, or sensitive content
- Reviewed for potential harm

---

## Collection Process

### 16. How was the data associated with each instance acquired?

All Q&A pairs were **manually authored** by TinyTorch contributors. No scraping, crowdsourcing, or automated generation was used for v1.0.

### 17. What mechanisms or procedures were used to collect the data?

1. **Systematic generation**: Each difficulty level was designed with specific learning objectives
2. **Manual authoring**: Contributors wrote Q&A pairs following style guidelines
3. **Review process**: Each pair reviewed by at least one other contributor
4. **Quality control**: Automated validation script checks format, grammar, and distribution

### 18. If the dataset is a sample from a larger set, what was the sampling strategy?

Not applicable. This is an original curated dataset, not a sample from a larger corpus.

### 19. Who was involved in the data collection process and how were they compensated?

TinyTorch contributors (open-source volunteers). No monetary compensation. Contributors are acknowledged in project documentation.

### 20. Over what timeframe was the data collected?

December 2024 - January 2025 (v1.0 release)

### 21. Were any ethical review processes conducted?

Informal ethical review by TinyTorch maintainers, focusing on:
- Appropriateness for educational use
- Absence of bias and offensive content
- Privacy considerations (no PII)
- Cultural sensitivity

No formal IRB review was required as no human subjects or sensitive data were involved.

---

## Preprocessing / Cleaning / Labeling

### 22. Was any preprocessing/cleaning/labeling of the data done?

Minimal preprocessing:
- **Formatting**: Standardized to `Q: ... \n A: ... \n\n` format
- **Encoding**: UTF-8 text encoding
- **Line endings**: Unix-style (LF)
- **Grammar**: Manual review and correction
- **Whitespace**: Consistent spacing

No automated cleaning or labeling was required as data was manually authored.

### 23. Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data?

Since the data was manually authored, the "raw" data is the authored text itself. The generation script (`scripts/generate_tinytalks.py`) serves as the source of truth and can regenerate the dataset identically.

### 24. Is the software used to preprocess/clean/label the instances available?

Yes:
- **Generation**: `scripts/generate_tinytalks.py` (Python)
- **Validation**: `scripts/validate_dataset.py` (Python)
- **Statistics**: `scripts/stats.py` (Python)

All scripts are open-source (MIT license) and included in the repository.

---

## Uses

### 25. Has the dataset been used for any tasks already?

Yes, the primary use case:
- **Task**: Autoregressive language modeling (transformer training)
- **Model**: TinyGPT (small GPT-style transformer)
- **Milestone**: TinyTorch Module 13 - Transformers
- **Performance**: Achieves ~80% accuracy on Level 1-2 questions after 3-5 minutes of training

### 26. Is there a repository that links to any or all papers or systems that use the dataset?

The dataset is hosted at: https://github.com/harvard-edge/cs249r_book/tree/main/tinytorch/tree/main/datasets/tinytalks

Usage examples:
- `milestones/05_2017_transformer/tinybot_demo.py` - Main training script
- `examples/demo_usage.py` - Data loading examples

### 27. What (other) tasks could the dataset be used for?

Potential uses:
- **Tokenization experiments** (character vs. BPE vs. word-level)
- **Attention visualization** (inspecting attention patterns on Q&A)
- **Embedding analysis** (visualizing learned representations)
- **Few-shot learning** (testing prompt-based learning)
- **Model debugging** (small enough to trace gradients manually)
- **Architecture experimentation** (testing transformer variants)

### 28. Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?

Considerations:
- **Size limitation**: 350 pairs may be insufficient for production models (by design)
- **Simplicity**: Limited complexity may not reflect real-world conversational AI challenges
- **English-only**: v1.0 is monolingual
- **Character-level**: Designed for character tokenization; may need adjustment for other tokenizers
- **No ambiguity**: Answers are deliberately unambiguous, unlike real conversations

These are intentional design choices for educational clarity, not limitations.

### 29. Are there tasks for which the dataset should not be used?

**Not suitable for:**
- ❌ Production conversational AI systems
- ❌ Benchmarking state-of-the-art models
- ❌ Research on complex reasoning or long-context understanding
- ❌ Multilingual or cross-cultural studies (v1.0 is English-only)
- ❌ Real-world chatbot deployment

**Designed for:**
- ✅ Educational transformer training
- ✅ Rapid prototyping
- ✅ Architecture testing
- ✅ Understanding transformer mechanics

---

## Distribution

### 30. Will the dataset be distributed to third parties outside of the entity on behalf of which it was created?

Yes. TinyTalks is **open-source** and freely available to everyone under CC BY 4.0 license.

### 31. How will the dataset be distributed?

- **GitHub repository**: https://github.com/harvard-edge/cs249r_book/tree/main/tinytorch/tree/main/datasets/tinytalks
- **Included with TinyTorch**: Ships with the framework (no download required)
- **Format**: Plain text files (.txt)

### 32. When will the dataset be distributed?

- **v1.0.0**: January 2025 (initial release)
- **Future versions**: As needed based on community feedback

### 33. Will the dataset be distributed under a copyright or other intellectual property (IP) license?

Yes. **Creative Commons Attribution 4.0 International (CC BY 4.0)**

- ✅ Free to share and adapt
- ✅ Commercial use allowed
- ✅ Must provide attribution
- ✅ No additional restrictions

### 34. Have any third parties imposed IP-based or other restrictions on the data?

No. All content is original or public-domain factual knowledge.

### 35. Do any export controls or other regulatory restrictions apply to the dataset?

No export controls or regulatory restrictions apply.

---

## Maintenance

### 36. Who will be supporting/hosting/maintaining the dataset?

**TinyTorch Contributors** (maintainers of the TinyTorch project)

Primary maintainer: VJ (@profvjreddi on GitHub)

### 37. How can the owner/curator/manager of the dataset be contacted?

- **GitHub Issues**: https://github.com/harvard-edge/cs249r_book/tree/main/tinytorch/issues
- **GitHub Discussions**: https://github.com/harvard-edge/cs249r_book/tree/main/tinytorch/discussions
- **Email**: tinytorch@example.com

### 38. Is there an erratum?

Not yet. Any discovered errors will be documented in:
- **GitHub Issues** (tagged `dataset` + `tinytalks`)
- **CHANGELOG.md** (in dataset directory)

### 39. Will the dataset be updated?

Yes, planned updates:
- **v1.1** - Bug fixes and minor additions (50-100 new pairs)
- **v2.0** - Multi-language support
- **v3.0** - Expanded to 1,000 pairs with more complex reasoning

Updates will follow semantic versioning:
- **Major** (X.0.0) - Breaking changes to format
- **Minor** (0.X.0) - Backward-compatible additions
- **Patch** (0.0.X) - Bug fixes only

### 40. If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances?

Not applicable. The dataset does not contain any personal data, PII, or information about real individuals.

### 41. Will older versions of the dataset continue to be supported/hosted/maintained?

Yes. All versions will remain available via Git tags:
- `git checkout tags/tinytalks-v1.0.0`

### 42. If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?

Yes:
- **Pull Requests**: Submit new Q&A pairs or improvements
- **Issues**: Report errors or suggest enhancements
- **Forks**: Create derivative datasets (with attribution)

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

---

## References

Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2018). Datasheets for datasets. *arXiv preprint arXiv:1803.09010*.

---

*Last updated: January 2025*

