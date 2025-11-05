# Betterbib DOI Verification Report

Generated: November 4, 2025

## Summary

Ran `betterbib update -i ./quarto/contents/core/**/*.bib` and found **17 DOIs returning 404 errors**.

Successfully synced: **443 of 460 entries** (96.3%)

## Status of Problematic DOIs

### ✅ VALID (Working Correctly)

These DOIs resolve properly despite betterbib warnings:

1. **10.48550/arXiv.2405.19522** ✅
   - Paper: "Artificial Intelligence Index Report 2024"
   - Authors: Maslej, Perrault, et al.
   - Location: `efficient_ai/efficient_ai.bib`
   - Status: **Verified on arXiv**

2. **10.48550/arXiv.2211.13895** ✅
   - Paper: "Identifying Incorrect Annotations in Multi-Label Classification Data"
   - Authors: Thyagarajan, Snorrason, Northcutt, Mueller
   - Location: `data_engineering/data_engineering.bib`
   - Status: **Verified on arXiv**

### ❌ INVALID - Famous Papers With Wrong DOIs

These are well-known papers that exist but have incorrect DOIs in our bibliography:

3. **10.5555/3433701.3433721** ❌
   - Paper: "ZeRO: Memory Optimization Towards Training Trillion Parameter Models"
   - Authors: Rajbhandari, Rasley, Ruwase, He
   - Location: `hw_acceleration/hw_acceleration.bib`
   - Year: 2020
   - Issue: **This is a famous Microsoft DeepSpeed paper that definitely exists**
   - Action Needed: Find correct DOI (likely from SC20 conference or arXiv)

4. **10.1109/JSSC.2015.2488709** ❌
   - Paper: "Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks"
   - Authors: Chen, Krishna, Emer, Sze
   - Location: `hw_acceleration/hw_acceleration.bib`
   - Year: 2016
   - Issue: **Famous MIT paper, definitely exists**
   - Action Needed: Find correct IEEE JSSC DOI

5. **10.1016/0010-4809(76)90063-5** ❌
   - Paper: "Computer-based consultations in clinical therapeutics: explanation and rule acquisition capabilities of the MYCIN system"
   - Author: Shortliffe, Edward H.
   - Location: `introduction/introduction.bib`
   - Year: 1976
   - Issue: **Classic AI paper, definitely exists**
   - Action Needed: Find correct DOI for this vintage paper

### 🚨 HIGHLY SUSPICIOUS - Likely Fabricated

6. **10.1109/MM.2022.1234567** 🚨
   - Paper: "Chiplet-Based Architectures: The Future of AI Accelerators"
   - Authors: Kannan, Dubey, Horowitz
   - Location: `hw_acceleration/hw_acceleration.bib`
   - Year: 2023, IEEE Micro, Vol 43, No 1
   - Issue: **DOI contains placeholder numbers "1234567"**
   - Action Needed: **Verify this paper actually exists** or remove entry

### ⚠️ INVALID - Need Manual Verification

7. **10.1109/SEC.2019.00035** ⚠️
   - Paper: "Edge Computing for Wildlife Conservation: A Case Study of Intelligent Camera Traps"
   - Authors: Chen, Zhang, Kumar, Patel
   - Location: `ai_for_good/ai_for_good.bib`
   - Action Needed: Search by title/authors to verify existence

8. **10.1007/s00146-021-01331-1** ⚠️
   - Paper: AI and Society journal article
   - Authors: Taylor, Surendranath, Bentley, Mohr
   - Location: `ai_for_good/ai_for_good.bib`
   - Year: 2022, Vol 37, No 4, pp 1421-1436
   - Action Needed: May be off by one digit (similar DOI -01331-9 exists)

9. **10.1109/MC.1981.1653991** ⚠️
   - Paper: "The 8087 Numeric Data Processor"
   - Author: Fisher, Lawrence D.
   - Location: `hw_acceleration/hw_acceleration.bib`
   - Year: 1981 (vintage paper)
   - Action Needed: Verify correct DOI for 1981 IEEE Computer article

10. **10.1109/MM.2019.2923951** ⚠️
    - Paper: "Tensor Cores: Understanding, Programming, and Performance Analysis"
    - Authors: Wu, Grot, Hardavellas
    - Location: `hw_acceleration/hw_acceleration.bib`
    - Year: 2019, IEEE Micro

11. **10.1109/TC.2018.2799212** ⚠️
    - Paper: "Accelerating Genomic Data Analysis with Domain-Specific Architectures"
    - Authors: Shang, Wang, Liu
    - Location: `hw_acceleration/hw_acceleration.bib`
    - Year: 2018

12. **10.2312/EGGH/EGGH07/055-064** ⚠️
    - Paper: "A hardware redundancy and recovery mechanism for reliable scientific computation on graphics processors"
    - Authors: Sheaffer, Luebke, Skadron
    - Location: `robust_ai/robust_ai.bib`
    - Year: 2007

### ⚠️ ADDITIONAL INVALID DOIs

13. **10.1109/MM.2022.3186575** ⚠️
14. **10.1145/3580309** ⚠️
15. **10.1109/MM.2020.2975796** ⚠️
16. **10.1109/TNNLS.2021.3088493** ⚠️
17. **10.48550/arXiv.2103.14749** ⚠️ (arXiv DOI format issue)

## Recommended Actions

### Immediate Actions

1. **Remove/Fix the suspicious entry**: 
   - `10.1109/MM.2022.1234567` (Chiplet paper with placeholder DOI)
   - Search for this paper by title to see if it exists

2. **Fix well-known papers**:
   - ZeRO paper: Search for correct SC20 or arXiv reference
   - Eyeriss paper: Find correct IEEE JSSC DOI
   - MYCIN paper: Find correct 1976 Elsevier DOI

### Manual Verification Process

For each invalid DOI:
1. Search Google Scholar using: `"exact title" author1 author2 year`
2. Check publisher website (IEEE Xplore, SpringerLink, ACM DL, etc.)
3. Verify paper actually exists before keeping the entry
4. Update with correct DOI or URL if found
5. **Remove entry entirely if paper doesn't exist**

### Commands to Help

```bash
# Check individual DOI
curl -s -o /dev/null -w "%{http_code}" "https://doi.org/YOUR_DOI"

# Search Google Scholar
# Use browser with: "exact paper title" author names

# Check arXiv
curl -s "https://arxiv.org/abs/XXXXX.XXXXX" | grep -o "<title>.*</title>"
```

## Files Affected

- `quarto/contents/core/ai_for_good/ai_for_good.bib`
- `quarto/contents/core/data_engineering/data_engineering.bib`
- `quarto/contents/core/efficient_ai/efficient_ai.bib`
- `quarto/contents/core/hw_acceleration/hw_acceleration.bib`
- `quarto/contents/core/introduction/introduction.bib`
- `quarto/contents/core/robust_ai/robust_ai.bib`

## Notes

- Some DOIs might be off by one digit (common typo)
- Vintage papers (pre-1990) may not have valid DOIs
- Some conference proceedings use non-standard DOI formats
- arXiv DOIs through DataCite sometimes have issues with Crossref lookups

