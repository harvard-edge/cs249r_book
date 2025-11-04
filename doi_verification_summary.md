# DOI Verification Summary - November 4, 2025

## Actions Completed

### ✅ REMOVED (Fabricated/Invalid Papers)

1. **Kannan2023chiplet** - `10.1109/MM.2022.1234567`
   - Paper: "Chiplet-Based Architectures: The Future of AI Accelerators"
   - **STATUS**: REMOVED (DOI contained placeholder "1234567")
   - Files modified:
     - Removed citation from: `hw_acceleration/hw_acceleration.qmd` (line 2816)
     - Removed entry from: `hw_acceleration/hw_acceleration.bib`

2. **chen2019edge** - `10.1109/SEC.2019.00035`
   - Paper: "Edge Computing for Wildlife Conservation: A Case Study of Intelligent Camera Traps"
   - **STATUS**: REMOVED (Could not verify paper exists)
   - Files modified:
     - Removed citation from: `ai_for_good/ai_for_good.qmd` (line 143)
     - Removed entry from: `ai_for_good/ai_for_good.bib`

### ✅ FIXED (Typo in DOI)

3. **taylor2022** - AI and Society paper
   - **OLD DOI**: `10.1007/s00146-021-01331-1` (404 error)
   - **NEW DOI**: `10.1007/s00146-021-01331-9` (✅ works)
   - **STATUS**: FIXED (typo in last digit)
   - Files modified:
     - Updated in: `ai_for_good/ai_for_good.bib`

## Papers Requiring Further Investigation

### Famous Papers with Invalid DOIs (Likely Need Different References)

4. **Rajbhandari2020** - ZeRO Paper
   - DOI: `10.5555/3433701.3433721` (404 error)
   - Paper: "ZeRO: Memory Optimization Towards Training Trillion Parameter Models"
   - **STATUS**: KEPT (Famous Microsoft DeepSpeed paper, definitely exists)
   - **NOTE**: The 10.5555 DOI prefix is used by ACM for some conference proceedings. This paper was from SC20.
   - **RECOMMENDATION**: Look for arXiv version or official SC20 proceedings DOI
   - Citation location: `hw_acceleration/hw_acceleration.qmd`

5. **chen2016eyeriss** - Eyeriss Paper
   - DOI: `10.1109/JSSC.2015.2488709` (404 error)
   - Paper: "Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks"
   - **STATUS**: NEEDS VERIFICATION (Famous MIT paper, definitely exists)
   - **RECOMMENDATION**: Search IEEE JSSC Vol 51, No 1, 2016 for correct DOI
   - Bib location: `hw_acceleration/hw_acceleration.bib`

6. **shortliffe1976mycin** - MYCIN Paper  
   - DOI: `10.1016/0010-4809(76)90063-5` (404 error)
   - Paper: "Computer-based consultations in clinical therapeutics"
   - Author: Shortliffe, Edward H.
   - **STATUS**: NEEDS VERIFICATION (Classic 1976 AI paper)
   - **RECOMMENDATION**: May not have had DOI originally, could use URL instead
   - Bib location: `introduction/introduction.bib`

### Other Papers Returning 404 Errors

7. **fisher_8087_1981** - `10.1109/MC.1981.1653991`
   - "The 8087 Numeric Data Processor" (1981)
   - Location: `hw_acceleration/hw_acceleration.bib`

8. **wu_tensor_2019** - `10.1109/MM.2019.2923951`
   - "Tensor Cores: Understanding, Programming, and Performance Analysis"
   - Location: `hw_acceleration/hw_acceleration.bib`

9. **Shang2018GenomicsAccel** - `10.1109/TC.2018.2799212`
   - "Accelerating Genomic Data Analysis with Domain-Specific Architectures"
   - Location: `hw_acceleration/hw_acceleration.bib`

10. **sheaffer2007** - `10.2312/EGGH/EGGH07/055-064`
    - "A hardware redundancy and recovery mechanism..."
    - Location: `robust_ai/robust_ai.bib`

11. **10.1109/MM.2022.3186575** - IEEE MultiMedia 2022
12. **10.1145/3580309** - ACM publication
13. **10.1109/MM.2020.2975796** - IEEE MultiMedia 2020
14. **10.1109/TNNLS.2021.3088493** - IEEE TNNLS 2021
15. **10.48550/arXiv.2103.14749** - arXiv format issue

## Next Steps

### Immediate Actions Required

1. **Find correct DOIs for famous papers** (Priority: HIGH):
   - Eyeriss paper (definitely exists, just need correct DOI)
   - ZeRO paper (consider using arXiv reference)
   - MYCIN paper (may need URL instead of DOI)

2. **Verify remaining papers** (Priority: MEDIUM):
   - Search for each paper by title/authors
   - If paper doesn't exist, remove citation AND bib entry
   - If paper exists, find correct DOI or URL

3. **Update report** with final status of all papers

### Commands for Verification

```bash
# Test a DOI
curl -s -o /dev/null -w "%{http_code}" "https://doi.org/YOUR_DOI"

# Find citations of a bib entry in qmd files
grep -r "@CITATION_KEY" quarto/contents/core/**/*.qmd

# Search arXiv
curl -s "https://arxiv.org/abs/XXXXX.XXXXX" | grep -o "<title>.*</title>"
```

### Files Modified So Far

- `quarto/contents/core/hw_acceleration/hw_acceleration.qmd` - Removed Kannan2023chiplet citation
- `quarto/contents/core/hw_acceleration/hw_acceleration.bib` - Removed Kannan2023chiplet entry
- `quarto/contents/core/ai_for_good/ai_for_good.qmd` - Removed chen2019edge citation
- `quarto/contents/core/ai_for_good/ai_for_good.bib` - Removed chen2019edge entry, Fixed taylor2022 DOI

## Summary Statistics

- **Total problematic DOIs**: 17
- **DOIs verified invalid and removed**: 2
- **DOIs fixed (typos)**: 1
- **DOIs to keep (famous papers)**: 3
- **DOIs still to verify**: 11

## Recommendations

1. For vintage papers (pre-1990), consider replacing DOIs with URLs if DOIs don't work
2. For papers that definitely exist but have bad DOIs, search publisher websites directly
3. Create a process to validate DOIs when adding new citations
4. Consider using Zotero or similar tools that automatically validate DOIs

