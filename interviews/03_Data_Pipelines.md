# Data Pipelines & Ingestion 📦

Bottlenecks in moving data from storage to the accelerator.

> **[➕ Add a Flashcard to this section](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/03_Data_Pipelines.md)** (Edit in Browser)

### 📋 Copy-Paste Template
```markdown
<details>
<summary><b>[LEVEL]: [Your Question Title Here]</b></summary>

**Answer:** [Direct answer here]
**Realistic Solution:** [Explain the physics/logic here]
**📖 Deep Dive:** [Optional: Link to Volume I or II Chapter]
</details>
```

---

<details>
<summary><b>🟢 LEVEL 1: Why is 'Data' considered the 'Source Code' of ML systems?</b></summary>

**Answer:** Because the dataset defines the operational logic, not the Python scripts.
**Realistic Solution:** In traditional software, changing the system's behavior requires changing the logic. In ML, the code is just the optimization loop. The dataset is what is "compiled" into the final logic (the weights). Debugging a wrong prediction means debugging the data pipeline, not tracing a code path.
**📖 Deep Dive:** [Volume I: Data Engineering](https://mlsysbook.ai/vol1/data_engineering.html)
</details>

<details>
<summary><b>🟡 LEVEL 2: Why is deduplication the highest-leverage optimization in training?</b></summary>

**Answer:** The Energy-Movement Invariant.
**Realistic Solution:** Moving a bit from DRAM to the ALU costs 100-1,000$\times$ more energy than the math operation itself. Pruning redundant data doesn't just save storage space; it eliminates the most physically expensive operation (memory fetching) in the training loop without reducing the Information Entropy of the dataset.
**📖 Deep Dive:** [Volume I: Data Engineering](https://mlsysbook.ai/vol1/data_engineering.html)
</details>

<details>
<summary><b>🔴 LEVEL 3: You have a 1PB dataset in US-East and a 100Gbps link to US-West. Do you move the data or the compute?</b></summary>

**Answer:** Move the compute to the data (The Physics of Data Gravity).
**Realistic Solution:** A 1PB transfer over a dedicated 100Gbps (12.5 GB/s) link takes over 20 hours. Furthermore, cloud egress fees for 1PB can exceed $90,000. For petabyte-scale ML workloads, it is physically and economically mandatory to provision compute in the same availability zone as the storage.
**📖 Deep Dive:** [Volume I: Data Engineering](https://mlsysbook.ai/vol1/data_engineering.html)
</details>
