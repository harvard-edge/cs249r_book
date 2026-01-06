# Expert Review: Storage Systems for ML (Volume 2)

**Reviewer**: Ion Stoica (UC Berkeley Professor, Co-founder Anyscale/Databricks)
**Chapter**: Storage Systems for ML
**Date**: 2026-01-04
**Review Type**: Technical and Pedagogical Guidance

---

## Executive Summary

The Storage chapter occupies a critical position in Volume 2, bridging the Infrastructure chapter's datacenter foundations with the Distributed Training chapter's compute coordination. Having built Ray and contributed to Spark, I understand intimately how storage architecture determines whether distributed ML systems achieve their theoretical potential or collapse under I/O bottlenecks. This review provides guidance on what this chapter must cover to prepare students for production-scale ML systems.

The current stub correctly identifies key topics (distributed storage, data lakes, feature stores) but lacks the distributed systems depth students need. My primary concern: students must understand that **storage in distributed ML is not just "bigger hard drives"** but involves fundamental trade-offs in consistency, availability, replication, and data locality that reshape how training and inference systems are designed.

---

## 1. Distributed Storage Fundamentals

### What Must Be Covered

**Consistency Models (Essential, 2-3 sections)**

Students need rigorous treatment of consistency models because storage choices cascade into training correctness:

- **Strong consistency vs. eventual consistency**: When training workers read dataset shards, what guarantees exist? If worker A writes a checkpoint and worker B immediately reads it, will B see A's write?

- **CAP theorem implications for ML**: This is mentioned briefly in Vol 1 Data Engineering but needs Volume 2 depth. Training storage can sacrifice availability for consistency (checkpoints must be correct), while serving storage might sacrifice consistency for availability (stale features acceptable for 100ms).

- **Linearizability requirements**: Checkpoint storage MUST be linearizable. A training run that resumes from a corrupted or partial checkpoint wastes thousands of GPU-hours. Students need to understand why.

**Recommended depth**: Full mathematical treatment of consistency models with concrete ML examples. Show the failure modes:

```
Failure scenario: Non-linearizable checkpoint storage
- Worker 0 writes checkpoint at step 10000
- Coordinator signals "checkpoint complete"
- Workers 1-7 write their shards
- Network partition: Worker 3's write lost
- Training resumes from incomplete checkpoint
- Model diverges, 48 hours of training wasted
```

**Replication Strategies (Essential, 1-2 sections)**

- **Synchronous vs. asynchronous replication**: Training data can use async (eventual consistency acceptable), checkpoints require sync (data loss unacceptable).

- **Replication factor selection**: Why is 3x standard? What's the math behind durability guarantees?

$$P(\text{data loss}) = P(\text{disk failure})^{\text{replication factor}}$$

For 10PB of training data across 1000 disks with 4% annual failure rate, 3x replication yields expected data loss of 6.4e-5 files/year. Show this calculation.

**Sharding and Partitioning (Essential, 1-2 sections)**

- **Hash partitioning vs. range partitioning**: Training data typically uses hash (uniform distribution), feature stores use range (time-series queries).

- **Partition key selection for ML workloads**: Bad partition keys create hot spots. If you partition user embeddings by user_id hash but 1% of users generate 50% of queries, you've created skewed load.

- **Resharding operations**: When you add storage nodes, how does data redistribute without stopping training?

### Pedagogical Recommendation

Start with a concrete failure: "A 1000-GPU training run crashed at hour 72 because checkpoint storage returned stale data. Total cost: $2.1M in wasted compute. Let's understand why this happened and how storage system design prevents it."

---

## 2. Data Pipeline Architecture

### Connection to Ray and Spark

**This is where I have strong opinions based on building these systems.**

**Object Store Model (Ray) - Essential Coverage**

Ray's object store represents a fundamentally different approach than traditional distributed file systems:

- **Plasma object store**: Objects are immutable, enabling zero-copy sharing between workers on the same node. This is critical for ML where the same dataset shard is read by multiple GPU workers.

- **Location-aware scheduling**: Ray's scheduler knows where objects reside and preferentially schedules tasks to co-located workers. Students must understand this data locality principle.

- **Spilling to disk**: When object store exceeds memory, objects spill to local SSD. The hierarchy is: GPU memory -> CPU memory -> local SSD -> distributed storage. Each level has 10-100x latency difference.

**Shuffle Storage (Spark) - Essential Coverage**

Distributed training preprocessing often uses Spark. Students need to understand:

- **Shuffle architecture**: When data must be repartitioned (e.g., grouping all examples for a user), every mapper sends data to every reducer. For N workers, this is O(N^2) connections.

- **External shuffle service**: Why does Spark use a separate shuffle service rather than worker-to-worker transfers? Answer: fault tolerance. If a worker dies mid-shuffle, its shuffle files are lost unless externalized.

- **Shuffle file format**: Sorted by partition key, enabling efficient merge during reduce. This is the same principle as LSM-tree compaction.

**Recommended Treatment**

Create a unified view showing how Ray (execution), Spark (preprocessing), and storage (persistence) form an integrated data plane:

```
Raw Data (S3/GCS/HDFS)
    |
    v
Spark Preprocessing (shuffle storage)
    |
    v
Ray Data Loading (object store)
    |
    v
GPU Training (HBM)
    |
    v
Checkpoint Storage (distributed FS)
```

Each arrow represents a data movement with distinct consistency/latency requirements.

---

## 3. Feature Stores

### My Assessment: Critical for Recommendation, Important for Others

**This requires nuanced treatment because importance varies dramatically by model type.**

**For Recommendation Systems (DLRM, Wide&Deep) - Feature Stores are Essential**

The editorial guidelines correctly identify this. Recommendation models have:

- **Trillion-parameter embedding tables**: User and item embeddings exceed 1TB. These are essentially a key-value store queried at inference time.

- **Real-time feature freshness requirements**: "User clicked item X 10 seconds ago" must be reflected in the next recommendation. This requires streaming feature updates.

- **Point-in-time correctness**: Training must use features as they existed when the label was generated, not current features. This is called "time-travel" and is a core feature store capability.

**For LLMs - Feature Stores are Minimal**

LLM training doesn't use feature stores in the traditional sense. The "features" are tokens, and the training data is static text corpora. Inference may use retrieval augmentation (RAG), which has feature-store-like characteristics, but the architecture differs.

**Recommended Section Structure**

1. **Feature Store Fundamentals** (1 section)
   - Online vs. offline stores
   - Feature versioning and lineage
   - Point-in-time joins

2. **Feature Stores for Recommendation** (1-2 sections, detailed)
   - Embedding table serving
   - Real-time feature computation
   - Case study: Meta's feature infrastructure

3. **When Feature Stores Matter Less** (1 section)
   - LLM training data pipelines
   - Computer vision training
   - Appropriate scope

**Quantitative Treatment Required**

Show the latency requirements:

| System | Feature Lookup Latency Budget | Why |
|--------|------------------------------|-----|
| Recommendation | < 10ms | User-facing, 100ms total budget |
| Fraud Detection | < 50ms | Transaction approval SLA |
| Ad Ranking | < 5ms | Auction deadlines |

Feature store architecture is driven by these latency constraints.

---

## 4. Checkpoint Storage for Distributed Training

### What Belongs Here vs. Distributed Training Chapter

**In This Chapter (Storage Focus)**:

- Checkpoint file formats and their storage characteristics
- Consistency requirements for checkpoint storage
- Tiered checkpoint storage (local SSD -> shared FS -> cold storage)
- Checkpoint storage bandwidth requirements
- Asynchronous checkpointing and storage I/O overlap

**In Distributed Training Chapter (Algorithm Focus)**:

- When to checkpoint (iteration frequency)
- Distributed checkpoint coordination (barrier synchronization)
- Checkpoint resume protocols
- Gradient checkpointing (memory optimization, not storage)

### Essential Content for This Chapter

**Checkpoint Size Analysis by Model Type**

The editorial guidelines correctly identify this variance:

| Model Type | Checkpoint Size | Frequency | Storage Bandwidth Needed |
|------------|----------------|-----------|-------------------------|
| LLM (175B) | 350 GB (FP16 weights) + 1.4 TB (optimizer) | Every 30 min | 100+ GB/s aggregate |
| Vision (ResNet-50) | 100 MB | Every epoch | 10 MB/s |
| Recommendation (DLRM) | 10 TB (embeddings) | Daily | Must be incremental |

**Hierarchical Checkpoint Storage**

Production systems use tiered storage:

1. **Local NVMe**: First checkpoint copy for fast restart
2. **Parallel File System (Lustre/GPFS)**: Accessible from all nodes
3. **Object Store (S3/GCS)**: Durable, cross-region backup

Students must understand why each tier exists and the cost/performance trade-offs.

**Asynchronous Checkpointing**

Critical insight: Checkpointing should not block training. The storage system must support:

- Background writes while training continues
- Atomic commit (checkpoint is either complete or doesn't exist)
- Read-after-write consistency for restart

Show the timeline:

```
Training: [compute][compute][compute][compute]...
Storage:       [async write checkpoint N    ]
                        [async write checkpoint N+1    ]
```

This requires careful synchronization between compute and storage that students must understand.

---

## 5. Object Store vs. File System

### The Pedagogically Correct Presentation

**Start with Access Patterns, Not Technology**

Students should first understand what operations their workloads perform:

| Operation | Training Data | Checkpoints | Feature Store |
|-----------|--------------|-------------|---------------|
| Read pattern | Sequential scan | Random read | Point lookup |
| Write pattern | Write-once | Overwrite | Update |
| Consistency | Eventual OK | Strong required | Session consistency |
| Latency tolerance | Seconds OK | Sub-second | Milliseconds |

**Then Map to Storage Systems**

| Storage Type | Best For | Why |
|--------------|----------|-----|
| Object Store (S3) | Training data, model artifacts | Cheap, durable, high throughput for large sequential reads |
| Distributed FS (HDFS/Lustre) | Checkpoint storage | POSIX semantics, random access, overwrite support |
| Key-Value Store (Redis/Cassandra) | Feature serving | Low latency point lookups |
| Columnar (Parquet on Delta Lake) | Feature engineering | Column pruning, predicate pushdown |

**Critical Misconception to Address**

Students often think "object stores are slow." This is wrong. S3 can deliver 100+ Gbps to a single client with sufficient parallelism. The issue is:

- Object stores have high first-byte latency (~50ms)
- Object stores don't support partial updates
- Object stores have eventual consistency (though S3 is now strongly consistent for PUTs)

For training data (large sequential reads, write-once), object stores are often BETTER than distributed file systems.

**Include Cost Analysis**

| Storage Type | $/GB/month | Access Patterns |
|--------------|-----------|-----------------|
| S3 Standard | $0.023 | Training data, models |
| S3 Glacier | $0.004 | Long-term checkpoints |
| EBS gp3 | $0.08 | Active checkpoints |
| FSx Lustre | $0.14 | High-performance training I/O |

A 10 PB training dataset costs $230K/month on S3 vs. $1.4M/month on Lustre. This economic reality drives architecture decisions.

---

## 6. Caching and Tiering

### Memory Hierarchy for ML

**Essential Treatment**

The memory hierarchy has ML-specific characteristics that differ from general computing:

```
GPU HBM (80 GB, 3.4 TB/s)
    |
CPU DRAM (2 TB, 200 GB/s)
    |
Local NVMe (30 TB, 15 GB/s)
    |
Network Storage (PB, 100+ GB/s aggregate)
    |
Cold Storage (Exabyte, 10 GB/s)
```

**ML-Specific Caching Patterns**

1. **Training data caching**: If dataset fits in aggregate CPU memory across workers, cache it. Eliminates I/O bottleneck entirely.

2. **Embedding table caching**: Hot embeddings in GPU memory, warm in CPU, cold in storage. Follow power-law distribution.

3. **KV cache for LLM inference**: This is a different kind of caching but students should understand its storage implications.

**Prefetching Strategies**

Ray Data and tf.data implement sophisticated prefetching:

```python
# Ray Data prefetching
ds = ray.data.read_parquet("s3://bucket/data")
ds = ds.map_batches(preprocess)
ds = ds.iter_batches(prefetch_batches=10)  # Critical for GPU utilization
```

Show how prefetch depth affects GPU utilization:

| Prefetch Depth | GPU Utilization | Memory Overhead |
|---------------|-----------------|-----------------|
| 1 | 40% (I/O bound) | Minimal |
| 4 | 75% | 4x batch size |
| 10 | 95% | 10x batch size |
| 20 | 98% (diminishing returns) | 20x batch size |

**Tiered Storage Policies**

Automated tiering based on access patterns:

- Data accessed in last 7 days: Hot tier (SSD)
- Data accessed in last 30 days: Warm tier (HDD)
- Older data: Cold tier (object storage)

For ML, this maps to:
- Current training run data: Hot
- Recent experiment data: Warm
- Historical experiments: Cold

---

## 7. Recommended Chapter Structure

Based on my analysis, here is the proposed structure:

### Part 1: Distributed Storage Foundations (4-5 sections)

1. **Storage Requirements for ML at Scale**
   - Access pattern analysis by workload type
   - Throughput vs. latency trade-offs
   - Consistency requirements

2. **Consistency and Replication**
   - CAP theorem for ML storage
   - Replication strategies
   - Consistency models (linearizability, sequential, eventual)

3. **Partitioning and Data Distribution**
   - Sharding strategies
   - Data locality principles
   - Rebalancing operations

4. **Storage System Taxonomy**
   - Object stores
   - Distributed file systems
   - Key-value stores
   - When to use each

### Part 2: Data Pipeline Storage (3-4 sections)

5. **Training Data Storage**
   - Format selection (Parquet, TFRecord, etc.)
   - Compression trade-offs
   - Data lake architecture

6. **Object Store Architecture** (Ray focus)
   - In-memory object sharing
   - Spilling policies
   - Location-aware scheduling

7. **Shuffle and Intermediate Storage** (Spark focus)
   - Shuffle architecture
   - External shuffle services
   - Sort-merge joins

### Part 3: Feature and Model Storage (3-4 sections)

8. **Feature Store Architecture**
   - Online vs. offline stores
   - Point-in-time correctness
   - When feature stores matter (recommendation) vs. don't (LLM training)

9. **Checkpoint Storage Systems**
   - Checkpoint file formats
   - Tiered checkpoint storage
   - Asynchronous checkpointing
   - Bandwidth requirements

10. **Model Registry and Artifact Storage**
    - Version control for models
    - Metadata management
    - Multi-model serving storage

### Part 4: Performance and Economics (2-3 sections)

11. **Caching and Memory Hierarchy**
    - ML-specific caching patterns
    - Prefetching strategies
    - Tiered storage policies

12. **Storage Economics**
    - Cost modeling by storage type
    - Capacity planning
    - Trade-off analysis

13. **Case Studies**
    - Google training data infrastructure
    - Meta feature store
    - Tesla vision data pipeline

---

## 8. Critical Gaps to Address

### Must Include

1. **Quantitative bandwidth analysis**: Show the math for when storage becomes the bottleneck. For 1000 GPUs at 50% compute efficiency with 10ms/batch, what storage bandwidth is needed?

2. **Failure mode analysis**: What happens when storage fails? How does training recover? What's the blast radius of a storage node failure?

3. **Data locality principles**: Why does Ray schedule tasks near data? What's the cost of remote data access?

4. **Format selection guidance**: When Parquet vs. TFRecord vs. raw files? Include quantitative comparison.

### Should Include

1. **Delta Lake / Iceberg for ML**: ACID transactions on data lakes are becoming important for reproducible ML.

2. **Streaming data ingestion**: Kafka integration for real-time feature updates.

3. **Multi-cloud storage considerations**: Data gravity, egress costs, cross-region replication.

### Nice to Have

1. **Storage hardware trends**: NVMe, CXL, computational storage.

2. **Disaggregated storage architectures**: How hyperscalers are separating compute from storage.

---

## 9. Connection to Other Chapters

### Prerequisites from Infrastructure (done)

Students should already understand:
- Datacenter network topology (relevant for storage access patterns)
- NVLink/NVSwitch (for GPU-to-GPU data sharing vs. storage)
- TCO analysis (storage is significant cost component)

### What This Chapter Enables

**For Distributed Training (next)**:
- Checkpoint storage architecture
- Training data loading at scale
- Gradient accumulation storage (for very large batches)

**For Communication**:
- Data staging for collective operations
- Overlap of communication and storage I/O

**For Fault Tolerance**:
- Checkpoint-based recovery
- Data replication for resilience

---

## 10. Final Recommendations

### Highest Priority

1. **Add rigorous consistency model treatment** - This is non-negotiable for distributed systems literacy.

2. **Include quantitative bandwidth analysis** - Students must be able to calculate whether storage is their bottleneck.

3. **Differentiate by model type** - Feature stores critical for RecSys, less so for LLMs. Make this explicit.

### Medium Priority

4. **Add Ray object store section** - Modern ML systems use this; students need exposure.

5. **Include cost analysis** - Storage economics drive real architecture decisions.

6. **Cover checkpoint storage in depth** - This is where training fails catastrophically.

### Lower Priority

7. **Add emerging technologies** - Delta Lake, computational storage, CXL.

8. **Multi-cloud considerations** - Important but secondary to fundamentals.

---

## Summary

The Storage chapter must establish that distributed storage is a first-class system design concern, not an afterthought. Students who complete this chapter should be able to:

1. Select appropriate storage systems for different ML workloads
2. Calculate storage bandwidth requirements and identify bottlenecks
3. Design checkpoint storage that ensures training reliability
4. Understand feature store architecture and when it matters
5. Apply consistency model knowledge to storage system selection

The key insight: **Storage architecture determines whether your distributed ML system achieves its theoretical performance or spends half its time waiting for I/O.**

---

*Review prepared by Ion Stoica, applying experience from building Apache Spark and Ray to provide guidance on distributed storage fundamentals for ML systems education.*
