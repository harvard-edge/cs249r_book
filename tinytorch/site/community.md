# Community Ecosystem

<iframe src="community/community.html" width="100%" height="600px" style="border: 1px solid #eee; border-radius: 8px; margin-bottom: 20px;"></iframe>

**See yourself on the TinyTorch Globe!** Create an account and join the global community to track your progress and connect with other builders.

```{mermaid}
:align: center
:caption: TinyTorch Community Ecosystem
flowchart TB
 %% Top benefit
 B["Benefit: Build Systems Intuition & Understand ML Frameworks<br/>From Tensors to Systems: Debug, Implement, Optimize"]

 %% Center
 C(("TinyTorch<br/>Community"))

 %% Main branches
 L["Progress & Connection"]
 R["Competition & Growth"]
 S["Learning & Support Ecosystem"]

 %% Left branch leaves
 L1["Share<br/>Progress"]
 L2["Connect with<br/>Builders"]
 L3["Add Yourself<br/>to the Map"]

 %% Right branch leaves
 R1["Torch Olympics<br/>(Capstone)"]
 R2["Leaderboard"]

 %% Support branch leaves
 S1["Course Structure<br/>& Resources"]
 S2["Historical<br/>Milestones"]
 S3["FAQ &<br/>Troubleshooting"]

 %% Connections
 B --> C
 C --> L
 C --> R
 C --> S

 L --> L1
 L --> L2
 L --> L3

 R --> R1
 R --> R2

 S --> S1
 S --> S2
 S --> S3

 %% Styling - TinyTorch brand colors
 classDef benefit fill:#fff3cd,stroke:#333,stroke-width:1px,color:#111;
 classDef center fill:#1b3a5f,stroke:#333,stroke-width:2px,color:#fff;
 classDef branch fill:#e1f5ff,stroke:#333,stroke-width:1px,color:#111;
 classDef leafProgress fill:#d4edda,stroke:#333,stroke-width:1px,color:#111;
 classDef leafCompete fill:#f8d7da,stroke:#333,stroke-width:1px,color:#111;
 classDef leafSupport fill:#e2d5f1,stroke:#333,stroke-width:1px,color:#111;

 class B benefit;
 class C center;
 class L,R,S branch;
 class L1,L2,L3 leafProgress;
 class R1,R2 leafCompete;
 class S1,S2,S3 leafSupport;
```

**Learn together, build together, grow together.**

TinyTorch is more than a course—it's a growing community of students, educators, and ML engineers learning systems engineering from first principles.

### Community Dashboard (Available Now )

Join the global TinyTorch community and see your progress:

```{image} _static/images/tinytorch-community.png
:alt: TinyTorch User Journey
:width: 100%
:align: center
```

## How to Join the TinyTorch Community:

There are two primary ways to join and engage with the TinyTorch community:

### Option 1: Explore the Live Dashboard

You can explore the interactive TinyTorch Community Dashboard directly in your browser:

<p align="center">
 <a href="community/community.html" target="_blank" class="btn btn-primary btn-lg" style="background-color: #ff6600; color: white; padding: 15px 30px; text-align: center; text-decoration: none; display: inline-block; font-size: 20px; margin: 10px 2px; cursor: pointer; border-radius: 8px; border: none;">
 Explore Dashboard <span class="badge badge-light"></span>
 </a>
</p>

### Option 2: Join via TinyTorch CLI

Download the TinyTorch CLI to set up your environment, manage your profile, and contribute to community statistics directly from your terminal.

```bash
# Join the community (first-time setup)
tito setup

# Or, if already set up, log in
tito community login

# View your profile
tito community profile

# Check your community status
tito community status

# Open the community map
tito community map
```

**Features:**
- **Anonymous profiles** - Join with optional information (country, institution, course type)
- **Cohort identification** - See your cohort (Fall 2024, Spring 2025, etc.)
- **Progress tracking** - Automatic milestone and module completion tracking
- **Privacy-first** - All data stored locally in `.tinytorch/` directory
- **Opt-in sharing** - You control what information to share

**Privacy:** All fields are optional. We use anonymous UUIDs (no personal names). Data is stored locally in your project directory. See [Privacy Policy](PRIVACY_DATA_RETENTION.md) for details.


## Connect Now
Validate your setup and track performance improvements:

```bash
# Quick setup validation (after initial setup)
tito benchmark baseline

# Full capstone benchmarks (after Module 20)
tito benchmark capstone

# Submit results to community (optional)
# Prompts automatically after benchmarks complete
```

**Baseline Benchmark:**
- Validates your setup is working correctly
- Quick "Hello World" moment after setup
- Tests: tensor operations, matrix multiply, forward pass
- Generates score (0-100) and saves results locally

**Capstone Benchmark:**
- Full performance evaluation after Module 20
- Tracks: speed, compression, accuracy, efficiency
- Uses Module 19's Benchmark harness for statistical rigor
- Generates comprehensive results for submission

**Submission:** After benchmarks complete, you'll be prompted to submit results (optional). Submissions are saved locally and can be shared with the community.

See [TITO CLI Reference](tito/overview.md) for complete command documentation.


## For Educators

Teaching TinyTorch in your classroom?

**[See Getting Started - For Instructors](getting-started.html#instructors)** for:
- Complete 30-minute instructor setup
- NBGrader integration and grading workflows
- Assignment generation and distribution
- Student progress tracking and classroom management


## Recognition & Showcase

Built something impressive with TinyTorch?

**Share it with the community:**
- Post in [GitHub Discussions](https://github.com/mlsysbook/TinyTorch/discussions) under "Show and Tell"
- Tag us on social media with #TinyTorch
- Submit your project for community showcase (coming soon)

**Exceptional projects may be featured:**
- On the TinyTorch website
- In course examples
- As reference implementations


## Stay Updated

**GitHub Watch**: [Enable notifications](https://github.com/mlsysbook/TinyTorch) for releases and updates

**Follow Development**: Check [GitHub Issues](https://github.com/mlsysbook/TinyTorch/issues) for roadmap and upcoming features


**Build ML systems. Learn together. Grow the community.**
