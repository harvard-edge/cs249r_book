# Community Ecosystem

<iframe src="community/community.html" width="100%" height="600px" style="border: 1px solid #eee; border-radius: 8px; margin-bottom: 20px;"></iframe>

**See yourself on the TinyTorch Globe!** Create an account and join the global community to track your progress and connect with other builders.

<p align="center">
<img src="_static/images/diagram_tiny-commununity.png" alt="TinyTorch Community Ecosystem" width="100%">
</p>

**Learn together, build together, grow together.**

TinyTorch is more than a course—it's a growing community of students, educators, and ML engineers learning systems engineering from first principles.

## Community Dashboard (Available Now )

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

**Privacy:** All fields are optional. We use anonymous UUIDs (no personal names). Data is stored locally in your project directory.


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

**See the [Getting Started](getting-started) guide** for:
- Complete 30-minute instructor setup
- NBGrader integration and grading workflows
- Assignment generation and distribution
- Student progress tracking and classroom management


## Recognition & Showcase

Built something impressive with TinyTorch?

**Share it with the community:**
- Post in [GitHub Discussions](https://github.com/harvard-edge/cs249r_book/discussions) under "Show and Tell"
- Tag us on social media with #TinyTorch
- Submit your project for community showcase (coming soon)

**Exceptional projects may be featured:**
- On the TinyTorch website
- In course examples
- As reference implementations


## Stay Updated

**GitHub Watch**: [Enable notifications](https://github.com/harvard-edge/cs249r_book) for releases and updates

**Follow Development**: Check [GitHub Issues](https://github.com/harvard-edge/cs249r_book/issues) for roadmap and upcoming features


**Build ML systems. Learn together. Grow the community.**
