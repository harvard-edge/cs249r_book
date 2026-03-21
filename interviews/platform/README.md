# The MLSys Interactive Platform (POC)

This directory contains the Proof of Concept for the interactive ML Systems simulator, designed to be deployed on Hugging Face Spaces.

## Architecture

*   **Backend:** FastAPI (Python) wrapping the `mlsysim` engine and serving the `corpus.json`.
*   **Frontend:** React / Next.js (or Gradio for rapid prototyping) providing a "Mission Control" terminal aesthetic.
*   **Data Source:** `../corpus.json` (generated from the Markdown flashcards).
