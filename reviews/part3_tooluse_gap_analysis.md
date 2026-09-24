# Gap Analysis: Part III: Tool Use (Volume III)

## 1. Covered Topics
Based on the provided chapters (Chapter 7: Tool Calling and Chapter 8: Sandboxes) and supplementary files, the current text comprehensively covers:
- **Chapter 7 (Tool Calling):**
  - Tool subsystem architecture and complete mediation
  - Tool interface schemas (JSON schema bounds, strict validation, closed-world assumption)
  - Interoperable tool discovery (MCP) and JIT schema injection/retrieval
  - Idempotent action execution and settlement before retry
  - Observation stream truncation, terminal output sanitization
  - Asynchronous tool dispatch
  - Toolkit granularity partitioning (narrow vs. general tools)
- **Chapter 8 (Sandboxes):**
  - Agent threat model (prompt injection, instruction-data cohabitation)
  - In-process sandbox failures (reflection, FFI memory corruption, ambient authority)
  - Capabilities, credential brokering, and monotonic attenuation
  - Choosing isolation envelopes (Containers, gVisor, microVMs like Firecracker)
  - Agent workspace (Copy-on-Write) and persistence
  - Egress control and network proxies
  - Tracking untrusted data (taint tracking)
  - Sandbox pools and fast-reset mechanisms

## 2. Strengths
- **Rigorous Systems Perspective:** The text successfully maps classical operating system concepts (e.g., complete mediation, file descriptor tables, capability attenuation, Ring 0 vs. Ring 3) to agentic workflows.
- **Concrete Threat Modeling:** Explaining the "instruction-data cohabitation" dilemma and outlining precisely why in-process Python `exec()` sandboxes fail (reflection, C-extensions) is a masterclass in AI security engineering.
- **Actionable Constraints:** Appendix B and the sections on JSON Schema anti-patterns give practitioners immediate, usable guidelines for defining tools (e.g., closed schemas, avoiding implicit units).

## 3. Missing Topics
The following topics are missing or under-represented in the current draft and represent notable gaps against 2025/2026 industry realities.

### A. Computer Use and GUI Automation Tools
- **Justification:** Foundation models now natively support "Computer Use" (e.g., Anthropic's Claude Computer Use APIs). Agents interact with desktop GUI environments, click coordinates, and read visual accessibility trees or raw DOMs, not just REST APIs and bash shells.
- **Priority:** **Critical**
- **Suggested Additions:**
  - `Ch 7: Computer Use and Visual Actuation` (handling coordinate scaling, screenshot pagination, and bounding box mapping).
  - `Ch 8: Display Server and Browser Sandboxing` (isolating X11/Wayland, Playwright/Puppeteer profiles).

### B. Hardware Accelerator (GPU) Sandboxing
- **Justification:** Machine Learning Systems engineers frequently build agents that write, train, or perform inference on *other models*. Giving an agent a microVM (like Firecracker) inherently strips GPU access, but agents need CUDA to run ML tasks. Securely exposing vGPUs or PCIe passthrough to agent sandboxes is a major missing piece.
- **Priority:** **Important**
- **Suggested Additions:** `Ch 8: Hardware Accelerator Isolation` (SR-IOV, vGPU partitioning, and device-level constraints).

### C. Delegated Auth and 3-Legged OAuth Flows
- **Justification:** Chapter 8 covers "credential brokering" (keeping secrets out of the sandbox). However, it assumes a server-to-server or service account model. Real-world consumer/enterprise agents must act *on behalf of a human user* through 3-legged OAuth flows (intercepting a tool call to ask the user to authenticate via a browser, then resuming).
- **Priority:** **Important**
- **Suggested Additions:** `Ch 8: Delegated User Authentication and OAuth Brokers`.

### D. Tool Versioning and Schema Evolution
- **Justification:** External APIs deprecate endpoints and change signatures. If a tool schema changes, cached prompt prefixes are invalidated, and fine-tuned models might fail. How runtimes handle backward compatibility for agent tools is a key systems engineering concern.
- **Priority:** **Important**
- **Suggested Additions:** `Ch 7: Tool Lifecycle and Backward Compatibility`.

### E. Semantic Caching and Memoization
- **Justification:** While idempotency prevents destructive retries, many tool calls are pure reads (e.g., web search, database lookups). Runtimes need semantic caching for tool results to save massive amounts of generation latency and API costs.
- **Priority:** **Nice-to-Have**
- **Suggested Additions:** `Ch 7: Semantic Result Caching for Read-Only Tools`.

## 4. Under-Developed Areas

- **Code Execution Environments (REPL vs. Script):** The text focuses on running bash commands or executing static files. It briefly under-develops stateful REPL execution (e.g., Jupyter Kernels), where variables and imports persist across multiple tool calls without committing to disk. *Suggested fix: Add a section in Ch 7 on "Stateful REPL vs. Ephemeral Script Execution."*
- **Database Query Risks:** While the text references dropping production tables, executing SQL as a tool requires its own isolation patterns (e.g., read-only replica routing, row-level security for multi-tenant agents). *Suggested fix: Expand credential brokering to include dynamic database lease generation.*
- **Model Context Protocol (MCP) Depth:** Although MCP is listed in the covered topics, the text must ensure it distinguishes between MCP's role as a *discovery and transport* layer versus the runtime's role as the *authorization* layer (which Principle 1 accurately identifies). Expanding on how MCP SSE/WebSockets are proxied through egress controls would bridge Ch7 and Ch8 effectively.
