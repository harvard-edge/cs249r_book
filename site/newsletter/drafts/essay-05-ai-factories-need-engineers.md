---
title: "NVIDIA GTC Berlin: AI Factories Will Need AI Engineers"
date: "2026-10-29"
author: "Vijay Janapa Reddi"
categories: ["essay"]
description: "GTC Berlin framed the modern data center as a token factory that can be pre-integrated into a container and deployed in 14 days. Hardware integration can be containerized; the engineering judgment required to turn megawatts and tokens into useful work cannot."
draft: true
image: "figures/gtc-berlin-jensen-closeup.jpg"
---

<!-- buttondown-editor-mode: plaintext -->

*From the Author's Desk at [MLSysBook.ai](https://mlsysbook.ai) · Field notes on turning the frontier of AI into a teachable engineering discipline, connecting how modern AI systems are built in industry to the first principles we teach the next generation.*

---

It has been a few months since my last essay here. Moving to Zürich for my academic sabbatical kept me away from this desk, and last week I traveled from Zürich to NVIDIA GTC Berlin.

## From Steam Engines to Token Factories

Walking the floor in Berlin brought me back to an observation from the first essay in this series ([The Shift Toward AI Engineering](https://mlsysbook.ai/newsletter/posts/2026/2026-01-27_the-shift-toward-ai-engineering.html)): if AI is the new electricity, our universities have been training almost everyone to use the appliances while very few people learn how to build and operate the power plants.

In Berlin, that comparison ceased to be a metaphor. Across the keynote and technical sessions, Jensen Huang framed the current moment as a new Industrial Revolution centered on the **AI factory**, a specialized manufacturing plant whose commodity output is **tokens**. A traditional cloud data center stores files and runs thousands of independent software services for different tenants. An AI factory takes electrical power and data as raw inputs and manufactures a continuous stream of tokens: floating-point representations of intelligence that translate into words, software code, molecular structures, video frames, or robot control actions.

Just as electrical utilities came to measure and price their output in kilowatt-hours, AI factories measure output in tokens generated per megawatt and per dollar. As reasoning models and agentic workflows generate hundreds or thousands of intermediate tokens before returning a single answer, token demand has surged. One keynote chart tracking agentic traffic across OpenRouter showed weekly token consumption rising 34-fold since the start of February, climbing from 3.9 trillion to 132.6 trillion tokens per week by early October. Token generation has become a heavy industrial production problem.

When I sit in an industry keynote as an academic and textbook author, I am not looking at these announcements the way a market analyst does. I am asking a curriculum question: if industry is building this infrastructure, what is the underlying body of knowledge required to engineer it, and how do we teach that discipline to the next generation.

Every past industrial transition faced that same human bottleneck. In the first Industrial Revolution, James Watt's steam engine converted coal into mechanical force, yet early steam plants wasted most of their fuel or failed under pressure until thermodynamics and mechanical engineering emerged to govern boilers, heat transfer, and control governors. A century later, electrical power grids forced electrical engineering to split from physics, and Henry Ford's moving assembly lines required a new discipline of industrial systems engineering.

Before Ford, skilled artisans built one carriage at a time on a workbench. Once production moved to an integrated assembly line, a bottleneck or tolerance error at a single station stalled the entire plant. Machine learning today sits in that exact transition. Most university courses still train students like artisans at a workbench, tuning a single model inside a notebook, while industry is bolting down continuous token factories.

![An AI factory illustration displayed during the GTC Berlin keynote.](figures/gtc-berlin-ai-factory.jpg)

**The AI factory as an industrial system.** Jensen Huang on stage at GTC Berlin, framing AI infrastructure as a token factory alongside European industrial partners. Photograph from my visit.

## What Arrives When a Token Factory Ships in a Container

One announcement in Berlin made this industrial framing physical. NVIDIA [introduced DSX POD](https://nvidianews.nvidia.com/news/nvidia-introduces-dsx-pod-ai-factories-to-accelerate-europes-infrastructure-deployment), a reference design for complete Vera Rubin AI factories built by ecosystem partners inside modular containers and validated before delivery. Inside each container, the compute racks, fiber-optic networking, power distribution busbars, fire suppression, and liquid cooling manifolds are wired, plumbed, and functionally tested on the manufacturing floor. Moving that integration out of the field reduces on-site installation at a prepared site from roughly six months to 14 days, with partner deliveries planned for early 2027.

![Cutaway render of the NVIDIA DSX POD modular container.](figures/nvidia-dsx-pod.jpg)

**A token factory inside a shipping container.** The NVIDIA DSX POD reference design pre-integrates Vera Rubin racks, networking cabling, power distribution, and liquid cooling manifolds on the factory floor before transport. (Image: NVIDIA)

A single container, however, is only one module inside a much larger machine. A second slide from the keynote showed how those modules come together inside a warehouse under the full-stack DSX platform. Seven distinct rack architectures, spanning Vera Rubin NVL72 compute nodes, NVLink 6 and Spectrum-6 switches, BlueField-4 compute and storage racks, and Groq 3 LPX inference racks, drop into the facility alongside CUDA-X libraries, platform runtimes, and open models.

![The NVIDIA DSX Full-Stack AI Factory Platform slide from the GTC Berlin keynote.](figures/gtc-berlin-dsx-warehouse.jpg)

**All the pieces dropping into the warehouse.** The full-stack AI factory combines seven specialized compute, switching, inference, and storage rack architectures with three layers of software libraries and models inside a single facility. Photograph from my visit.

Jensen Huang explained the economic logic behind this modular approach in a single sentence: an AI factory creates value when it starts producing tokens, not when construction is complete. A gigawatt-scale facility represents tens of billions of dollars in capital, so every month spent pulling cables on site delays the moment the plant begins generating tokens.

Compressing physical installation from six months to two weeks exposes the next bottleneck immediately. Physical construction no longer hides the systems engineering problem. Once those racks are bolted into the warehouse and connected to site power, water, and networking, turning installed hardware into sustained output depends entirely on the engineers operating the line. We can containerize hardware assembly on a manufacturing floor, whereas we cannot containerize the engineering judgment required to run changing workloads across that warehouse.

## From Energy to Tokens: The Stack Expands Downward

In earlier essays ([The Builder's Gap](https://mlsysbook.ai/newsletter/posts/2026/2026-04-21_the-builders-gap.html) and [From the Builder's Gap to a Builder's Stack](https://mlsysbook.ai/newsletter/posts/2026/2026-05-19_from-the-builders-gap-to-a-builders-stack.html)), we traced how failures cross four layers inside a single machine, from hardware through the runtime and framework to the application. At the close of both essays, I promised to examine what changes when that stack expands from one machine to a fleet. In Berlin, that fleet stack was presented on stage as five layers: **Energy**, **Compute**, **AI Factories**, **Models**, and **Applications**.

![Applications, models, AI factories, compute, and energy shown as five layers during the GTC Berlin keynote.](figures/gtc-berlin-stack.jpg)

**The five-layer stack in Berlin.** Applications sit above Models, AI Factories, Compute, and Energy. Decisions at the application layer activate physical constraints all the way down to the power grid. Photograph from my visit.

The quantitative results released alongside the [NVIDIA DSX platform](https://nvidianews.nvidia.com/news/european-ecosystem-builds-worlds-most-efficient-ai-factories-with-nvidia-dsx) show how tightly the top and bottom of this stack are now coupled. In most regions, electrical power sets the hard ceiling on how much compute an operator can deploy. Using dynamic power allocation (DSX MaxLPS), operators can generate the same number of tokens with up to 30 percent less power; in a proof of concept on HGX B200 systems, Lambda achieved 24 percent higher cluster token throughput and supported 19 percent more GPU capacity within a fixed power budget.

Cooling and grid stability follow the same pattern. Because Vera Rubin infrastructure supports liquid cooling with inlet temperatures up to 45 degrees Celsius, dry coolers can reject heat into ambient air without evaporative towers, saving up to 54 million liters of water per day at 2-gigawatt scale. In a trial with the U.K. National Grid, Nebius, Emerald AI, and EPRI, a Blackwell Ultra cluster reduced its power draw by up to 40 percent during simulated grid events while keeping critical workloads running.

Power provisioning, coolant temperatures, and grid demand response once belonged entirely to facilities staff. Inside a token factory, those physical limits interact directly with batch sizes, admission control, and cluster scheduling. The unit of computer architecture has expanded from the chip and the server to the entire energy-to-token conversion pipeline.

## Why Manufacturing Tokens Is Only Half the Problem

Measuring an AI factory in tokens per second per megawatt captures raw plant capacity, yet raw token volume does not tell us whether an application succeeds. An electrical power plant can measure its output in kilowatt-hours because every electron delivered to the grid performs the same physical work.

Tokens do not share that property. A thousand tokens generated from irrelevant retrieved documents, or a thousand reasoning tokens that arrive after an industrial control deadline has passed, consume the same electricity, memory bandwidth, and cooling capacity as a thousand tokens that solve a user's problem on time.

Consider an engineering assistant that retrieves maintenance manuals and answers diagnostic questions for field technicians. During a small demonstration, each query arrives alone, retrieves a brief passage, and returns an accurate answer within the deadline. When a shift begins and hundreds of technicians submit queries at once, response times degrade even though the model weights and the cluster hardware have not changed.

To diagnose that slowdown, an engineer has to look past aggregate token throughput and trace how each request moves through the system: waiting in an admission queue, retrieving documents, processing the prompt during prefill, and generating output tokens step by step during decoding. Longer retrieved passages and multi-step reasoning chains enlarge the key-value (KV) cache that each active request holds in high-bandwidth memory. Once high-bandwidth memory fills with KV-cache state, the serving runtime must restrict batch concurrency, which lowers arithmetic utilization on the accelerators while lengthening the queue of users waiting outside them.

Every candidate fix trades one system requirement against another. Shortening the retrieved context frees memory and reduces time to first token, while risking the loss of evidence the model needs for a correct maintenance instruction. Quantizing weights or KV-cache entries from 16 bits to 8 bits fits more concurrent requests into memory, provided the hardware executes the lower-precision format efficiently and answer quality holds on edge cases. The machine manufactures tokens; the AI engineer measures these competing effects across model, runtime, and hardware to turn token capacity into dependable work.

## The Builder's Gap Inside the Factory

In [The Builder's Gap](https://mlsysbook.ai/newsletter/posts/2026/2026-04-21_the-builders-gap.html), I summarized this diagnostic challenge in a single principle: failures propagate upward through the stack, whereas understanding must flow downward. At the scale of an AI factory, that asymmetry crosses organizational boundaries as well as technical layers.

As we saw in [The Model Is Not the Product](https://mlsysbook.ai/newsletter/posts/2026/2026-03-10_the-model-is-not-the-product.html), Meta recorded 419 unexpected hardware interruptions during the 54-day training run for Llama 3 across 16,384 H100 GPUs, or roughly one failure every three hours. In an installation of that size, application engineers track task accuracy, runtime engineers track batching and memory allocation, network engineers track collective communication and tail latency, and facility engineers track rack power and cooling loops. When a service misses its deadline or exceeds its power budget, the symptom appears at the top of the stack while the cause sits several layers below.

No individual engineer needs to design every subsystem from the transformer architecture down to the heat exchangers. A functioning engineering organization, however, needs a shared systems vocabulary so specialists can connect the evidence each role sees.

To make that shared structure explicit for our book and curriculum, we recently released a draft of the [ML Systems Taxonomy](https://mlsysbook.ai/taxonomy/), an open map of 101 subjects and 735 concepts across foundations, the system stack, the lifecycle, system qualities, and system types. Historically, academic courses have split between a compute-stack tradition and an operational MLOps tradition while leaving out datacenter infrastructure, energy, safety assurance, and physical AI. Our goal with the taxonomy is to join those traditions into a single teachable map of the discipline.

## Teaching the Physics Behind the Factory

Preparing engineers for this environment does not require placing a shipping-container cluster on every university campus. The physical laws that govern a gigawatt token factory are the same laws that govern a laptop or a thirty-dollar microcontroller, scaled across orders of magnitude. Memory capacity still determines whether weights and working state fit on the device. Memory bandwidth relative to arithmetic throughput still determines whether a processor spends its cycles computing or waiting on data movement. Queueing and scheduling still determine how shared hardware translates into individual response times.

That continuity is the foundation of our [Machine Learning Systems](https://mlsysbook.ai) textbook. Whether a student is calculating analytical bottlenecks, deploying onto a constrained microcontroller kit, or trying the new [MLSysBook Playground](https://mlsysbook.ai/games/), where interactive simulations such as *KV Cache Packer* and *Cluster Commander* turn memory fragmentation and fleet scheduling into sixty-second decisions, the underlying engineering habit is identical: predict the physical limit, measure the system at its boundaries, explain the gap between prediction and observation, and defend the resulting trade-off.

Over the next two essays, we will follow this thread from the factory floor into daily engineering practice and classroom design. The next essay traces a single request through an AI service to show where waiting and execution accumulate across the stack, and the third examines how we can teach and grade the decisions an AI engineer must defend.

## Continuing the Conversation

I write this newsletter to connect what is happening at the industrial frontier to the curriculum we are building in the book, and I want it to be a two-way exchange with the engineers, researchers, and educators who read it. Selecting one of the options in the quick poll below records your response in a single click, and the confirmation page includes an optional box if you would like to add a brief note:

{{ survey.gtc_berlin_pulse }}

You can also reply directly to this email to share where raw token throughput stops being the bottleneck in the systems your team builds or teaches, and which layer of the stack actually limits useful work. I read every reply, and the cases you send will inform the examples we examine in the next two essays.

Vijay

---

*P.S. If this essay helped clarify why token factories will need AI engineers, consider forwarding it to three colleagues:*

*An infrastructure or platform lead planning new cluster capacity who knows that plugging in the racks is where the systems engineering work begins.*

*An ML engineer whose model works in a notebook, and who wants to understand how queueing, KV-cache memory, and power limits govern production serving in [MLSysBook](https://mlsysbook.ai).*

*An educator designing next semester's AI curriculum who wants to explore the new [ML Systems Taxonomy](https://mlsysbook.ai/taxonomy/) and the [Instructor Hub](https://mlsysbook.ai/instructors/).*

[MLSysBook](https://mlsysbook.ai) · [Taxonomy](https://mlsysbook.ai/taxonomy/) · [Playground](https://mlsysbook.ai/games/) · [TinyTorch](https://mlsysbook.ai/tinytorch/) · [MLSys·im](https://mlsysbook.ai/mlsysim/) · [StaffML](https://mlsysbook.ai/staffml/) · [Subscribe](https://mlsysbook.ai/newsletter/) · [GitHub](https://github.com/harvard-edge/cs249r_book)
