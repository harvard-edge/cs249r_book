---
title: "Follow One Request Through an AI Factory"
date: "2026-11-12"
author: "Vijay Janapa Reddi"
categories: ["essay"]
description: "A token factory may report high aggregate throughput while individual users still wait too long for an answer. Following a single request across the stack reveals where time accumulates and where engineering decisions matter."
draft: true
image: "figures/gtc-berlin-stack.jpg"
---

<!-- buttondown-editor-mode: plaintext -->

*From the Author's Desk at [MLSysBook.ai](https://mlsysbook.ai) · Field notes on turning AI systems into a teachable engineering discipline, connecting how modern AI infrastructure is built and operated in industry to the first principles we teach the next generation.*

---

In the [previous essay](https://mlsysbook.ai/newsletter/), I wrote from NVIDIA GTC Berlin about the rise of the AI factory, an industrial installation designed to convert electrical power and data into a continuous stream of tokens. Operators measure the capacity of that facility in tokens generated per second per megawatt. A person using an application built on that facility, however, never experiences aggregate cluster throughput. The user experiences the wall-clock time of a single request from submission to completion.

As I argued in [The Model Is Not the Product](https://mlsysbook.ai/newsletter/posts/2026/2026-03-10_the-model-is-not-the-product.html), the model weights account for only a fraction of what determines whether a production system succeeds. The surrounding serving, retrieval, and orchestration infrastructure governs both cost and reliability.

Following a single request gives an AI systems engineer a concrete diagnostic starting point across that surrounding system. Before an answer appears on a screen or actuates a machine, a request accepts network input, waits for an available worker, retrieves supporting context, processes the prompt to produce its first token, and decodes subsequent tokens one step at a time. A response deadline belongs to that entire path. Measuring the model in isolation leaves most of the deadline unexplained.

![A layered AI stack displayed during the GTC Berlin keynote.](figures/gtc-berlin-stack.jpg)

**A single request crosses every layer of the stack.** The GTC Berlin keynote placed Applications above Models, AI Factories, Compute, and Energy. A live request traverses those boundaries on every call. Photograph from my visit.

![Follow the complete request.](figures/essay-06-request-path.png)

**Follow the complete request.** A request path separates time spent waiting in queues from time spent retrieving context and generating tokens.

## Where Time Accumulates Along the Path

Consider a production service whose responses slow down whenever traffic peaks at the start of a workday. The model itself may execute each batch in nearly the same time as it did during quiet hours. Individual requests still miss their deadlines because they spend longer waiting in an admission queue for an available worker. In that regime, the queue has grown even though model execution speed has remained steady.

Separating waiting time from execution time changes which intervention makes sense. Optimizing a matrix multiplication kernel inside the model helps the queue only to the extent that it frees workers sooner. Reducing unnecessary work before the model, adjusting batching and admission rules, or adding serving capacity addresses queueing delay directly, and each option carries a different cost.

Recording timestamps at the boundaries between stages allows an engineer to distinguish time spent waiting from time spent computing. Those boundary measurements also expose whether a small fraction of long-tail requests is being hidden by an acceptable average latency.

Even inside the model execution stage, token generation divides into two distinct physical regimes. During the **prefill** phase, the accelerator processes all input tokens from the user prompt and retrieved documents in parallel to produce the first output token, which makes prefill largely compute bound on long inputs. During the **decode** phase, the model generates one token at a time, reading the model weights and the growing key-value (KV) cache from high-bandwidth memory on every step, which makes decoding largely memory-bandwidth bound. A service that feels sluggish before the first word appears has a different bottleneck from a service that pauses between words once generation begins.

## An Improvement Must Survive the Whole Service

Because a request crosses multiple stages, an optimization that improves one component in isolation must still be evaluated across the complete service. Suppose an engineering team reduces model weight precision from 16 bits to 8 bits to conserve high-bandwidth memory, a trade-off we first examined on a single node in [The Builder's Gap](https://mlsysbook.ai/newsletter/posts/2026/2026-04-21_the-builders-gap.html). The smaller memory footprint frees capacity for larger KV caches and allows more concurrent requests to share an accelerator.

Whether end-to-end response time improves depends on the processor's native support for 8-bit arithmetic, the overhead of the runtime kernels, and how the scheduler uses the newly available memory. At the same time, the team must verify that the lower-precision representation preserves answer quality on the domain tasks the application actually performs.

The same end-to-end discipline governs retrieval and reasoning depth. Returning additional source documents from a vector index can help a model answer a complex technical question, but those extra passages lengthen the input prompt, increase prefill computation, and expand the KV cache held in memory throughout decoding. Allowing a reasoning model to generate hundreds of intermediate chain-of-thought tokens before answering can improve accuracy on hard problems, yet those hidden tokens consume decode bandwidth and occupy worker slots that other requests are waiting to use. An engineering change succeeds only when the gain in answer quality justifies the additional time, memory, and power consumed along the path.

## A Practical Investigation

Return to the maintenance assistant introduced in the Berlin essay, which helps field technicians diagnose equipment faults using technical manuals. Suppose response latency spikes whenever a shift begins and dozens of technicians submit questions simultaneously. A disciplined investigation starts by instrumenting four boundary timestamps for every request: when the request arrives at the gateway, when document retrieval starts and finishes, when model prefill begins and emits the first token, and when the final token is delivered.

Alongside those timestamps, the engineer records the work each request demands, including input token count, retrieved document count, generated output token count, and whether the request was retried or canceled. Without those measurements, two requests that perform very different amounts of work can be misread as unstable system performance.

Comparing busy periods against quiet periods for requests of similar length reveals where the additional delay enters. If the extra time accumulates before prefill begins, the investigation focuses on admission queues, worker concurrency, and KV-cache capacity. If execution time per token degrades during busy periods, the investigation turns to batch composition, memory bandwidth contention, and network traffic across nodes.

## Choosing an Intervention and Accounting for Its Cost

Once measurements identify the dominant source of delay, the engineer must select an intervention and account for its trade-offs across the whole user population. If queueing delay dominates because long-context requests monopolize workers, the team might cap maximum concurrency so that admitted requests receive predictable latency. That policy protects active users, but it also forces excess arrivals to wait outside the service or receive a retry message. A complete engineering evaluation must report those rejected or delayed arrivals alongside the latency of completed requests. Otherwise, the service appears faster merely because it stopped accepting hard work.

Alternatively, the team might separate short interactive queries from long diagnostic runs, routing each class to its own worker pool or scheduling queue. Isolating short requests prevents them from stalling behind multi-thousand-token generations, but it requires the team to provision enough capacity so that long requests still finish within an acceptable window. Every scheduling policy distributes waiting across users and workloads. The application's requirements determine which distribution is acceptable, and boundary measurements show whether the deployed policy achieves it.

## Practicing the Habit at Any Scale

Tracing a request from arrival to final token builds a core habit of AI systems engineering: connect the user-visible symptom to a physical mechanism before changing the system. A slow response is a symptom. KV-cache exhaustion or head-of-line queue blocking is a mechanism that an engineer can model, measure, and fix.

Inside an AI factory, this habit governs how megawatts of installed computing capacity translate into responsive applications. Learners and educators can practice the exact same reasoning on a laptop or a single workstation, using the analytical models in [MLSys·im](https://mlsysbook.ai/mlsysim/) to predict where the first wall will appear before running an experiment. The third essay in this series examines how we structure university courses and hands-on assignments around these decisions so that students learn to defend a systems trade-off with evidence.

## Continuing the Conversation

Selecting one of the options in the quick poll below records your response in a single click, and the confirmation page includes an optional box if you would like to share a note on what you observe when profiling your own services.

{{ survey.essay_06_pulse }}

You can also reply directly to this email to share which stage boundary most often surprises your team when you profile an AI service under load.

Vijay

---

[MLSysBook](https://mlsysbook.ai) · [Taxonomy](https://mlsysbook.ai/taxonomy/) · [TinyTorch](https://mlsysbook.ai/tinytorch/) · [MLSys·im](https://mlsysbook.ai/mlsysim/) · [StaffML](https://mlsysbook.ai/staffml/) · [Subscribe](https://mlsysbook.ai/newsletter/) · [GitHub](https://github.com/harvard-edge/cs249r_book)
