---
title: "The Shift Toward AI Engineering"
date: "2026-01-27"
author: "Vijay Janapa Reddi"
description: "<!-- buttondown-editor-mode: plaintext -->My dad told me if I didn't study computer science in college, he'd throw me out of the house. So I did. In hindsight, he was right. CS gave me the foundation..."
categories: ["essay"]
image: "https://assets.buttondown.email/images/7c78f49d-9110-4132-88e9-a049e4176c49.png?w=960&fit=max"
---

<!-- buttondown-editor-mode: plaintext -->My dad told me if I didn't study computer science in college, he'd throw me out of the house. So I did. In hindsight, he was right. CS gave me the foundation for everything I've done over the past 25 years, the way of thinking and problem-solving instincts that still matter every day. 

My career didn’t stop at CS. I studied computer engineering and electrical engineering, and over time worked across the stack, from hardware and systems to software and machine learning. Seeing the same constraints resurface at every layer is what made the systems gap impossible to ignore.

That perspective is what makes the current moment in AI stand out to me. Signals from education and industry suggest the landscape has expanded more than most people realize.

## The Shift You Can See in the Data

Computer science enrollment is down at [62% of computing programs](https://cra.org/crn/2025/10/cerp-pulse-survey-a-snapshot-of-2025-undergraduate-computing-enrollment-patterns/) this year. CS, software engineering, and information systems are all declining. Meanwhile, [66% of programs report](https://cra.org/crn/2025/10/cerp-pulse-survey-a-snapshot-of-2025-undergraduate-computing-enrollment-patterns/) that recent graduates are struggling to find jobs. The “learn to code” promise that defined a generation of career advice isn’t landing the way it used to. That makes sense. Coding was never the destination, it was the foundation.

But this isn’t a retreat from technology, and it’s not a sign that CS doesn’t matter. It’s a redirection. Students are asking a more nuanced question: what kind of tech career will last?

The fields that are *growing*? Computer engineering. Cybersecurity. AI. [Engineering overall is up 7.5%](https://www.insidehighered.com/news/admissions/traditional-age/2025/11/11/short-term-credentials-bolster-enrollment-boom). Students aren’t abandoning tech. They’re rethinking what kind of tech career will last.

The [CRA report](https://cra.org/crn/2025/10/cerp-pulse-survey-a-snapshot-of-2025-undergraduate-computing-enrollment-patterns/) captures it well: students are gravitating toward majors that feel "more physical and less susceptible to the impact of AI." They want to build real-world devices, not just write code that might get automated. They're looking for skills that feel durable.

I think this signals something important. And at the same time, there’s a bigger picture that’s easy to miss.

## The Gap That's Quietly Widening

Here’s what I’ve observed after years of teaching computer and machine learning systems, and working with industry partners: everyone wants to use AI, but very few people understand how to *build* the systems that make it work at scale.

Who optimizes models to run on phones? Who designs the chips, builds the frameworks, creates the deployment pipelines? Who debugs why a model that worked perfectly in a notebook crashes in production? Who figures out how to serve millions of users without the infrastructure costs exploding?

These questions don't fit neatly into computer science alone, though CS fundamentals remain essential. They're not pure software engineering either. And they're not just electrical engineering.

They require a different kind of thinking—one that spans the full stack from silicon to cloud, from training to deployment, from algorithms to real-world constraints.

## What Is AI Engineering?

I've started calling this AI Engineering. Not as a replacement for existing fields. CS, EE, and software engineering remain foundational. But building production AI systems requires a synthesis of skills that don't live neatly in any one discipline.

Think about what it actually takes to deploy an AI system that works:

**The ML theory:** Optimization, statistical learning, how models generalize and when they fail. This is the ML research foundation—understanding *why* things work, not just *that* they work.

**The ML systems:** Distributed training, hardware acceleration, memory hierarchies, serving infrastructure. This is what makes AI run efficiently at scale, the difference between a demo and a product.

**The ML applications:** Problem framing, data collection, domain constraints, what "good enough" actually means in a specific context. This is what connects AI to real-world impact.

Most education gives you one of these. Maybe two if you're lucky. Almost no one gets all three.

AI Engineering sits at the intersection:

![CleanShot 2026-01-22 at 21.11.00@2x.png](https://assets.buttondown.email/images/ef8698a9-e990-4df9-ad5e-15fc09213911.png?w=960&fit=max)

The ML Systems Researcher lives in the overlap of theory and systems. The Data Scientist bridges theory and application. The MLOps Engineer connects systems and application. The AI Engineer needs to move fluidly across all three. Not as an expert in everything, but with enough literacy across the stack to see how the pieces connect, and enough depth in at least one area to actually build things.

## Why This Matters Right Now

Andrew Ng famously said that AI is the new electricity. He was right. But here's what's easy to miss: if AI is electricity, we're training everyone to use the appliances while almost no one learns how to build the power plants.

Look at what's actually happening. [OpenAI has committed to spending $1.15 trillion on infrastructure](https://tomtunguz.com/openai-hardware-spending-2025-2035/) over the next decade. Microsoft just opened [Fairwater, the world's most powerful AI datacenter](https://blogs.microsoft.com/blog/2025/09/18/inside-the-worlds-most-powerful-ai-datacenter/). It covers 315 acres, required 26.5 million pounds of structural steel, and runs 120 miles of underground cable. Analysts predict that [building AI infrastructure globally will exceed $1 trillion by 2030](https://www.datacenters.com/news/openai-and-the-trillion-dollar-ai-infrastructure-race), including data centers, power generation, cooling systems, and semiconductor supply chains.

These aren't software problems. They're engineering problems. Physical problems. Power, cooling, hardware, optimization, efficiency.

And that's only half the picture. AI isn't just running in massive data centers. It's also moving to the edge.

## AI Is Going Everywhere

The [TinyML market is growing at 34% annually](https://finance.yahoo.com/news/tinyml-market-analysis-report-2025-152200302.html), driven by the need to run AI on devices with severe constraints: phones, wearables, medical devices, agricultural sensors, industrial equipment. These systems can't send data to the cloud and wait for a response. They need to process locally, in real time, with minimal power.

NVIDIA calls this ["Physical AI"](https://nvidianews.nvidia.com/news/nvidia-releases-new-physical-ai-models-as-global-partners-unveil-next-generation-robots). Jensen Huang recently said that "the ChatGPT moment for robotics is here." Companies like Boston Dynamics, Caterpillar, and LG are building robots that can reason, plan, and act in the real world. [Robotics is now the fastest-growing category on Hugging Face](https://techcrunch.com/2026/01/05/nvidia-wants-to-be-the-android-of-generalist-robotics/).

This is where the "physical computing" instinct that students have makes sense. AI that interacts with the real world requires understanding hardware constraints, power budgets, latency requirements, and sensor systems. You can't just train a model and throw it over the wall. You have to engineer the entire system.

The students shifting toward computer engineering and hardware? They're sensing this. The companies scrambling to hire people who understand deployment and optimization? They're responding to it.

But we do not yet have the educational infrastructure to train AI Engineers at scale. The knowledge exists. It is just scattered across research groups, companies, and a handful of courses. Most people never see the full picture.

That is the gap the [MLSys Book](https://mlsysbook.ai) exists to fill. Not by replacing existing disciplines, but by making the connective tissue visible. And that is what this community is here to build together.

And if you want to get hands on, [TinyTorch](https://mlsysbook.ai/tinytorch/intro.html) lets you build your own ML framework from scratch. Because you cannot debug what you did not build.

One last thing that matters to me. The reason this book and these materials are freely available is simple. The ability to build and reason about real AI systems should not depend on access to a specific lab, company, or country.

A lot of the most important engineering knowledge today is still learned through proximity. This project is an attempt to make that knowledge explicit, so more people can participate in building AI systems that are reliable, accountable, and actually work in the real world.

## What’s Coming

Over the coming months, I will share what we are learning as this work evolves. New chapters as they are ready. Updates in the field. Hands on labs that surface real system tradeoffs. Insights from practitioners building and deploying production AI. Resources for educators teaching this emerging field.

This is a work in progress, shaped by the people who engage with it. If there is a topic you want to go deep on, efficient inference, on device learning, ML operations, or something else entirely, tell me. I read the messages that come in, and they shape what we prioritize next.

If this resonated, forward it to someone who should see it. A student trying to make sense of their path. A colleague watching how the work is changing. Someone who wants to understand what it actually takes to build AI systems in the real world.

Vijay

---
**Supporting the work:**

The MLSys Book, TinyTorch, and the TinyML Kit materials are freely available. Their ongoing development is supported by Harvard, [community donations](https://opencollective.com/mlsysbook), industry sponsorship, and partnerships.

Many people support the project simply by giving it [GitHub ⭐ stars](https://github.com/harvard-edge/cs249r_book) and providing feedback.

**Sources:**

- [CRA CERP Pulse Survey: 2025 Undergraduate Computing Enrollment Patterns](https://cra.org/crn/2025/10/cerp-pulse-survey-a-snapshot-of-2025-undergraduate-computing-enrollment-patterns/)
- [Inside Higher Ed: Short-Term Credentials Bolster Enrollment Boom](https://www.insidehighered.com/news/admissions/traditional-age/2025/11/11/short-term-credentials-bolster-enrollment-boom)
- [OpenAI's $1 Trillion Infrastructure Spend](https://tomtunguz.com/openai-hardware-spending-2025-2035/)
- [Inside the World's Most Powerful AI Datacenter](https://blogs.microsoft.com/blog/2025/09/18/inside-the-worlds-most-powerful-ai-datacenter/)
- [OpenAI and the Trillion Dollar AI Infrastructure Race](https://www.datacenters.com/news/openai-and-the-trillion-dollar-ai-infrastructure-race)
- [TinyML Market Analysis Report 2025-2029](https://finance.yahoo.com/news/tinyml-market-analysis-report-2025-152200302.html)
- [NVIDIA Releases New Physical AI Models](https://nvidianews.nvidia.com/news/nvidia-releases-new-physical-ai-models-as-global-partners-unveil-next-generation-robots)
- [Nvidia Wants to Be the Android of Generalist Robotics](https://techcrunch.com/2026/01/05/nvidia-wants-to-be-the-android-of-generalist-robotics/)
