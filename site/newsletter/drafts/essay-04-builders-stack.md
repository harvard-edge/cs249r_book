---
title: "From the Builder's Gap to a Builder's Stack"
date: "2026-05-21"
author: "Vijay Janapa Reddi"
categories: ["essay"]
description: "Most ML engineers I interview can build a model. Many cannot tell you why it runs as fast as it does, costs what it costs, or whether it will fit on the hardware they plan to deploy it on. That gap has a shape. Three years of curriculum-building gave it a name: the Builder's Stack."
draft: true
image: "figures/essay-04-builders-stack.png"
---

<!-- buttondown-editor-mode: plaintext -->

Most ML engineers I interview can build a model. Many cannot explain, with numbers, why it runs as fast as it does, why it costs what it costs, or whether it will fit on the hardware they plan to deploy it on. That is not their fault. The field has been teaching the model and skipping the system underneath it.

Disciplines do not get planned into existence. They get distilled out of practice. Software engineering was named at a [NATO conference](https://en.wikipedia.org/wiki/Software_engineering#History) in 1968 to describe what programmers had been doing badly for fifteen years. Computer engineering split from electrical engineering when building computers became different enough from designing circuits to need its own training. Every other engineering discipline went through some version of the same arc: years of practice, then a decade of confused apprenticeship, then a shared vocabulary, then a curriculum. **AI engineering is in that arc right now.**

The textbook was the vocabulary stage of that arc. Eighteen months in, it was clear the book alone was not going to transmit the discipline. The rest of the curriculum followed, in the order the gaps surfaced.

[Last Thursday](https://www.linkedin.com/in/vijay-janapa-reddi-63a6a173/) I shipped the biggest release push to date: six artifacts pulled together in one repository, under one open license, as the single curriculum they had become. This issue is the retrospective. Not *"what got shipped,"* but *"why each piece had to exist."* Six origin moments, and one shape underneath them. I will name the shape at the end, because the field needs the name.

## The Six Things, and the Moments That Forced Them

**The textbook.** Computer architecture had [Hennessy and Patterson](https://www.elsevier.com/books/computer-architecture/hennessy/978-0-12-383872-8). Operating systems had Tanenbaum. Compilers had the dragon book. ML systems had a scattering of papers, blog posts, and slide decks that did not yet add up to a shared vocabulary. So the notes that had been accumulating turned into chapters, and the chapters turned into a book. It is the spine of this project, and it does what textbooks do: it gives a discipline a vocabulary and a structure. But teaching is not the same as transmitting, and somewhere in the second year it became clear the textbook alone was not going to transmit.

![**Machine Learning Systems**, forthcoming from MIT Press. Volume I in 2026, Volume II in 2027. The spine of the project, open source under CC BY-NC-SA.](figures/textbook-covers.jpg)

**TinyTorch.** In every other engineering discipline, students build. EE students wire breadboards. ME students machine parts. Civil engineers pour test slabs. ML had become the strange exception. Students learned what backpropagation *is* without ever implementing it, and then the field acted surprised when those same students could not predict where their training runs would break. That is not a defensible exception. So we built [TinyTorch](https://mlsysbook.ai/tinytorch/): a from-scratch ML framework you assemble module by module, from raw tensors up through transformers, profiling, and quantization. ([The Builder's Gap](https://buttondown.com/mlsysbook/archive/the-builders-gap-4483/) was the longer argument for why this rung matters; TinyTorch is the artifact that answers it.) That is how every other engineering discipline transmits its core machinery. There is no reason ours should not.

![**TinyTorch.** Twenty modules from raw tensors to a working transformer. Build the framework yourself; the framework teaches the discipline.](figures/tinytorch-landing.jpg)

**MLSys·im.** [MLSys·im](https://mlsysbook.ai/mlsysim) was an accident. The textbook needed numbers, actual calculations rather than hand-waving, and what started as a small calculator grew, then grew again, until it was clearly no longer a calculator. It had become the analytical engine that lets a student reason about systems trade-offs without owning a cluster. There are 22 codified systems walls in MLSys·im now, each drawn from the published literature, each a closed-form equation tied to a real physical limit. Memory bandwidth. Arithmetic intensity. Interconnect saturation. You can run them on a laptop and feel which wall is binding before you ever touch real silicon. Once we had an analytical engine, trade-off analysis could become a teachable activity rather than a thought experiment.

![**The 22 systems walls in MLSys·im.** Each row is a physical constraint with a closed-form equation. Run them on a laptop and feel which wall is binding before you ever touch silicon.](figures/mlsysim-walls-table.jpg)

**Hardware kits.** Even the best simulator has the same weakness: simulated constraints are easier to wave away than physical ones. So we wired the curriculum to actual hardware students can own: [Arduino, Seeed, Grove, Raspberry Pi](https://mlsysbook.ai/kits). The point is not that every learner needs to ship TinyML in production. The point is that you cannot deploy your model on the 10,000-node cluster you do not have, but you can absolutely close the loop on a thirty-dollar board on your desk that refuses to load your model. Either you face deployment as a physical reality or you do not. There is no middle path that produces engineers.

![**The XIAOML Kit.** A board, a camera, a screen, a microSD, a USB cable. Around thirty dollars, fits in your hand, refuses to load most modern models. Deployment as physical fact rather than abstraction.](figures/xiaoml-kit.png)

**StaffML.** [StaffML](https://staffml.ai) began with a single email. A student asked me how to prepare for an ML systems interview. There was no LeetCode for ML systems. Writing questions for that one student exposed the deeper problem: there was no good way for a learner to find *what they did not know they did not know.* "Interview prep" is the polite framing for what StaffML actually does. The honest framing is that most people overestimate what they understand about ML systems, and the only way to discover where the gaps live is to be tested against problems that resist surface answers. You cannot prompt your way out of physics.

![**StaffML in practice mode.** Filters that surface the gaps in a learner's mental model of ML systems. Questions are graded for what they reveal, not what they reward.](figures/staffml-practice-ui.png)

**The Instructor Hub.** The last failure was the slowest to admit. The materials worked in my classroom. They did not always work in someone else's. Educators do not need raw materials; they need scaffolding. A semester plan they can defend at a curriculum committee. Slides they can pick up on a Sunday afternoon. Rubrics that survive a TA changeover. So we packaged the curriculum as a [course in a box](https://mlsysbook.ai/instructors/) and released it under the same open license as everything else. Most teaching infrastructure stays locked inside individual universities. This one does not.

![**Lecture slides from the Instructor Hub.** The full deck ships open so a colleague can pick up the curriculum on a Sunday afternoon and teach it on Monday.](figures/teaching-slides.jpg)

That is the sequence. Six artifacts. Each one exists because the previous one stopped transmitting at some point, and that stopping point named the next gap.

## The Shape Underneath: The Builder's Stack

In retrospect, those six are not six tools. They are six rungs of one ladder.

Volume I gave readers the **ML Systems Stack**, a layered abstraction of what a single machine is made of, from silicon up through applications. Volume II gave them the **Fleet Stack**, the same kind of layered abstraction at warehouse scale, from infrastructure up through governance. Both stacks describe the *system*. They answer the question *what is the thing made of?*

What has been missing is a third stack. One that describes the *engineer*. And the third stack matters beyond any single classroom. AI is being taught right now to a billion people as users of systems somebody else built, and that gap is geographic as much as it is generational. The same kits that ship to Harvard ship to TinyML4D partner universities in Malawi, Colombia, Bangladesh, and dozens of other institutions across five continents. The same labs run in classrooms where the GPU budget is zero. Whether the next decade of AI is built by ten thousand organizations or by ten depends on whether the climb past rung one is available outside the institutions that already have it.

![**The Builder's Stack.** Six rungs of AI engineering competence and the open artifacts that get a learner to each rung. The third stack in the trilogy alongside the System Stack (Vol I) and the Fleet Stack (Vol II).](figures/essay-04-builders-stack.png)

> *Two stacks for what the system is. One for what the builder must be.*

Yes, the stack maps cleanly onto the six artifacts above. That alignment is the point. The same way the textbook distilled from a decade of notes, this stack distilled from the artifacts. We did not design a six-rung curriculum and build to it. We taught, hit a wall, built the missing piece, taught again, hit the next wall. The shape only became visible in retrospect. The map is real because the climb was real.

The rungs go bottom to top. **Use** at the foundation: call APIs, run notebooks, get something working. **Build** above it: implement the core machinery yourself, where intuition actually starts. **Model**: reason about trade-offs analytically before you commit hardware to them. **Deploy**: operate under physical limits that will not negotiate. **Test**: probe what you do not know you do not know, because self-assessment is the most overrated skill in this field. **Teach** at the top: hand the discipline to a stranger and watch whether it survives the handoff. Each rung depends on the rungs below it. Skipping is possible; transmitting is not.

### A five-question self-diagnostic

Before reading further, place yourself. Answer honestly. Where you stop is your current rung.

1. Have you implemented backprop, an optimizer, or a transformer block in code you wrote yourself? **(Build)**
2. Can you predict, before launching a training run, whether it will fit on the hardware you plan to use? **(Model)**
3. Have you closed the loop on a deployed model that refused to load on the target device? **(Deploy)**
4. Have you written a test or an interview question that revealed something *you yourself* did not know? **(Test)**
5. Have you handed your materials to another teacher and watched the discipline survive the handoff? **(Teach)**

From three years of interviewing AI engineering candidates, most operate around rung two. Not because they are bad engineers. Because the field has not had a coherent way to teach the climb. That is the structural problem this curriculum was built to address.

If you stopped at question one, start with [TinyTorch](https://mlsysbook.ai/tinytorch/). At question two, the [MLSys·im](https://mlsysbook.ai/mlsysim) labs. At three, the [hardware kits](https://mlsysbook.ai/kits). At four, [StaffML](https://staffml.ai). At five, the [Instructor Hub](https://mlsysbook.ai/instructors/). The artifacts are arranged in the order of the climb.

I am naming this model now, three years in, because the field needs the name. Hiring managers and interviewers can ask candidates to describe the highest rung they have actually operated at, with specifics. The answer is more diagnostic than any whiteboard problem. Curriculum designers can map their syllabi against the rungs and see where they stop. Self-taught engineers can use it as an honest mirror. None of those uses require this curriculum specifically. The Builder's Stack is meant to be borrowed.

## Credit Where It Is Due

None of this is mine alone. Day to day, a small team of staff and students keeps the project alive: building labs, maintaining infrastructure, copyediting chapters, running workshops, and answering email when we cannot keep up. None of it ships without them. Around them, over **100 contributors** have made these materials sharper than would have been possible alone. Students patching typos. Instructors stress-testing pedagogy. Hardware engineers validating boards. Reviewers calling things out when something was wrong. Around all of us, institutional and industry partners keep the work open by funding, equipping, and publishing it: Harvard SEAS, MIT Press, the [tinyML](https://www.tinyml.org/) Foundation, ICTP, edX, the Edge AI Foundation, NSF, Google, Arduino, Edge Impulse, Seeed Studio, and others. The full roster is on the [partners page](https://mlsysbook.ai/community/partners). GenAI helped on parts of the codebase, the labs, and the tooling, but did not replace human review anywhere it matters.

## What's Next

The Builder's Stack as drawn here stops at the boundary of one machine. That is where the published curriculum currently has the most depth. [The Builder's Gap](https://buttondown.com/mlsysbook/archive/the-builders-gap-4483/) promised the fleet-scale essay next, and it is still next: what changes when the same six rungs have to be climbed across fleets where hardware failures are continuous and bandwidth, not compute, sets the ceiling. Volume II is the long-form version of that argument. The next newsletter is the short one.

Where are you on the Builder's Stack?

And where are our students?

Vijay

---

*P.S. If this resonated, three people to send it to:*

*A junior engineer you have ever had to ask "why didn't you check whether this would fit in memory before launching the run?" The Builder's Stack names the rung they have not yet climbed; [TinyTorch](https://mlsysbook.ai/tinytorch/) is where they start it.*

*An AI engineering candidate you watched fall apart at the deployment question. Send them to [StaffML](https://staffml.ai); it will surface what they actually do not know.*

*A colleague who teaches this material and whose slides go out of date faster than they can update them. The [Instructor Hub](https://mlsysbook.ai/instructors/) will save them a weekend.*

*P.P.S. [MLSys 2026](https://mlsys.org) runs this week, the conference where much of the systems-side work this newsletter covers gets first aired. If you are attending and would like to say hello, drop me a note. If you are not familiar with MLSys, it is worth knowing about; the proceedings are open.*

[MLSysBook](https://mlsysbook.ai) · [Subscribe](https://mlsysbook.ai/newsletter/) · [TinyTorch](https://mlsysbook.ai/tinytorch/) · [MLSys·im](https://mlsysbook.ai/mlsysim) · [StaffML](https://staffml.ai) · [GitHub](https://github.com/harvard-edge/cs249r_book) · [LinkedIn announcement](https://www.linkedin.com/in/vijay-janapa-reddi-63a6a173/)
