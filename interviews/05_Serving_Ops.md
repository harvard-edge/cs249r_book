# Model Serving & MLOps 💼

The economics and reliability of production AI systems.

---

<details>
<summary><b>LATENCY: Why focus on P99 tail latency?</b></summary>

**Answer:** Non-linear queueing delays at high utilization.
**Explanation:** Average latency masks outliers. At high loads, request wait times grow exponentially (Erlang-C), leading to poor user experience despite high mean throughput.
</details>

<details>
<summary><b>ECONOMICS: What are the primary costs of AI infrastructure?</b></summary>

**Answer:** CapEx (Hardware) and OpEx (Energy/Cooling).
**Explanation:** 3-year Total Cost of Ownership (TCO) is often split evenly between the initial purchase price and the ongoing electricity and cooling costs.
</details>

<details>
<summary><b>DRIFT: Why do accurate models fail in production?</b></summary>

**Answer:** Statistical distribution shift.
**Explanation:** Models fail when live data differs significantly from training data. Continuous monitoring is required to detect this "silent" failure.
</details>
