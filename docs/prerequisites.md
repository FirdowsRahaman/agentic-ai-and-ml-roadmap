# Prerequisites & Self-Assessment

[← Back to Documentation](./index.md) | [Getting Started](./getting-started.md) | [Roadmap](../ROADMAP.md)

---

## 1. Self-Assessment Checklist

Before starting the curriculum, review your baseline proficiency. If you lack experience in a specific area, refer to the recommended prerequisite resources below.

### Software Engineering Baseline
- [ ] Comfortable writing modular Python functions, type hints, and exception handling.
- [ ] Familiarity with Git basics: `git clone`, `branch`, `commit`, and `pull request`.
- [ ] Basic SQL querying: `SELECT`, `WHERE`, `GROUP BY`, and `JOIN`.
- [ ] Understanding of virtual environments (`venv`, `conda`) and package managers (`pip`).

### Mathematical & Analytical Baseline
- [ ] Linear Algebra: Matrix multiplication, transpose, dot products, vector norms.
- [ ] Calculus: Derivatives, partial derivatives, and the chain rule of differentiation.
- [ ] Probability: Mean, variance, standard deviation, conditional probability $P(A|B)$, and Bayes' Theorem.

---

## 2. Prerequisite Bridging Resources

If you need a refresher before diving into Module 01, we recommend these focused, high-signal materials:

| Domain | Recommended Resource | Time Investment |
|---|---|---|
| **Python Fundamentals** | [Python Official Tutorial](https://docs.python.org/3/tutorial/) | 4–6 Hours |
| **Linear Algebra** | [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) | 3 Hours |
| **Calculus & Backprop** | [Andrej Karpathy: Building Micrograd](https://karpathy.ai/zero-to-hero.html) | 2.5 Hours |
| **Modern SQL** | [Mode SQL Tutorial](https://mode.com/sql-tutorial/) | 3 Hours |

---

## 3. Prerequisite Dependency Graph

The following conceptual dependency map illustrates how foundational engineering and mathematical primitives build toward production-grade systems:

<div align="center">
  <img src="../assets/diagrams/prerequisite_dependency_graph.svg" alt="Prerequisite Dependency Graph" width="850" />
</div>

<br/>

```mermaid
flowchart TD
    A["Python + Mathematics + Git"]
    B1["Classical ML"]
    B2["Data Engineering"]
    C["Deep Learning + Evaluation"]
    D["Transformers + Generative AI"]
    E1["RAG Systems"]
    E2["Agentic Systems"]
    F["Production AI Engineering"]

    A --> B1 & B2
    B1 & B2 --> C
    C --> D
    D --> E1 & E2
    E1 & E2 --> F

    classDef base fill:#0B1325,stroke:#1E3A5F,stroke-width:1.5px,color:#FFFFFF,font-weight:600;
    classDef branch fill:#070A0F,stroke:#1E293B,stroke-width:1px,color:#F1F5F9,font-size:15px,font-weight:600;
    classDef prod fill:#082618,stroke:#1B4D32,stroke-width:1.5px,color:#FFFFFF,font-weight:600;

    class A,C,D base;
    class B1,B2,E1,E2 branch;
    class F prod;
```

---

## 4. Prerequisite Mapping by Tier

- **Tier 1 (Foundations):** Requires only high-school math and basic Python scripting.
- **Tier 2 (Classical ML):** Builds upon Tier 1 NumPy broadcasting and hypothesis testing.
- **Tier 3 (Deep Learning):** Requires vector calculus (chain rule) and matrix tensor operations.
- **Tier 4 (Generative AI):** Requires understanding of attention mechanisms and high-dimensional vector spaces.
- **Tier 5 (Autonomous Agents):** Requires stateful software architecture, asynchronous programming, and JSON schemas.
- **Tier 6 (Production AI):** Requires Docker containerization, REST API fundamentals (HTTP methods, status codes), and Linux command-line literacy.
