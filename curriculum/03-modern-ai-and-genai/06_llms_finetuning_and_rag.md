# Module 06: Generative AI, LLMs, Fine-Tuning (`LoRA`/`DPO`), `Qdrant` & Production RAG

> **Estimated Study Time:** 10–12 Hours | **Prerequisites:** [Module 05](./05_nlp_cv_and_multimodal.md) | **Focus:** LLM Post-Training, Structured Outputs, `Qdrant` Vector Indexing, Hybrid RAG & `vLLM` Serving

[← Previous: Module 05 — NLP, CV & Multimodal](./05_nlp_cv_and_multimodal.md) | [Home](../../README.md) | [Next: Module 07 — AI Agents Fundamentals →](../04-agentic-ai/07_ai_agents_fundamentals.md)

---

## Mental Model: How Base Models Become Grounded Enterprise Systems

A raw pre-trained LLM is merely a next-token internet autocomplete engine. Turning it into a reliable enterprise system requires **Post-Training Alignment (`SFT` + `DPO`/`GRPO`)**, **Structured Schema Constraints**, **Filterable Hybrid Vector Search (`Qdrant` + Reranking)**, and **Memory-Efficient Inference (`vLLM`)**.

```mermaid
flowchart LR
    PRE["1. Pre-Training — Trillions of Web Tokens"] --> POST["2. Post-Training — SFT + DPO / GRPO Reasoning"]
    POST --> ADAPT{"3. Domain Adaptation"}
    ADAPT -->|"Dynamic / Private Facts"| RAG["Qdrant Hybrid RAG + Cross-Encoder"]
    ADAPT -->|"Style / Niche Format"| LORA["PEFT: LoRA / QLoRA"]
    RAG & LORA --> SERVE["4. Production Serving — vLLM PagedAttention + JSON Schema"]

    classDef base fill:#E0F2FE,stroke:#0284C7,stroke-width:1.5px,color:#0369A1;
    classDef mid fill:#EDE9FE,stroke:#7C3AED,stroke-width:1.5px,color:#4C1D95;
    classDef prod fill:#DCFCE7,stroke:#16A34A,stroke-width:2px,color:#15803D;

    class PRE,POST base;
    class ADAPT,RAG,LORA mid;
    class SERVE prod;
```

---

## Tier 1: Core Foundations (LLM Mechanics, Structured Outputs, Chunking & Vector Stores)

### 1. LLM Landscape, VRAM Sizing & Quantization

#### 1.1 Discriminative vs. Generative Paradigms & GPU Memory Sizing
- **Discriminative Models** learn conditional boundaries $P(y \mid x)$ to classify or score inputs; **Generative Models** model autoregressive or diffusion distributions $P(x_t \mid x_{<t})$ to synthesize text, code, images, or audio.
- **Modern Model Ecosystem:**
  - *Frontier API Models:* **Google Gemini 2.5 Pro / Flash**, **OpenAI GPT-4.1 / o3 / o4-mini**, **Anthropic Claude 4 Opus / Sonnet**.
  - *Open-Weight Models (Self-Hostable):* **Meta Llama 3.3 / Llama 4**, **Qwen 3 / Qwen 2.5**, **DeepSeek-V3 / R1**, **Mistral / Gemma 3**.
- **GPU VRAM Capacity Formula:**
  - `FP32` = 4 bytes/param | `FP16` / `BF16` = **2 bytes/param** | `INT8` / `FP8` = **1 byte/param** | `INT4` (`AWQ` / `GGUF`) = **0.5 bytes/param** (+ ~20% overhead for KV cache & activations).
  - *Sizing Example:* A **70B** model in `BF16` requires $70 \times 2 \times 1.2 \approx \mathbf{168\text{ GB VRAM}}$ (2× 80GB GPUs), whereas **4-bit quantization** shrinks it to $70 \times 0.5 \times 1.2 \approx \mathbf{42\text{ GB VRAM}}$.

---

### 2. Prompting, Structured Outputs & Hallucination Grounding

#### 2.1 Constrained Decoding & Guaranteed JSON Schemas
In production software, asking an LLM *"Please reply in valid JSON"* via plain text still fails 2–5% of the time with markdown wrappers or missing keys. Modern APIs and open-source engines (`Outlines`, `vLLM`) enforce **Grammar-Constrained Decoding** — masking invalid vocabulary tokens at every decoding step so outputs are **mathematically guaranteed** to conform to your `Pydantic` schema:

```python
from pydantic import BaseModel, Field

class InvoiceExtraction(BaseModel):
    vendor_name: str
    invoice_total_usd: float = Field(ge=0.0)
    line_items: list[str]
    requires_manual_audit: bool

# Passed directly to Gemini response_schema / OpenAI response_format / vLLM guided_json
json_schema = InvoiceExtraction.model_json_schema()
```

---

#### 2.2 Why Hallucinations Happen — and What RAG Actually Guarantees
LLMs optimize next-token likelihood rather than factual truth, smoothly extrapolating plausible-sounding fabrications when knowledge is missing or ambiguous.
- **Engineering Reality:** **RAG grounds the model in retrieved context and drastically reduces hallucinations, but NEVER 100% eliminates them.** If retrieval pulls noisy chunks, if key facts sit buried in the middle of a long context window, or if system prompts lack strict abstention rules, the model can still misinterpret context or fabricate citations. Always pair RAG with explicit abstention prompts (*"If unsupported by context, state insufficient information"*) and automated faithfulness checks.

---

### 3. Document Chunking & RAG vs. Fine-Tuning

#### 3.1 Document Chunking Strategies
Poor chunking destroys retrieval quality before embeddings even run:
1. **Fixed-Token with Overlap (e.g., `512` tokens, `64` overlap):** Simple baseline; risks slicing tables or clauses mid-sentence.
2. **Recursive Character / Markdown-Header Splitting:** Splits hierarchically by headings (`#`, `##`), paragraphs (`\n\n`), then sentences.
3. **Layout-Aware Parsing (`Docling`, `Marker`, AWS Textract):** Preserves multi-column PDFs, financial tables, and figures intact as structured Markdown/JSON.
4. **Contextual Retrieval:** Prepends a 1-sentence document-level summary (e.g., *"From Acme Corp Q3 2026 10-Q filing:"*) to every chunk prior to embedding and sparse indexing, cutting retrieval misses by ~49%.

---

#### 3.2 Choosing Between RAG and Fine-Tuning (`LoRA`)

| Dimension | Retrieval-Augmented Generation (RAG) | Fine-Tuning (`LoRA` / `SFT`) |
|---|---|---|
| **Primary Purpose** | Dynamic facts, private knowledge bases, source citations, per-user ACLs | Teaching domain tone, compact output formats, specialized SQL/DSL syntax, or distilling a large teacher into a fast 8B student |
| **Data Freshness** | Instant (upsert/delete vectors in milliseconds) | Static until the next training job |
| **Hallucination Control** | High (verifiable chunk citations) | Low on unseen facts |

---

## Tier 2: Production Architecture & Scale (Post-Training, `Qdrant`, Hybrid RAG & `vLLM`)

### 4. Post-Training (`SFT`, `LoRA`/`QLoRA`, `DPO`/`GRPO`) & Test-Time Compute

#### 4.1 Parameter-Efficient Fine-Tuning & Alignment Mechanics
1. **`LoRA` (Low-Rank Adaptation):** Freezes pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ and injects trainable low-rank adapters $\Delta W = B A$ where $A \in \mathbb{R}^{r \times k}, B \in \mathbb{R}^{d \times r}$ with rank $r \ll d$ (e.g., $r = 16$). Cuts trainable parameters by **99%+** and prevents catastrophic forgetting. **`QLoRA`** quantizes frozen $W_0$ to 4-bit `NF4` while training `BF16` LoRA adapters on a single 24 GB GPU.
2. **Preference Alignment (`RLHF` vs. `DPO`):** **Direct Preference Optimization (`DPO`)** optimizes the policy directly on `(prompt, chosen, rejected)` preference pairs via a closed-form loss without maintaining a separate `PPO` reward model.
3. **Reasoning Models (`GRPO` & Test-Time Compute):** Models like `o3`, `Gemini 2.5 Thinking`, and `DeepSeek-R1` train via **Reinforcement Learning with Verifiable Rewards (`GRPO`)** against code/math verifiers, spending dynamic **test-time thinking tokens** to backtrack and self-correct.

---

### 5. Production Vector Databases & `Qdrant` Architecture

#### 5.1 Selecting the Right Vector Search Engine

| Engine | Architecture | Sweet Spot & Trade-Offs |
|---|---|---|
| **`Qdrant`** | Dedicated Rust engine (`HNSW` + Sparse + Payload Index) | **Top choice for Production RAG & Agents:** local `:memory:`/disk mode up to distributed cloud, **Filterable HNSW**, native Dense+Sparse RRF, and 32× Binary Quantization |
| **`pgvector`** | PostgreSQL extension (`HNSW` / `StreamingDiskANN`) | Great when vectors live alongside relational ACID tables at moderate scale (`<10M–50M` vectors) |
| **`FAISS`** | C++/GPU in-memory ANN library (Meta) | Ultra-fast raw ANN math inside custom Python/C++ services; **not** a database (no CRUD API, WAL durability, or metadata filtering engine out of the box) |
| **`OpenSearch`** | Distributed Lucene + `k-NN` cluster | Heavy enterprise log/search stacks already standardized on AWS OpenSearch |

---

#### 5.2 Why `Qdrant` Excels: Filterable HNSW, Native Hybrid RRF & Quantization
1. **Solving the Metadata Filtering Trap (`Filterable HNSW`):**
   - *Naive Post-Filtering* runs ANN `Top-K` first and drops unauthorized/mismatched tenant docs afterward — often returning **0 results** when a tenant owns `<1%` of the index!
   - *Naive Pre-Filtering* masks out nodes in an `HNSW` graph, severing graph edges and ruining recall.
   - **`Qdrant` Filterable HNSW** builds additional payload-indexed subgraph links directly inside `HNSW` (and switches automatically to exact/indexed scan at tiny filter cardinality), guaranteeing `<10ms` filtered recall for multi-tenant ACLs (`tenant_id`, `role`, `date_range`).
2. **Binary & Scalar Quantization:** Compresses `FP32` vectors to `INT8` (**4× less RAM**) or 1-bit **Binary Quantization** (**32× less RAM** on high-dimensional embeddings $\ge 1024$d), oversampling candidates in RAM and rescoring the top hits from disk with ~99% original accuracy.
3. **Single-Call Hybrid Search (`Prefetch` + Server-Side `RRF`):**

```python
from qdrant_client import QdrantClient, models

# Runs 100% in-process locally (swap url="https://..." for Docker / Qdrant Cloud!)
client = QdrantClient(":memory:")

client.create_collection(
    collection_name="enterprise_kb",
    vectors_config={"dense": models.VectorParams(size=4, distance=models.Distance.COSINE)},
    sparse_vectors_config={"bm25": models.SparseVectorParams()},
)

client.upsert(
    collection_name="enterprise_kb",
    points=[
        models.PointStruct(
            id=1,
            vector={
                "dense": [0.9, 0.1, 0.2, 0.0],
                "bm25": models.SparseVector(indices=[101, 409], values=[1.8, 3.2]),
            },
            payload={"tenant_id": "acme", "doc_id": "POL-409", "text": "OAuth token rotation policy."},
        )
    ],
)

# Native Hybrid Search (Dense + Sparse) with ACL Payload Filter & Server-Side RRF Fusion:
results = client.query_points(
    collection_name="enterprise_kb",
    prefetch=[
        models.Prefetch(query=[0.88, 0.12, 0.18, 0.0], using="dense", limit=20),
        models.Prefetch(
            query=models.SparseVector(indices=[409], values=[1.0]),
            using="bm25",
            limit=20,
        ),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    query_filter=models.Filter(
        must=[models.FieldCondition(key="tenant_id", match=models.MatchValue(value="acme"))]
    ),
    limit=5,
)
```

---

### 6. End-to-End Advanced RAG Pipeline & `RAGAS` Evaluation

#### 6.1 Hybrid Retrieval, Cross-Encoder Reranking & The `RAGAS` Triad
Dense embeddings capture semantic gist but blur exact part numbers, error codes (`ERR_409_AUTH`), and product SKUs. Combine `Qdrant` Hybrid `RRF` (`Top 50`) with a **Cross-Encoder Reranker (`BGE-Reranker` / `Cohere Rerank` -> `Top 5`)** and evaluate continuously using the **`RAGAS` Triad**:

```mermaid
flowchart TD
    Q["User Query + ACL Token"] --> QT["1. Query Rewrite / Multi-Query"]
    QT --> QD["2. Qdrant Hybrid Prefetch (BM25 Sparse + Dense HNSW + ACL Filter)"]
    QD --> RRF["3. Server-Side Reciprocal Rank Fusion (Top 50)"]
    RRF --> CE["4. Cross-Encoder Reranker (BGE / Cohere -> Top 5)"]
    CE --> GEN["5. Grounded LLM Generation + Verified Citations"]
    GEN --> EVAL["6. RAGAS Triad: Context Precision, Faithfulness, Answer Relevance"]

    classDef inp fill:#E0F2FE,stroke:#0284C7,stroke-width:2px,color:#0369A1;
    classDef ret fill:#FEF3C7,stroke:#D97706,stroke-width:1.5px,color:#92400E;
    classDef rank fill:#EDE9FE,stroke:#7C3AED,stroke-width:2px,color:#4C1D95;
    classDef out fill:#DCFCE7,stroke:#16A34A,stroke-width:2px,color:#15803D;

    class Q,QT inp;
    class QD,RRF ret;
    class CE,GEN rank;
    class EVAL out;
```

---

### 7. LLM Inference Optimization: `TTFT`, `TPOT`, `KV Cache` & `vLLM`

#### 7.1 Prefill vs. Decode Bottlenecks & `PagedAttention`
- **`TTFT` (Time-To-First-Token — Prefill):** Processes all prompt tokens in parallel; **compute-bound**. Accelerated via **Prefix / Prompt Caching**.
- **`TPOT` (Time-Per-Output-Token — Decode):** Emits 1 token per step; **memory-bandwidth bound** (streams weights + KV cache from HBM every token). Accelerated via **Quantization (`FP8`/`AWQ`)** and **Speculative Decoding**.
- **`PagedAttention` (`vLLM`) & Continuous Batching:** Stores `KV Cache` in non-contiguous fixed-size memory pages (eliminating 60%+ contiguous VRAM waste) and swaps finished requests at token granularity, increasing GPU concurrency **2x–4x**.

---

## Mini Hands-On Lab (15-Minute Exercise)

Implement **Reciprocal Rank Fusion (`RRF`)** — the exact rank-merging algorithm running inside `Qdrant`'s `Fusion.RRF` — to combine lexical `BM25` hits and dense vector hits without brittle score normalization:

```python
def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60, top_n: int = 3
) -> list[tuple[str, float]]:
    """RRF(d) = sum_{list} 1 / (k + rank_in_list(d)), 1-indexed."""
    rrf_scores: dict[str, float] = {}
    for ranking in ranked_lists:
        for rank_idx, doc_id in enumerate(ranking, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank_idx)
    return sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:top_n]

bm25_hits = ["doc_sku_99", "doc_refund", "doc_faq"]
dense_hits = ["doc_refund", "doc_policy", "doc_sku_99"]

fused = reciprocal_rank_fusion([bm25_hits, dense_hits], k=60, top_n=2)
top_ids = {doc for doc, _ in fused}
assert top_ids == {"doc_sku_99", "doc_refund"}
print("Lab 06 Passed! Top Hybrid RRF Results:", fused)
```

---

## Key Engineering Takeaways
- Size GPU memory fast: `BF16` $\approx 2$ bytes/param, `INT4` $\approx 0.5$ bytes/param (+ ~20% `KV Cache` overhead).
- Enforce **Structured Outputs (grammar-constrained JSON decoding)** in production pipelines rather than regex parsing.
- Use **`Qdrant` Filterable HNSW** (with payload ACL filters, native Dense+Sparse `RRF`, and Scalar/Binary Quantization) + **Cross-Encoder Reranking**, gated by the **`RAGAS` Triad**.
- Choose **RAG** for dynamic/cited facts, **`LoRA`/`QLoRA`** for domain formatting or distillation, and **`vLLM`** to optimize compute-bound **`TTFT`** and memory-bound **`TPOT`**.

---

[← Previous: Module 05 — NLP, CV & Multimodal](./05_nlp_cv_and_multimodal.md) | [Home](../../README.md) | [Next: Module 07 — AI Agents Fundamentals →](../04-agentic-ai/07_ai_agents_fundamentals.md)
