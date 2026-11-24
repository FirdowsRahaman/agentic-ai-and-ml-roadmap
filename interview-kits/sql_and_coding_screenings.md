# SQL, PyTorch Debugging & Live Coding Screening Kit

> **Note:** For pure NumPy ML-from-scratch implementations (`Linear/Logistic Regression`, `Stable Softmax`, `K-Means`, `Top-K Cosine`, `BM25`, `Jittered Backoff`), see **[Module 13 — Hands-On Algorithms from Scratch](../curriculum/05-production-and-practice/13_algorithms_from_scratch.md)**. This kit covers complementary **SQL Window Queries, PyTorch Bug-Hunts, Point-in-Time Joins, and Systems Data Structures** tested in Round-1 screenings.

[← GenAI & Agents Questions](./genai_and_agents_questions.md) | [Home](../README.md) | [Next: Resume Guide & Flashcards →](./resume_and_flashcards.md)

---

## Section 1: High-Frequency Analytical SQL Screenings

### Q1: Top-2 Highest-Grossing Products per Category (Handling Ties)
```sql
WITH product_revenue AS (
    SELECT
        category,
        product_id,
        SUM(quantity * unit_price) AS revenue,
        DENSE_RANK() OVER (
            PARTITION BY category
            ORDER BY SUM(quantity * unit_price) DESC
        ) AS rnk
    FROM order_items
    GROUP BY category, product_id
)
SELECT category, product_id, revenue
FROM product_revenue
WHERE rnk <= 2;
```
> **Why `DENSE_RANK()` over `ROW_NUMBER()`?** `ROW_NUMBER()` arbitrarily drops tied #2 products; `DENSE_RANK()` preserves ties without skipping rank numbers.

---

### Q2: Month-over-Month (`MoM`) Revenue Growth % Using `LAG()`
```sql
WITH monthly AS (
    SELECT
        DATE_TRUNC('month', order_date) AS month_start,
        SUM(amount_usd) AS rev
    FROM orders
    GROUP BY 1
)
SELECT
    month_start,
    rev,
    ROUND(
        100.0 * (rev - LAG(rev) OVER (ORDER BY month_start))
        / NULLIF(LAG(rev) OVER (ORDER BY month_start), 0),
        2
    ) AS mom_growth_pct
FROM monthly
ORDER BY month_start;
```

---

## Section 2: Leak-Free Point-in-Time Feature Joins (`Pandas`)

### Q3: Join User Transactions with Historical Credit Scores *Without* Future Leakage
**Problem:** Standard `pd.merge` joins all timestamps for a `user_id`, leaking next month's credit score into last month's loan application! Use `pd.merge_asof` (`direction='backward'`).

```python
import pandas as pd

loans = pd.DataFrame({
    "user_id": ["u1", "u1"],
    "apply_time": pd.to_datetime(["2026-03-10", "2026-06-15"]),
}).sort_values("apply_time")

credit_snapshots = pd.DataFrame({
    "user_id": ["u1", "u1", "u1"],
    "snapshot_time": pd.to_datetime(["2026-02-01", "2026-05-01", "2026-08-01"]),
    "fico_score": [680, 725, 790],  # 790 is in August (FUTURE relative to June!)
}).sort_values("snapshot_time")

pit_df = pd.merge_asof(
    loans,
    credit_snapshots,
    left_on="apply_time",
    right_on="snapshot_time",
    by="user_id",
    direction="backward",  # Strictly snapshot_time <= apply_time!
)
assert list(pit_df["fico_score"]) == [680, 725]  # 790 never leaks!
```

---

## Section 3: Classic PyTorch Interview "Spot the 5 Bugs" Challenge

### Q4: Find and Fix All 5 Bugs in This Broken PyTorch Validation/Training Snippet
```python
# --- BROKEN INTERVIEW SNIPPET ---
# def evaluate_and_step(model, loader, optimizer):
#     loss_fn = nn.CrossEntropyLoss()
#     for x, y in loader:
#         probs = torch.softmax(model(x), dim=-1)   # BUG 1: Double-softmax inside CrossEntropyLoss!
#         loss = loss_fn(probs, y)
#         loss.backward()                           # BUG 2: Missing optimizer.zero_grad() -> gradients accumulate!
#         optimizer.step()
#     val_out = model(x)                            # BUG 3 & 4: Missing model.eval() AND missing torch.no_grad()!
#     total_loss += loss                            # BUG 5: Accumulating tensor graph instead of loss.item() -> OOM!
```

**Corrected Production Version:**
```python
import torch
import torch.nn as nn

def train_one_epoch(model: nn.Module, loader, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()  # Expects raw unnormalized logits
    running_loss = 0.0

    for x, y in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item())  # Detach scalar float!
    return running_loss / max(len(loader), 1)
```

---

## Section 4: Systems & Infrastructure Coding (`O(1)` Embedding LRU Cache)

### Q5: Implement an `O(1)` `LRU` Embedding Cache to Cut Duplicate API Calls
```python
from collections import OrderedDict

class EmbeddingLRUCache:
    def __init__(self, capacity: int = 1024):
        self.capacity = capacity
        self._store: OrderedDict[str, list[float]] = OrderedDict()

    def get(self, text: str) -> list[float] | None:
        if text not in self._store:
            return None
        self._store.move_to_end(text)  # Mark as most recently used in O(1)
        return self._store[text]

    def put(self, text: str, vector: list[float]) -> None:
        if text in self._store:
            self._store.move_to_end(text)
        self._store[text] = vector
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)  # Evict oldest entry in O(1)

cache = EmbeddingLRUCache(capacity=2)
cache.put("a", [0.1]); cache.put("b", [0.2])
_ = cache.get("a")       # Refreshes "a"; "b" is now least recently used
cache.put("c", [0.3])    # Evicts "b"
assert cache.get("b") is None and cache.get("a") == [0.1]
```

---

## Section 5: 2026 AI-Assisted Coding Round — "Audit This AI-Generated Agent PR"

In 2026 interviews, many teams explicitly allow `Cursor`, `Copilot`, or `Claude` — grading how ruthlessly you **audit, debug, and harden AI-generated code** before shipping.

### Q6: Spot All 5 Production & Security Flaws in This AI-Generated Multi-Tenant RAG/Agent Snippet
```python
# --- BROKEN AI-GENERATED PR (DO NOT MERGE!) ---
# async def answer_tenant_ticket(tenant_id: str, user_query: str, sub_queries: list[str]):
#     # FLAW 1 (Prefix Cache Bust): Dynamic timestamp at top of system prompt busts KV prefix cache every call!
#     sys_prompt = f"Current time: {time.time()}. Static 8,000-token policy manual..."
#     # FLAW 2 (Sequential N+1 I/O): Awaiting inside a for-loop runs 5 network calls serially instead of asyncio.gather!
#     hits = []
#     for sq in sub_queries:
#         hits.extend(await vector_db.top_k(sq, k=10))
#     # FLAW 3 (Post-Filter Tenant Leak / Zero-Result Bug): Filtering tenant_id AFTER global Top-10 returns 0 docs for small tenants!
#     allowed = [h for h in hits if h.tenant_id == tenant_id]
#     # FLAW 4 (SQL Injection + Mutation Risk): Raw f-string interpolation into SQL without parameterization or SELECT-only check!
#     order = db.execute(f"SELECT * FROM orders WHERE note = '{user_query}'")
#     # FLAW 5 (Unsafe Eval / Unvalidated Output): Running eval() on LLM JSON string instead of Pydantic validation!
#     return eval(await llm(sys_prompt, allowed, order))
```

**Hardened Production Implementation:**
```python
import asyncio
from pydantic import BaseModel

class TicketAnswer(BaseModel):
    reply: str
    cited_ids: list[str]

async def mock_qdrant_filtered_search(sub_query: str, tenant_id: str) -> list[str]:
    await asyncio.sleep(0.01)
    # Pre/Payload-filtered inside Qdrant Filterable HNSW!
    return [f"{tenant_id}:{sub_query}:doc1"]

def build_cache_friendly_messages(static_policy: str, dynamic_timestamp: str, query: str) -> list[dict]:
    # Static prefix FIRST (cached!), dynamic turn LAST
    return [
        {"role": "system", "content": static_policy},
        {"role": "user", "content": f"[{dynamic_timestamp}] {query}"},
    ]

async def hardened_tenant_pipeline(tenant_id: str, sub_queries: list[str]) -> TicketAnswer:
    # Parallel I/O fan-out via asyncio.gather + tenant filter pushed into vector DB
    nested = await asyncio.gather(
        *(mock_qdrant_filtered_search(sq, tenant_id=tenant_id) for sq in sub_queries)
    )
    doc_ids = [doc for batch in nested for doc in batch]
    raw_llm_json = '{"reply": "Verified tenant policy.", "cited_ids": ["acme:refund:doc1"]}'
    return TicketAnswer.model_validate_json(raw_llm_json)

res = asyncio.run(hardened_tenant_pipeline("acme", ["refund", "sla"]))
assert res.cited_ids == ["acme:refund:doc1"]
assert build_cache_friendly_messages("STATIC_RULES", "2026-11-24", "hi")[0]["content"] == "STATIC_RULES"
```

---

[← GenAI & Agents Questions](./genai_and_agents_questions.md) | [Home](../README.md) | [Next: Resume Guide & Flashcards →](./resume_and_flashcards.md)
