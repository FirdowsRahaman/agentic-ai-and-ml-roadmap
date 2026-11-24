# Classical ML, RecSys & MLOps Interview Questions

[← System Design Framework](./ml_system_design_framework.md) | [Home](../README.md) | [Next: GenAI & Agents Questions →](./genai_and_agents_questions.md)

---

## 1. Evaluation & Bias–Variance

**Q1: Explain Precision vs. Recall, and why ROC-AUC can mislead you on rare-event datasets.**
*Answer:*
- **Precision** ($\frac{TP}{TP + FP}$) penalizes False Positives (spam filter, legal auto-actions).
- **Recall** ($\frac{TP}{TP + FN}$) penalizes False Negatives (cancer screening, fraud).
- On a `0.1%` positive dataset (`1,000` negatives per `1` positive), $TN$ is so huge that 500 false alarms barely move $\text{FPR} = \frac{FP}{FP + TN}$, keeping **ROC-AUC > 0.97** while precision is awful. Always report **PR-AUC** and **Recall at fixed Precision/FPR** for imbalanced problems.

**Q2: Walk through the Bias–Variance Tradeoff and compare Bagging vs. Boosting.**
*Answer:*
- **High Bias (Underfitting):** Model is too constrained (both train & validation error high). **Boosting (`XGBoost`, `LightGBM`)** reduces bias by training shallow trees sequentially on the residual gradients of prior trees.
- **High Variance (Overfitting):** Model memorizes training noise (low train error, high validation error). **Bagging (`Random Forest`)** reduces variance by averaging deep independent trees trained on bootstrap samples + random feature subsets.

---

## 2. Preprocessing, Scaling & Leakage

**Q3: Why does `LogisticRegression` with L1/L2 regularization fail if you skip `StandardScaler`, while `XGBoost` is unaffected?**
*Answer:*
L1/L2 penalties shrink raw weight magnitudes ($\lambda \|\mathbf{w}\|_2^2$). If feature $x_1$ is measured in cents (`0–10,000,000`) and $x_2$ in years (`0–50`), $w_1$ is naturally tiny while $w_2$ is large — regularization unfairly crushes $w_2$! Decision trees only compare rank order (`x <= threshold`), making them invariant to monotonic scaling.

**Q4: Name three subtle forms of Data Leakage beyond "fitting `StandardScaler` before `train_test_split`".**
*Answer:*
1. **Group Leakage:** Multiple medical scans from the same patient appear in both train and test (`Fix:` `GroupKFold` by `patient_id`).
2. **Look-Ahead Temporal Leakage:** Using monthly aggregated customer features to predict an event that happened on the 5th of that same month (`Fix:` point-in-time joins).
3. **Out-of-Fold Target Encoding Leakage:** Computing category target means across the full training fold including the row itself (`Fix:` out-of-fold smoothed `TargetEncoder`).

---

## 3. Recommendation Systems & Production MLOps

**Q5: How do you design a Recommendation System for 20M items under a `40ms` latency SLA, including Cold Start?**
*Answer:*
1. **Stage 1 Retrieval (`<10ms`):** Two-Tower user/item encoder + `FAISS HNSW` ANN search retrieving `Top 300` candidates (optimizing `Recall@300`).
2. **Stage 2 Ranking (`~15ms`):** `LightGBM LambdaMART` or `DCNv2` scoring cross-features (user×category historical CTR, recency, price sensitivity) optimizing **`NDCG@10`**.
3. **Stage 3 Re-Ranking:** `MMR` diversity + freshness + business rules.
4. **Cold Start:** Content/multimodal embeddings (`CLIP`/text title) for zero-interaction new items + contextual bandit exploration (`5%` $\epsilon$-greedy / Thompson Sampling).

**Q6: Live F1 dropped 12% this month, yet feature `PSI` across every input column is `< 0.04`. What happened?**
*Answer:*
**Concept Drift** ($P(Y \mid X)$ shifted while $P(X)$ stayed constant) — e.g., fraudsters adapted behavior within normal transaction ranges, or a macroeconomic shift altered buyer conversion. Fix by collecting fresh ground-truth labels, retraining on a recent sliding window with sample decay, and validating via Shadow -> Canary -> A/B rollout.

**Q7: What is Training-Serving Skew, and how do you eliminate it?**
*Answer:*
Discrepancies between batch feature code (SQL/Pandas during training) and real-time microservice feature code (Python/Go at inference). Eliminate it by centralizing definitions in a **Feature Store (`Feast`)** serving point-in-time historical Parquet/Warehouse joins offline and sub-5ms Redis lookups online.

---

[← System Design Framework](./ml_system_design_framework.md) | [Home](../README.md) | [Next: GenAI & Agents Questions →](./genai_and_agents_questions.md)
