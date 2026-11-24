---
module: "13B"
title: "AI Systems Engineering Patterns from Scratch"
difficulty: "advanced"
estimated_hours: 14
prerequisites:
  - "01"
learning_levels:
  - foundation
  - applied
  - production
topics:
  - lru-cache
  - token-bucket-rate-limiter
  - exponential-backoff-jitter
  - circuit-breaker
  - async-batcher
---

# Module 13B: AI Systems Engineering Patterns from Scratch

[Module 13 Index](./README.md) | [Phase 5 Index](../README.md) | [Roadmap](../../../ROADMAP.md)

---

## 1. Overview

Deploying AI systems in production requires battle-tested distributed software engineering patterns: caching expensive embedding lookups, rate limiting upstream LLM provider calls, retrying failed external APIs with exponential backoff and jitter, and opening circuit breakers during downstream outages. This practice track builds these core systems patterns from scratch in pure Python.

---

## 2. Who Should Study This?

- **AI Platform & Infrastructure Engineers** building high-concurrency gateways for model inference and tool execution.
- **Backend Developers** protecting services from cascading rate-limit errors and latency spikes.
- **Systems Architects** designing resilient, self-healing agentic runtimes.

---

## 3. Learning Objectives

By the end of this module, you will be able to:
- **Implement an O(1) Least Recently Used (LRU) Cache** using a doubly linked list and hash map.
- **Implement a Token Bucket Rate Limiter** to control requests per second (RPS) and token consumption.
- **Implement Exponential Backoff with Full Jitter** to eliminate the "thundering herd" problem on API retries.
- **Implement a Circuit Breaker state machine** (Closed, Open, Half-Open) to fail fast during external outages.

---

## 4. Prerequisites

- **Python Engineering:** Object-oriented programming, hash maps, and time operations from [Module 01](../../01-foundations/01_python_sql_and_data_stack.md).

---

## 5. Learning Levels

### Foundation
- Understand why caching and retries are critical for LLM API integration.
- Distinguish between fixed window, sliding window, and token bucket rate limiters.

### Applied Engineering
- Build a thread-safe token bucket rate limiter in Python.
- Implement exponential backoff with random jitter.

### Production Engineering
- Build a complete Circuit Breaker interceptor wrapping external tool and model calls.

---

## 6. Core Concepts & Practical Implementation

### 6.1 Token Bucket Rate Limiter
The token bucket algorithm regulates bursty API calls by refilling tokens at a constant rate $r$ up to capacity $C$:

```python
import time

class TokenBucketLimiter:
    """Thread-safe Token Bucket Rate Limiter for LLM API request throttling."""
    def __init__(self, capacity: int, fill_rate_per_sec: float):
        self.capacity = float(capacity)
        self.fill_rate = fill_rate_per_sec
        self.tokens = float(capacity)
        self.last_fill = time.time()

    def consume(self, tokens: int = 1) -> bool:
        now = time.time()
        elapsed = now - self.last_fill
        # Add newly generated tokens up to capacity
        self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
        self.last_fill = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

# Validation
limiter = TokenBucketLimiter(capacity=3, fill_rate_per_sec=10.0)
assert limiter.consume(1) is True
assert limiter.consume(1) is True
assert limiter.consume(1) is True
assert limiter.consume(1) is False  # Bucket empty!
print("Token Bucket Rate Limiter Validated Successfully!")
```

### 6.2 Exponential Backoff with Full Jitter
When an upstream API rate-limits multiple concurrent clients, retrying at fixed intervals causes all clients to strike simultaneously (the thundering herd). Adding random jitter spreads requests evenly across time:

```python
import random

def calculate_backoff_jitter(
    attempt: int, base_delay: float = 1.0, max_delay: float = 60.0
) -> float:
    """Calculates exponential backoff delay with full randomized jitter."""
    # Exponential backoff upper ceiling
    ceiling = min(max_delay, base_delay * (2 ** attempt))
    # Full jitter: pick uniform random value between 0 and ceiling
    return random.uniform(0.0, ceiling)

delays = [calculate_backoff_jitter(i, base_delay=1.0) for i in range(5)]
print("Simulated Jittered Delays (Seconds):", [round(d, 2) for d in delays])
```

---

## 7. Hands-on Exercises

### Beginner
Implement an in-memory dictionary-based Least Recently Used (LRU) Cache supporting `get(key)` and `put(key, value)` with maximum capacity $N$.

### Intermediate
Implement a Circuit Breaker with 3 states (`CLOSED`, `OPEN`, `HALF_OPEN`) that trips to `OPEN` after 5 consecutive exceptions.

### Advanced
Build an asynchronous batching queue in `asyncio` that aggregates incoming single-item embedding requests and executes them as a single multi-item vector batch when either batch size reaches 32 or timeout reaches 15ms.

---

## 8. Completion Checklist

- [ ] Can implement a Token Bucket Rate Limiter from scratch.
- [ ] Understand why full jitter is mandatory for distributed retries.
- [ ] Can implement an LRU cache with $O(1)$ operations.
- [ ] Understand Circuit Breaker failure isolation.
