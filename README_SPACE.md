---
title: What Does The AI Act Say
emoji: 📘
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# What does the AI Act say?

Question answering over Regulation (EU) 2024/1689 (the AI Act) and Regulation
(EU) 2026/1744, the 2026 amendment that changed several dates of application.

Answers are built only from the text of those two documents. Every claim is
checked in code against the passage it cites before the answer is shown, and
when the text does not answer a question the system says so.

Retrieval combines BM25 keyword search with vector search, merged by reciprocal
rank fusion, followed by cross-encoder reranking. Measured at 82% retrieval hit
rate on a 60-question test set, with 100% correct refusals on out-of-scope
questions.

Source code: https://github.com/ka00689/ask-my-docs

Not legal advice.
