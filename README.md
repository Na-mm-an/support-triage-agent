# Support Triage Agent — Final Documentation & Submission Checklist

---

## 1. README.md

```markdown
# Support Triage Agent

A multi-agent RAG system that classifies, retrieves, and answers customer
support queries with cited sources — and escalates to a human when it
isn't confident enough to answer safely.

🔗 **Live demo**: https://support-triage-agent-48kdua32jxaemnritfrbuf.streamlit.app
   (Free-tier hosting — if you see a "sleeping" screen, click "wake up,"
   it takes about 30-60 seconds to spin back up)

## Architecture

    [classify] → [retrieve] → [respond] → END
         ↓escalate    ↓escalate    ↓escalate
         └──────────→ [escalate] ←──┘

- **Classifier agent**: intent routing (27 classes) + calibrated
  out-of-scope detection via top-2 cumulative confidence
- **Retriever agent**: hybrid BM25 + dense retrieval (ChromaDB) combined
  via Reciprocal Rank Fusion (RRF)
- **Responder agent**: grounded generation (Llama-3.3-70B via Groq),
  citation-required, self-flags insufficient context
- **Orchestration**: LangGraph state machine — any agent can trigger
  escalation, which short-circuits all downstream calls

## Key Design Decisions

1. **Two-layer safety, not one.** A retrieval-confidence gate catches
   weak retrieval before generation even runs; a second LLM-level check
   catches cases where retrieval looked fine but the retrieved content
   didn't actually answer the question. Both layers independently caught
   real failures during testing (see Findings below).
2. **Domain-matched classifier.** Originally trained on Banking77 (a
   different domain than the retrieval corpus), which caused legitimate
   queries like "my payment failed" to be wrongly escalated. Retrained
   on the retrieval corpus's own intent labels after catching this
   during integration testing.
3. **Top-2 cumulative confidence, not top-1.** Fixed a case where
   legitimate queries were wrongly escalated because confidence split
   almost evenly between two valid, related intents (e.g. "track my
   refund" vs. "get a refund" for the query "where's my refund").
   Summing the top-2 class probabilities correctly recognizes "confident
   this is refund-related, just torn between two sub-flavors" as
   in-scope, while still catching genuinely unclear/irrelevant queries.

## Findings & Limitations

- Retrieval on the corpus struggles with heavily paraphrased queries
  that share little vocabulary with the corpus (e.g. "the thing I
  bought never showed up" vs. "delivery"/"shipment" terminology). This
  is caught reliably by the two-layer safety design rather than by
  perfect retrieval — the system correctly escalates rather than
  confidently answering with irrelevant retrieved content.
- Classification accuracy (99.3%) is likely inflated by the training
  corpus's templated phrasing; real, organically-phrased customer
  messages would likely see somewhat lower raw accuracy. The
  confidence-based abstention mechanism is specifically designed to
  degrade gracefully in exactly that scenario rather than fail silently.
- Abstention threshold was recalibrated using hand-written, naturally
  phrased queries (not the training corpus's own held-out data) after
  discovering the original calibration was overly optimistic due to
  being validated on the same templated distribution it was trained on.

## Evaluation

| Metric | Value |
|---|---|
| Intent classification accuracy | 99.3% |
| Intent classification macro-F1 | 99.3% |
| OOS detection rate (calibrated) | 94–98%, tunable via threshold sweep |
| Retrieval method | Hybrid BM25 + dense (Reciprocal Rank Fusion) |

### Latency

| Query type | p50 | Max | n |
|---|---|---|---|
| Answered (full pipeline: classify → retrieve → respond) | 624 ms | 1094 ms | 7 |
| Escalated (classifier-stage short-circuit) | 16 ms | 23 ms | 3 |

Escalated queries short-circuit before retrieval or LLM generation,
producing a **~40x latency reduction** and avoiding unnecessary LLM API
cost for queries the system correctly determines it shouldn't answer.
*(Measured on a small manual sample; a production deployment would log
this continuously across real traffic rather than a manual spot-check.)*

## Tech Stack

Python · LangGraph · Groq (Llama-3.3-70B) · ChromaDB · rank_bm25 ·
sentence-transformers · scikit-learn · Streamlit

## Run Locally

\`\`\`bash
pip install -r requirements.txt
export GROQ_API_KEY="your-key"
streamlit run app.py
\`\`\`
```

---

## 2. Push this README to your repo

```bash
cd /Users/namanbhatia/Desktop/FATM
# Save the README content above into README.md, then:
git add README.md
git commit -m "Add project documentation with eval numbers"
git push
```

Note: pushing to `main` will also trigger a redeploy on Streamlit Cloud (since it's connected to the repo) — this is harmless, just a normal redeploy, your app will briefly show a rebuild screen.

---

## 3. Final pre-submission checklist

- [ ] README pushed and visible on GitHub (check it renders correctly on the repo's main page)
- [ ] Live Streamlit link tested fresh — click through it yourself right before submitting, including clicking "wake up" if it's asleep
- [ ] All 4 core test queries still work on the **live, deployed** version (not just local):
  - "I want to cancel my order" → answered
  - "my payment failed, what should I do" → answered
  - "where's my refund" → answered
  - "can you recommend a good pizza place" → escalated
- [ ] Repo is **public** (check Settings → General → Danger Zone on GitHub if unsure)
- [ ] No API keys committed anywhere in the repo (check `git log -p | grep -i "gsk_"` or similar to be safe — your key should only live in Streamlit Cloud's secrets, never in code)
- [ ] Resume bullet finalized (from the earlier documentation doc)

---

## 4. Submitting to the Razorpay AI Builders form

Based on the form fields you screenshotted earlier:

**Step 1 field**: "Paste a live link or working demo"
→ Paste: `https://support-triage-agent-48kdua32jxaemnritfrbuf.streamlit.app`

**If there's a GitHub/code link field further in the form**:
→ Paste: `https://github.com/Na-mm-an/support-triage-agent`

**If there's a description/write-up field**:
→ Use your one-line pitch:
*"A multi-agent RAG system that classifies, retrieves, and answers customer support queries with cited sources, and automatically escalates to a human whenever it isn't confident enough to answer safely — instead of guessing."*
→ Optionally follow with 1-2 of your strongest eval numbers (99.3% classification accuracy, 94-98% OOS detection, ~40x latency reduction on escalated queries) to immediately signal rigor.

---

## 5. After submission (optional but valuable)

Consider adding this project to your resume's Projects section now (using the entry from the earlier documentation doc), since it's now a real, live, deployed artifact — not just local code — which is a stronger claim to make in interviews.
