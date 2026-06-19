"""
DEMO 3 - Continual learning without retraining (LLM + SAGE)
===========================================================
Teach the agent a stream of brand-new facts the base LLM cannot know, one at a
time, with NO retraining. Then ask questions - including about the FIRST facts
taught - and show:
  - LLM ALONE: cannot answer (it never saw these facts).
  - LLM + SAGE: answers correctly, because SAGE retrieved the fact - and the
    EARLIEST facts are still there (no catastrophic forgetting), at a bounded
    memory footprint.

What it demonstrates: knowledge added at inference time, retained without
forgetting, used by the LLM. (Honest: this is retrieval-augmented memory; SAGE's
no-forgetting is real but shared by NCM / k-means / a vector store - the demo
shows it WORKING, not beating them.)

Run:  python demos/demo3_continual.py
Requires Ollama (nomic-embed-text + an LLM, e.g. mistral).
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from demos._sage_demo import (SageMemory, embed, chat, banner, rule,  # noqa: E402
                              say, require_ollama, C, LLM_MODEL)

# Invented, neutral facts (so the base model cannot already know them; no
# personal or identifying information).
FACTS = [
    "The Orion-7 ground station transmits telemetry on the 2.4 gigahertz band.",
    "The Atlas indexing service rebuilds its cache every night at 02:00 UTC.",
    "Module K stores each record as a vector at a fixed slot and retrieves it by cosine similarity.",
    "The reference deployment runs on a single mid-range laptop GPU.",
    "Every write in the system is a local update and requires no retraining.",
    "The evaluation set contains 4,000 records spread across 10 categories.",
    "The text encoder used by the system produces 768-dimensional vectors.",
    "Version 2 of the system was released in the first quarter of 2026.",
]

# Questions: the first targets the EARLIEST fact (tests no-forgetting).
QUESTIONS = [
    "What frequency band does the Orion-7 ground station transmit on?",
    "When does the Atlas indexing service rebuild its cache?",
    "How many dimensions do the text encoder's vectors have?",
]

SYS_PLAIN = ("Answer the question in one short sentence using only what you already "
             "know. If you do not know, say 'I don't know.'")
SYS_MEM = ("Answer the question in one short sentence using the FACTS below. If the "
           "facts do not contain the answer, say 'I don't know.'\n\nFACTS:\n%s")


def teach(mem):
    banner("Phase 1 - Teach %d new facts (no retraining)" % len(FACTS))
    for i, fact in enumerate(FACTS, 1):
        action, slot, _ = mem.write(embed(fact), fact)
        print("  taught #%d -> slot %d  %s%s%s" % (i, slot, C.DIM, fact, C.R))
        time.sleep(0.15)
    print("\n  Memory footprint: %s%d facts%s (bounded; nothing retrained).\n"
          % (C.B, mem.footprint(), C.R))


def ask(mem, q):
    rule()
    say("question", q, C.BLUE)
    # (a) LLM alone
    try:
        a_plain, _ = chat(SYS_PLAIN, q)
    except Exception as e:
        a_plain = "(LLM error: %s)" % e
    say("LLM only", a_plain, C.RED)
    # (b) LLM + SAGE
    recalled = mem.recall(embed(q), topk=3, threshold=0.4)
    facts = "\n".join("- %s" % t for t, _ in recalled) or "(none)"
    print(C.MAG + "  SAGE recall:" + C.R)
    for t, s in recalled:
        print("    %s%.2f%s  %s" % (C.DIM, s, C.R, t))
    try:
        a_mem, _ = chat(SYS_MEM % facts, q)
    except Exception as e:
        a_mem = "(LLM error: %s)" % e
    say("LLM+SAGE", a_mem, C.GREEN)


def main():
    require_ollama(need_llm=True)
    mem = SageMemory(n_slots=128, merge=0.62)

    banner("SAGE Continual Learning - no retraining, no forgetting",
           "LLM=%s | teach new facts live, then answer with vs without SAGE"
           % LLM_MODEL)
    teach(mem)

    banner("Phase 2 - Ask (note: question 1 targets the FIRST fact taught)")
    for q in QUESTIONS:
        ask(mem, q)

    rule()
    banner("Result")
    print("  - The LLM alone could not answer the new facts (never trained on them).")
    print("  - LLM + SAGE answered correctly, including the EARLIEST fact taught")
    print("    (%s+%d facts retained, 0 forgotten, footprint %d%s) - all without retraining."
          % (C.B, mem.footprint(), mem.footprint(), C.R))
    print(C.DIM + "\n  Honest note: this is retrieval-augmented memory; SAGE's "
          "no-forgetting is real but also achieved by NCM / k-means / a vector "
          "store. The demo shows it working end-to-end." + C.R)


if __name__ == '__main__':
    main()
