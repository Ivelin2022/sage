"""
DEMO 1 - Personal memory assistant (LLM + SAGE)
================================================
A terminal chatbot whose long-term memory is SAGE. It:
  - remembers facts you tell it, ACROSS sessions (memory saved to disk),
  - retrieves the relevant memories and feeds them to the LLM as context,
  - keeps answering when the LLM is switched OFF (graceful degradation:
    it returns the nearest stored memory directly - no model needed).

What it demonstrates: a working, gradient-free, continually-updated memory layer
for an LLM agent. (Honest: SAGE here = a bounded vector store + merge; the point
is that it WORKS and is LLM-independent, not that it beats a vector DB.)

Run:  python demos/demo1_assistant.py
Commands:  /off  /on  /mem  /save  /forget  /help  /quit

Requires Ollama (nomic-embed-text + an LLM, e.g. mistral).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from demos._sage_demo import (SageMemory, embed, chat, badge, banner, rule,  # noqa: E402
                              say, require_ollama, C, LLM_MODEL)

MEM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                        'assistant_mem.npz')

SYSTEM = (
    "You are a helpful assistant with a persistent memory. Below is MEMORY recalled "
    "for this turn - facts the user told you earlier (possibly in past sessions). "
    "Use it to answer naturally and specifically. If the user states a new fact "
    "about themselves, acknowledge it briefly. Keep replies short.")


def build_system(recalled):
    if not recalled:
        return SYSTEM + "\n\nMEMORY: (empty so far)"
    lines = "\n".join("- %s" % t for t, _ in recalled)
    return SYSTEM + "\n\nMEMORY:\n" + lines


def show_recall(recalled):
    if not recalled:
        print(C.DIM + "  (memory empty - nothing recalled yet)" + C.R)
        return
    print(C.MAG + "  SAGE recall:" + C.R)
    for t, s in recalled:
        print("    %s%.2f%s  %s" % (C.DIM, s, C.R, t))


def main():
    os.makedirs(os.path.dirname(MEM_PATH), exist_ok=True)
    require_ollama(need_llm=True)
    mem = SageMemory.load(MEM_PATH, n_slots=256)

    banner("SAGE Personal Memory Assistant",
           "LLM=%s | persistent SAGE memory | type /help for commands" % LLM_MODEL)
    print("Loaded memory: %s%d%s items remembered from previous sessions.\n"
          % (C.B, mem.footprint(), C.R))
    if mem.footprint() == 0:
        print(C.DIM + "  Tip: tell it facts (e.g. 'the deployment runs every "
              "Tuesday at 09:00'), then ask about them later - even after "
              "restarting." + C.R + "\n")

    llm_on = True
    history = []
    while True:
        try:
            user = input(C.BLUE + "you> " + C.R).strip()
        except (EOFError, KeyboardInterrupt):
            user = "/quit"
        if not user:
            continue

        if user == "/quit":
            mem.save(MEM_PATH)
            print(C.GREEN + "Saved %d memories to disk. Bye." % mem.footprint() + C.R)
            return
        if user == "/help":
            print("  /off  switch LLM off (graceful degradation: answer from memory)")
            print("  /on   switch LLM back on")
            print("  /mem  show what SAGE is holding")
            print("  /save save memory now    /forget wipe memory    /quit save+exit")
            continue
        if user == "/off":
            llm_on = False; print(badge(llm_on) + " LLM disabled - now answering "
                                  "from SAGE memory only.\n"); continue
        if user == "/on":
            llm_on = True; print(badge(llm_on) + " LLM re-enabled.\n"); continue
        if user == "/mem":
            rule()
            print("SAGE memory: %d slots used. Strongest first:" % mem.footprint())
            for t, st in mem.dump()[:12]:
                print("   %s[%.1f]%s %s" % (C.DIM, st, C.R, t))
            rule(); continue
        if user == "/save":
            mem.save(MEM_PATH); print(C.GREEN + "Saved." + C.R); continue
        if user == "/forget":
            mem = SageMemory(n_slots=256); print(C.YELLOW + "Memory wiped." + C.R)
            continue

        # ---- normal turn ----
        k = embed(user)
        recalled = mem.recall(k, topk=3, threshold=0.4)
        print(badge(llm_on))
        show_recall(recalled)

        if llm_on:
            try:
                reply, secs = chat(build_system(recalled), user, history=history[-4:])
            except Exception as e:
                print(C.RED + "  LLM call failed (%s) - falling back to memory." % e
                      + C.R)
                llm_on_local = False
                reply = None
            else:
                say("assistant", reply, C.GREEN)
                print(C.DIM + "  (%.1fs via %s)" % (secs, LLM_MODEL) + C.R)
                history += [{"role": "user", "content": user},
                            {"role": "assistant", "content": reply}]
        if not llm_on:
            # graceful degradation: return the nearest stored memory directly
            if recalled:
                say("assistant", "[from memory] " + recalled[0][0], C.YELLOW)
            else:
                say("assistant", "[from memory] I have nothing relevant stored yet.",
                    C.YELLOW)

        # ---- write the user's statement into SAGE (so it is remembered) ----
        action, slot, sim = mem.write(k, user)
        tag = {"consolidate": "updated existing memory",
               "new": "stored new memory", "evict": "stored (evicted weakest)"}[action]
        print(C.DIM + "  SAGE: %s -> slot %d (%d/%d used)\n"
              % (tag, slot, mem.footprint(), mem.B) + C.R)


if __name__ == '__main__':
    main()
