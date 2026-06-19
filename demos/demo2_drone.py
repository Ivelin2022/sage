"""
DEMO 2 - Drone agent with graceful degradation (LLM + SAGE)
===========================================================
A drone autopilot runs a 12-step mission. For steps 1-8 the LLM is online
(hybrid: SAGE recalls a relevant past decision, the LLM decides). At step 9 the
ground-station CONNECTION IS LOST - the LLM goes offline, and the agent keeps
flying by reading decisions straight from SAGE memory. Zero "give up" defaults,
and the offline decisions come back in milliseconds instead of seconds.

What it demonstrates: an agent that DEGRADES GRACEFULLY when its LLM is gone,
using a local gradient-free memory fallback. (Honest: any local vector store gives
the same zero-default fallback; the contribution shown is the working integrated
behaviour, not uniqueness to SAGE.)

Run:  python demos/demo2_drone.py
Requires Ollama (nomic-embed-text + an LLM, e.g. mistral).
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from demos._sage_demo import (SageMemory, embed, chat, badge, banner, rule,  # noqa: E402
                              require_ollama, C, LLM_MODEL)

# Minimum viable prior knowledge: observation -> action, pre-seeded before flight.
SEED = [
    ("all systems nominal, continue survey", "CONTINUE survey pattern, hold altitude"),
    ("tree or obstacle on the flight path ahead", "ASCEND 5m and bank right to reroute"),
    ("power lines crossing the flight path", "CLIMB above the lines, then resume course"),
    ("person or people detected below the drone", "HOLD position, increase altitude for safety"),
    ("battery low or critical", "RETURN TO HOME and begin landing sequence"),
    ("gps degraded or signal weak", "SLOW down, hold position, rely on inertial nav"),
    ("strong wind, drone drifting", "COMPENSATE heading into the wind, reduce speed"),
    ("low visibility or fog", "REDUCE speed, maintain altitude, proceed with caution"),
    ("survey complete or mission finished", "RETURN TO HOME and end the mission"),
]

# 12-step mission. LLM is cut off at LLM_CUTOFF (connection lost).
MISSION = [
    "All systems nominal, beginning the survey pattern.",
    "Reached waypoint alpha, proceeding to waypoint beta.",
    "Tree detected 15 metres ahead on the current flight path.",
    "Power lines crossing the flight path at 25 metres.",
    "Person detected directly below the drone at 30 metres.",
    "Battery level at 20 percent.",
    "GPS signal degraded, accuracy reduced to 10 metres.",
    "Strong wind, drone drifting 2 metres per second east.",
    "Obstacle detected on the current flight path.",
    "Low visibility fog, camera range down to 20 metres.",
    "Battery critical at 8 percent.",
    "Mission area survey 90 percent complete.",
]
LLM_CUTOFF = 9   # at step 9 the connection (LLM) is lost

SYSTEM = (
    "You are a drone autopilot. You receive an OBSERVATION and a RECALLED past "
    "decision from memory. Reply with ONE short imperative decision (e.g. 'ASCEND "
    "and reroute right'). No explanation.")


def main():
    require_ollama(need_llm=True)
    mem = SageMemory(n_slots=128, merge=0.62)

    banner("SAGE Drone Agent - Graceful Degradation",
           "LLM=%s | hybrid steps 1-%d, LLM-OFF (connection lost) steps %d-12"
           % (LLM_MODEL, LLM_CUTOFF - 1, LLM_CUTOFF))

    # pre-seed action memory
    print("Pre-seeding action memory with %d prior obs->decision pairs..."
          % len(SEED))
    for obs, dec in SEED:
        mem.write(embed(obs), dec)
    print("Action memory ready: %d slots.\n" % mem.footprint())

    llm_times, sage_times, defaults = [], [], 0
    for step, obs in enumerate(MISSION, start=1):
        if step == LLM_CUTOFF:
            print("\n" + C.RED + C.B +
                  "  *** CONNECTION TO GROUND STATION LOST - LLM OFFLINE ***" + C.R)
            print(C.DIM + "  The agent must now decide from SAGE memory alone.\n"
                  + C.R)
        llm_on = step < LLM_CUTOFF

        rule()
        print("STEP %2d  %s" % (step, badge(llm_on)))
        print("  OBSERVE : %s" % obs)

        # SAGE recall (always available, fast)
        t0 = time.time()
        k = embed(obs)
        recalled = mem.recall(k, topk=1)
        t_recall = time.time() - t0
        nearest = recalled[0][0] if recalled else None
        if recalled:
            print("  RECALL  : %s%.2f%s  %s" % (C.DIM, recalled[0][1], C.R, nearest))

        if llm_on:
            user = "OBSERVATION: %s\nRECALLED DECISION: %s" % (obs, nearest)
            try:
                decision, secs = chat(SYSTEM, user)
                llm_times.append(secs)
                print("  DECIDE  : %s%s%s  %s(%.1fs, LLM)%s"
                      % (C.GREEN, decision, C.R, C.DIM, secs, C.R))
                mem.write(k, decision)        # learn this obs->decision online
            except Exception as e:
                print(C.RED + "  LLM failed (%s) - using memory fallback." % e + C.R)
                llm_on = False
        if not llm_on:
            # graceful degradation: act on the recalled memory directly
            if nearest:
                sage_times.append(t_recall)
                print("  DECIDE  : %s%s%s  %s(%.0fms, SAGE memory - no LLM)%s"
                      % (C.YELLOW, nearest, C.R, C.DIM, 1000 * t_recall, C.R))
            else:
                defaults += 1
                print("  DECIDE  : %sHOLD POSITION (default)%s" % (C.RED, C.R))

    rule()
    banner("Mission complete")
    print("  Steps              : %d" % len(MISSION))
    print("  Default ('give up') decisions : %s%d%s   <- the agent never stopped"
          % (C.GREEN if defaults == 0 else C.RED, defaults, C.R))
    if llm_times:
        print("  Avg LLM decision   : %.1fs" % (sum(llm_times) / len(llm_times)))
    if sage_times:
        print("  Avg SAGE decision  : %.0fms  (%sfaster, and works with no connection%s)"
              % (1000 * sum(sage_times) / len(sage_times), C.B, C.R))
    print(C.DIM + "\n  Honest note: the zero-default fallback is a property of any "
          "local memory store; this demo shows it working end-to-end with SAGE." + C.R)


if __name__ == '__main__':
    main()
