Upgrade of /code_trails/Heat. The problem stays the same but:
- the actuator position is not fixed but is a parameter of the problem
- we use tracking and terminal losses (better gradient)
- different sys initialization to avoid getting stuck
- defibrillator (if the plan decays to near-zero we injects noise to revive it)

NOTE: 
- result is quite dependent on the initialization (maybe some heuristic initialization can help, or suing a multi-start approach)
- likely need some advanced weighting techniques in the loss
- the problem is more complex than the version in /code_trails/Heat