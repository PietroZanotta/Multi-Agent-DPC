dpc_project/
├── tesseracts/
│   └── fem_solver/           # The "Legacy" Simulator (just RK4 in our case I'd say)
│       ├── tesseract_api.py  # Must define 'apply' and 'vjp'
│       ├── solver_logic.py   # Your legacy C++/Julia/Python FEM code
│       └── Dockerfile        # Environment for the legacy solver (not necessary if we use a Jax solver)
├── model/
│   └── policy.py             # JAX/Equinox MLP: (state) -> [u, xi_dot]
├── dpc_engine/
│   ├── dynamics.py           # Wraps apply_tesseract for a single time step
│   └── loss_functions.py     # Tracking, Force, and Collision losses
└── train.py                  # Main unrolling loop (BPTT)


to do:
- look at jax-fem
- build the fem solver out of that potentially
- build dpc