dpc_project/
├── examples/
│   ├── heat1d/
│       ├── dynamics.py                 # Wraps apply_tesseract for a single time step
│       └── train.py                    # Main unrolling loop (BPTT)
│   ├── another_pde_Nd/
│       ├── dynamics.py
│       └── train.py                    
├── models/
│   └── policy.py                       # JAX model: (state, xi) -> [u, xi_dot]. Includes both centralized and decentralized versions 
│
├── tesseracts/                         # The "Legacy" Simulators
│   └── solverV1/                       # For heat1d
│       ├── tesseract_api.py            # Must define 'apply' and 'vjp'
│       ├── solver.py                   # Our legacy C++/Julia/Python FEM code (Jax in this case) logic
│       └── Dockerfile                  # Environment for the legacy solver (not necessary if we use a Jax solver)
│   └── solverV2/                       # For another PDE
│       ├── tesseract_api.py 
│       ├── solver.py  
│       └── Dockerfile       
│  
└── README                            



to do:
<!-- - correct the sliding behaviour, likely from numerical error in the gradient computation -->
- nuovi esperimentini:
    - cambia training
    - basic zero-shot transfer: train on 10, deploy on 6, 8, 12, 15, 20  
    - plot \hat u vs M (testing for self-normalization)
    - plot ||B|| vs M (forcing consistency)
    - 
    - cross pde transfer (Train on Heat equation ($\nu = 0.2$), test on Heat equation with different $\nu \in \{0.1, 0.3, 0.5\}$, then test on Fisher-KPP (if tracking target is similar))
- clean up the code

