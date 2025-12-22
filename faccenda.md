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
- other pde example
- how to exploit tesseract for gpu 
- use barrier function instead of bondaru penalties potentially
- vedi se per fkpp siamo in bolla lato bc