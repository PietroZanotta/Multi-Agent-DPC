# Multi-Agent Differentiable Predictive Control for Zero-Shot PDE Scalability

This project introduces and experiments with a decentralized control framework for systems described by PDEs. Leveraging [Tesseract-Jax](https://github.com/pasteurlabs/tesseract-jax) to implement the PDE solver as a differentiable layer, we leverage the [Differentiable Predictive Control](https://arxiv.org/abs/2011.03699) framework to enable autonomous agents to interact with the physical field for trajectory tracking.

This project was ideated and evaluated by [Pietro Zanotta](https://github.com/PietroZanotta)<sup>1</sup>, [Dibakar Roy](https://github.com/RoyDibs)<sup>1</sup> and [Honghui Zheng](https://github.com/Honghui-Zheng). 

Contacts:
- Pietro Zanotta: pzanott1@jhu.edu
- Dibakar Roy: droysar1@jh.edu
- Honghui Zheng: hzheng39@jh.edu

<sup>1</sup>: shared first authorship

---

## Key Features
- **Differentiable Operator Learning for Control**: we recast policy synthesis for PDE systems as an operator learning problem using the DeepONet framework. By treating the PDE solver as a differentiable layer through the Tesseract differentiable programming library, we compute exact sensitivity gradients for policy optimization then used within the *Differentiable Predictive Control* framewok.
- **Zero-Shot Scalability**: Policies trained on a fixed swarm size $N$ generalize to unseen cardinalities $M$ (e.g., training on 20 agents and deploying on 60) without further tuning.
- **Communication-Free Coordination:** We test the scenarion where agents operate using local-only sensing and zero inter-agent communication, where we observe an *emerging self-normalization property*, coming from stigmergic interaction, preventing overactuation. 
- **Theoretical Gradient Consistency**: We provide a mathematical foundation theorem ensuring that discrete policy gradients converge to the mean-field limit as the swarm size $N \rightarrow \infty$.
- **Parameter Efficiency:** In our toy examples, the decentralized approach utilizes *48% fewer parameters* than centralized benchmarks while maintaining competitive performance.

For a more rigorous discussion about all the above points we suggest reading through our technical document.

---

## TOC
- Numerical Experiments
- About this Project
- Structure of this Repository
- Getting Started
- Future Work
- Tech Stack

---

## Numerical Experiments

The framework was validated on two primary physical systems:
1.  **Linear Heat Equation:** Focused on temperature tracking and heat spreading.
2.  **Nonlinear Fisher-KPP Equation:** Modeled population dynamics and chemical fronts, where agents must overcome natural growth to achieve stability.

### Performance Summary

| Metric                     | Heat (Centralized) | Heat (Decentralized) | Fisher-KPP (Centralized) | Fisher-KPP (Decentralized) |
| :------------------------- | :----------------: | :------------------: | :----------------------: | :------------------------: |
| **Branch Input Dim**       | 200                | 40                   | 200                      | 40                         |
| **Total Parameters**       | 21,794             | 11,298               | 21,794                   | 11,298                     |
| **Final Tracking Loss**    | 5.2e-3             | 6.4e-3               | 7.0e-3                   | 8.3e-3                     |
| **Scalability**            | Zero-shot          | Zero-shot            | Zero-shot                | Zero-shot                  |
| **Communication**          | Global             | None                 | Global                   | None                       |
| **Training Time (500 ep.)**| ~1 min             | ~1 min               | ~3 min                   | ~3 min                     |

---

## About this Project

---

## Structure of this Repository
```text
tesseract-hackathon/
├── examples/                       # High-level scripts for specific PDE problems
│   ├── fkpp1d/                     # Fisher-KPP 1D reaction-diffusion examples
│   │   ├── centralized/            # Training and visualization for global control
│   │   └── decentralized/          # Multi-agent/local control versions
│   └── heat1d/                     # 1D Heat Equation examples
│       ├── centralized/
│       └── decentralized/
│
├── models/                         # Core neural network architectures
│   └── policy.py                   # JAX implementation of the DPC policies
│
├── tesseracts/                     # The "Legacy" Simulator Wrappers
│   ├── solverFKPP_.../             # Solvers specifically for FKPP problems
│   ├── solverHeat_.../             # Solvers specifically for Heat problems
│   │   ├── solver.py               # The underlying physics engine logic
│   │   ├── tesseract_api.py        # Interface defining 'apply' and 'vjp' for JAX
│   │   └── tesseract_config.yaml
│   └── ...
│
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## Getting Started

1. Create your virtual environment: 
```bash
python -m venv .venv
```
and activate it:
- Linux/MacOS:
```bash
source .venv/bin/activate
```
- Windows:
```powershell
.venv/Scripts/activate
```
2. Install requirements:
```bash
pip install -r requirements.txt
```
If you have access to a GPU and are planning to train the policies on your device we suggest you also download `jax[cuda]`.

4. build the Tesseracts of interests (in this demo only the Heat equation Tesseracts. The process is similar for the other Tesseracts):
```bash
# Build the centralized policy Tesseract
cd tesseracts/solverHeat_centralized && Tesseract build .

# Build the decentralized policy Tesseract
cd ../solverHeat_decentralized && Tesseract build .
```
---

## Future Work

---

## Tech Stack

- **Processor:** Intel Core Ultra 9 275HX (24 cores, up to 5.4 GHz)
- **GPU:** NVIDIA GeForce RTX 5090 Laptop GPU (24GB GDDR7 VRAM)
- **Operating System:** Ubuntu 22.04 running under Windows Subsystem for Linux (WSL2)
- **Main Frameworks:** JAX (v0.8.1) for numerical computing; Tesseract-JAX (v0.2.2) for differentiable PDE solvers
- **Hardware Acceleration:** CUDA backend with NVIDIA driver v581.57

See our technical document for details about our experimental setup.
---
