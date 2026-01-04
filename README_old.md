# Tesseract Hackathon Template

A ready-to-use template for building projects with [Tesseract Core](https://github.com/pasteurlabs/tesseract-core) and [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax), featuring two interacting tesseracts that demonstrate vector scaling and similarity computation. Intended as a starting point for participants of the [Tesseract Hackathon](link TODO).

> [!WARNING]
> Using this template is *not* required to participate in the Hackathon. You may use any tools at your disposal, including [Tesseract Core](https://github.com/pasteurlabs/tesseract-core), [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax), and [Tesseract-Streamlit](https://github.com/pasteurlabs/tesseract-streamlit) — or composing Tesseracts via `docker run` calls in a glorified shell script. Your imagination is the limit!

#### See also
- [Tesseract Core Documentation](https://github.com/pasteurlabs/tesseract-core)
- [Tesseract-JAX Documentation](https://github.com/pasteurlabs/tesseract-jax)
- [Tesseract showcase](https://si-tesseract.discourse.group/c/showcase/11)
- [Get help @ Tesseract User Forums](https://si-tesseract.discourse.group/)

## Overview

This template demonstrates how to:
1. Define Tesseracts ([`tesseracts/*`](tesseracts)).
2. Build them locally ([`buildall.sh`](buildall.sh)).
3. Serve Tesseracts locally, and compose them into a (differentiable) pipeline via the [Tesseract Core SDK](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/api/tesseract-api.html) and [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax) ([`main.py`](main.py)).

### Included Tesseracts

Example Tesseracts are minimal and meant as starting point for you to build upon.

1. scaler ([`tesseracts/scaler`](tesseracts/scaler))
   - Scales input vectors by a given factor.
   - Implements a vector-Jacobian product by hand for autodiff.
2. dotproduct ([`tesseracts/dotproduct`](tesseracts/dotproduct))
   - Computes dot product between two vectors.
   - Calculates cosine similarity.
   - Uses the [Tesseract JAX recipe](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/creating-tesseracts/create.html#initialize-a-new-tesseract) to enable automatic differentiation.

### Pipeline Demo

The example script [`main.py`](main.py) demonstrates two ways to compose Tesseracts into pipelines.

#### Path 1: Calling Tesseracts manually

- Call Tesseracts via [Tesseract Core SDK](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/api/tesseract-api.html).

#### Path 2: Composing Tesseracts with Tesseract-JAX

- Wrap Tesseract calls in a differentiable JAX function using [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax).

## Get Started

### Prerequisites

- Python 3.10 or higher, ideally with a virtual environment (e.g. via `venv`, `conda`, or `uv`).
- Working Docker setup for the current user ([Docker Desktop recommended](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/introduction/installation.html#installing-docker)).

### Quickstart

1. Create a new repository off this template and clone it
   ```bash
   $ git clone <your-repo-url>
   $ cd <myrepo>
   ```

2. Set up virtual environment (if not done already). `uv` or `conda` can also be used.
   ```bash
   $ python3 -m venv .venv
   $ source .venv/bin/activate
   ```

3. Install dependencies
   ```bash
   $ pip install -r requirements.txt
   ```

4. Build Tesseracts
   ```bash
   $ ./buildall.sh
   ```

5. Run the example pipeline
   ```bash
   $ python main.py
   ```

## Code Structure

This repository implements **Differentiable Predictive Control (DPC)** for PDE systems using neural operator-based controllers. The project demonstrates how to train DeepONet controllers that command mobile actuators to control partial differential equations (heat, Fisher-KPP, Navier-Stokes) using the Tesseract framework for end-to-end differentiable pipelines.

### Directory Organization

```
.
├── examples/                    # Training experiments for different PDEs
│   ├── heat1/                  # 1D Heat equation control
│   │   ├── centralized/        # Global sensing controller
│   │   └── decentralized/      # Local sensing controller
│   ├── fkpp1d/                 # 1D Fisher-KPP reaction-diffusion
│   │   ├── centralized/
│   │   └── decentralized/
│   └── ns2D*/                  # 2D Navier-Stokes variants
│       └── centralized/
├── tesseracts/                 # Containerized autodifferentiable solvers
│   ├── solverHeat_centralized/
│   ├── solverHeat_decentralized/
│   ├── solverFKPP_centralized/
│   ├── solverFKPP_decentralized/
│   └── solverNS_shape*/
├── models/                     # Shared neural architectures
│   └── policy.py              # DeepONet-based controllers
├── tests/                      # Unit tests for solvers
├── buildall.sh                # Build all tesseracts
├── run_examples.sh            # Run all 1D experiments
└── main.py                    # Template demo (not used in experiments)
```

### Control Loop Architecture

All experiments follow this pattern:

1. **Policy Network** (DeepONet): Observes error field `e(x,t) = z(x,t) - z_target(x,t)` and outputs control actions `(u, v)` per actuator
   - `u` = control intensity (forcing strength)
   - `v` = velocity (actuator movement)

2. **Actuator Forcing**: Applies Gaussian-filtered forcing `B(x,t) = Σ u_i(t) · b(x, ξ_i(t))` where `b` is a spatial kernel

3. **PDE Solver**: Advances state `∂z/∂t = A(z) + B(x,t)` using numerical methods (Crank-Nicolson, operator splitting, spectral)

4. **Gradient Flow**: JAX autodiff propagates gradients backward through the entire trajectory for end-to-end training

### Neural Policy Structure

Policies inherit a **Branch-Trunk-Fusion** architecture (DeepONet variant):

- **Branch Network**: Processes error field
  - Centralized: Full (error, ∇error) field
  - Decentralized: Local patch around each agent

- **Trunk Network**: Encodes actuator coordinates using Fourier features `[sin(kπξ), cos(kπξ)]`

- **Fusion Layer**: Combines Branch + Trunk outputs → `(u, v)` control commands

### Training Paradigms

**Direct JAX Training** (1D problems):
- PDE solver in pure JAX (`dynamics_dual.py`)
- Fast JIT-compiled training
- See `examples/heat1/centralized/train.py`

**Tesseract-Based Training** (2D NS):
- Solver wrapped as containerized service with custom VJP
- Enables distributed deployment
- Tesseract exposes `/apply` and `/vjp` HTTP endpoints
- See `tesseracts/solverNS_shape/tesseract_api.py`

### Key Innovations

1. **Zero-shot scalability**: Train with N=8 agents, deploy with N=100 (decentralized only)
2. **Discretization-invariant gradients**: Loss magnitude independent of agent count
3. **Stigmergic coordination**: Agents coordinate through sensing `∂e/∂t`, not communication
4. **End-to-end differentiability**: Gradients flow through PDE solver via Tesseract VJP

### Typical Workflow

```bash
# 1. Build solvers as tesseracts
./buildall.sh

# 2. Train controller for specific PDE
cd examples/heat1/centralized/
python data_utils.py    # Generate GRF training data
python train.py         # Train controller (saves trained_params.pkl)
python visualize.py     # Plot results

# 3. Or run all 1D experiments
./run_examples.sh
```

### Centralized vs Decentralized

**Centralized Controllers**:
- Observe entire error field (200-D for 100 grid points)
- Performance upper bound
- Fixed agent count

**Decentralized Controllers**:
- Each agent observes 8-point local patch
- Zero communication between agents
- 48% fewer parameters
- Generalizes to arbitrary agent counts
- ~50% performance degradation vs centralized

For more details, see [CLAUDE.md](CLAUDE.md) and [main.tex](main.tex) for the theoretical foundation.

## Now go and build your own!

Some pointers to get you started:

1. **Change Tesseract definitions**.
     - Just update the code in `tesseracts/*`. You can add / remove Tesseracts at will, and `buildall.sh` will... build them all.
     - Make sure to check out the [Tesseract docs](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/creating-tesseracts/create.html) to learn how to adapt existing configuration and define Tesseracts from scratch.
2. **Use gradients to perform optimization**.
   - Exploit that Tesseract pipelines with AD endpoints are [end-to-end differentiable](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/introduction/differentiable-programming.html).
   - Check [showcases](https://si-tesseract.discourse.group/c/showcase/11) for inspiration, e.g. the [Rosenbrock optimization showcase](https://si-tesseract.discourse.group/t/jax-based-rosenbrock-function-minimization/48) for a minimal demo.
3. **Deploy Tesseracts anywhere**.
   - Since built Tesseracts are just Docker images, you can [deploy them virtually anywhere](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/creating-tesseracts/deploy.html).
   - This includes [HPC clusters via SLURM](https://si-tesseract.discourse.group/t/deploying-and-interacting-with-tesseracts-on-hpc-clusters-using-tesseract-runtime-serve/104).
   - Have a look at [Tesseract Streamlit](https://github.com/pasteurlabs/tesseract-streamlit) that can turn Tesseracts into web apps.
   - Show us how and where you run Tesseracts over the local network, on clusters, or in the cloud!
4. **Happy Hacking!** 🚀
   - Don't let these pointers constrain you. We're looking for creative solutions, so thinking out of the box is always appreciated.
   - Have fun, and [reach out](https://si-tesseract.discourse.group/) if you need help.

## License

Licensed under Apache License 2.0.

All submissions must use the Apache License 2.0 to be eligible for the Tesseract Hackathon. See [LICENSE](LICENSE) file for details.
