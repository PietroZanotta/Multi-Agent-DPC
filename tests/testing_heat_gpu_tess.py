# import subprocess
# import time
# import requests
# import numpy as np
# import os

# # Configuration
# IMAGE_ID = "sha256:72937ae000194ee94b4dad5a6df80d825498a4f79e40ab2f204e3b6fee414bbb"
# PORT = "8080"
# URL = f"http://localhost:{PORT}/apply"
# CONTAINER_NAME = "heat_solver_bench"

# def manage_container(use_gpu: bool, start=True):
#     """Manages the lifecycle using your verified Docker command."""
#     if not start:
#         subprocess.run(f"docker rm -f {CONTAINER_NAME}", shell=True, capture_output=True)
#         return

#     # Cleanup
#     subprocess.run(f"docker rm -f {CONTAINER_NAME}", shell=True, capture_output=True)
    
#     gpu_flag = "--gpus all" if use_gpu else ""
#     docker_cmd = (
#         f"docker run -d --name {CONTAINER_NAME} {gpu_flag} "
#         f"-p {PORT}:{PORT} {IMAGE_ID} serve --port {PORT} --host 0.0.0.0"
#     )
    
#     label = "RTX 5090 (GPU)" if use_gpu else "CPU"
#     print(f"\n--- Booting {label} Server ---")
#     subprocess.run(docker_cmd, shell=True, check=True)
    
#     # Wait for init
#     wait = 8 if use_gpu else 4
#     print(f"Waiting {wait}s for initialization...")
#     time.sleep(wait)

# def run_simulation_logic():
#     """Your exact working simulation logic."""
#     T_STEPS = 100
#     Z_DIM = 100 # Kept at 100 to ensure no 500 errors
#     XI_DIM = 2 

#     x = np.linspace(-1, 1, Z_DIM)
#     z_init = np.exp(-25 * x**2).astype(np.float32)

#     payload = {
#         "inputs": {
#             "z_init": z_init.tolist(),
#             "xi_init": np.zeros((XI_DIM,), dtype=np.float32).tolist(),
#             "u_seq": np.random.normal(0, 0.1, (T_STEPS, 2)).tolist(),
#             "v_seq": np.random.normal(0, 0.1, (T_STEPS, 2)).tolist()
#         }
#     }

#     # Warm-up (Compiles JAX)
#     print("  Warming up...")
#     requests.post(URL, json=payload, timeout=60)

#     # Timed Run
#     print("  Running timed benchmark...")
#     start_wall = time.perf_counter()
#     response = requests.post(URL, json=payload, timeout=60)
#     response.raise_for_status()
#     end_wall = time.perf_counter()
    
#     data = response.json()
#     return {
#         "wall_ms": (end_wall - start_wall) * 1000,
#         "internal_ms": data.get("compute_time_ms", 0),
#         "device": data.get("device_info", "Unknown")
#     }

# def main():
#     results = {}

#     # --- GPU Run ---
#     try:
#         manage_container(use_gpu=True, start=True)
#         results['gpu'] = run_simulation_logic()
#     except Exception as e:
#         print(f"GPU Run failed: {e}")
#     finally:
#         manage_container(use_gpu=True, start=False)

#     # --- CPU Run ---
#     try:
#         manage_container(use_gpu=False, start=True)
#         results['cpu'] = run_simulation_logic()
#     except Exception as e:
#         print(f"CPU Run failed: {e}")
#     finally:
#         manage_container(use_gpu=False, start=False)

#     # --- Results Table ---
#     if 'gpu' in results and 'cpu' in results:
#         g, c = results['gpu'], results['cpu']
#         print("\n" + "="*65)
#         print(f"{'Metric':<25} | {'GPU (RTX 5090)':<15} | {'CPU':<15}")
#         print("-" * 65)
#         print(f"{'Reported Device':<25} | {g['device'][:15]:<15} | {c['device'][:15]:<15}")
#         print(f"{'Internal Compute (ms)':<25} | {g['internal_ms']:>12.2f} | {c['internal_ms']:>12.2f}")
#         print(f"{'Total Wall Time (ms)':<25} | {g['wall_ms']:>12.2f} | {c['wall_ms']:>12.2f}")
        
#         if g['internal_ms'] > 0:
#             speedup = c['internal_ms'] / g['internal_ms']
#             print("-" * 65)
#             print(f"PURE JAX SPEEDUP: {speedup:.2f}x")
#         print("="*65)

# if __name__ == "__main__":
#     main()

import subprocess
import time
import time as timer
import numpy as np
import matplotlib.pyplot as plt
from tesseract_core import Tesseract

# Configuration
IMAGE_ID = "sha256:72937ae000194ee94b4dad5a6df80d825498a4f79e40ab2f204e3b6fee414bbb"
PORT = "52863"
CONTAINER_NAME = "heat_solver_bench"

def manage_container(use_gpu=False, start=True):
    """
    Manually handles Docker lifecycle. 
    start=True: Boots the container.
    start=False: Kills the container.
    """
    if not start:
        print(f"--- Cleaning up {CONTAINER_NAME} ---")
        subprocess.run(f"docker rm -f {CONTAINER_NAME}", shell=True, capture_output=True)
        return

    # Cleanup any existing zombie first
    subprocess.run(f"docker rm -f {CONTAINER_NAME}", shell=True, capture_output=True)
    
    gpu_flag = "--gpus all" if use_gpu else ""
    docker_cmd = (
        f"docker run -d --name {CONTAINER_NAME} {gpu_flag} "
        f"-p {PORT}:{PORT} {IMAGE_ID} serve --port {PORT} --host 0.0.0.0"
    )
    print(f"--- Booting {'GPU' if use_gpu else 'CPU'} Server ---")
    subprocess.run(docker_cmd, shell=True, check=True)
    
    # Wait for JAX/Driver initialization
    wait_time = 8 if use_gpu else 4
    print(f"Waiting {wait_time}s for initialization...")
    time.sleep(wait_time)

def run_science_benchmark(use_gpu: bool):
    client = Tesseract.from_url(f"http://localhost:{PORT}")

    T_STEPS, Z_DIM, XI_DIM = 100, 100, 2
    x = np.linspace(-1, 1, Z_DIM)
    z_init = np.exp(-25 * x**2).astype(np.float32)

    inputs = {
        "z_init": z_init.tolist(),
        "xi_init": np.zeros((XI_DIM,)).tolist(),
        "u_seq": np.random.normal(0, 0.1, (T_STEPS, 2)).tolist(),
        "v_seq": np.random.normal(0, 0.1, (T_STEPS, 2)).tolist()
    }

    print(f"  [{'GPU' if use_gpu else 'CPU'}] Warming up...")
    client.apply(inputs)

    print(f"  [{'GPU' if use_gpu else 'CPU'}] Running Simulation...")
    start_wall = timer.perf_counter()
    result = client.apply(inputs)
    end_wall = timer.perf_counter()

    # The Tesseract client automatically unpacks data, 
    # but we cast to numpy to be safe for plotting.
    return {
        "x": x,
        "z_traj": np.array(result['z_trajectory']),
        "wall_ms": (end_wall - start_wall) * 1000,
        "internal_ms": result.get("compute_time_ms", 0),
        "device": result.get("device_info", "Unknown"),
        "t_steps": T_STEPS
    }

def main():
    results = {}
    try:
        # 1. Run GPU Test
        manage_container(use_gpu=True, start=True)
        results['gpu'] = run_science_benchmark(use_gpu=True)
        manage_container(start=False)

        # 2. Run CPU Test
        manage_container(use_gpu=False, start=True)
        results['cpu'] = run_science_benchmark(use_gpu=False)
        manage_container(start=False)

        # 3. Print Comparison
        print("\n" + "="*65)
        print(f"{'Metric':<25} | {'RTX 5090 (GPU)':<15} | {'CPU':<15}")
        print("-" * 65)
        print(f"{'Device Info':<25} | {results['gpu']['device'][:15]:<15} | {results['cpu']['device'][:15]:<15}")
        print(f"{'Internal Compute (ms)':<25} | {results['gpu']['internal_ms']:>12.2f} | {results['cpu']['internal_ms']:>12.2f}")
        print(f"{'Total Wall Time (ms)':<25} | {results['gpu']['wall_ms']:>12.2f} | {results['cpu']['wall_ms']:>12.2f}")
        
        speedup = results['cpu']['internal_ms'] / results['gpu']['internal_ms']
        print("-" * 65)
        print(f"SPEEDUP: {speedup:.2f}x")
        print("="*65)

    except Exception as e:
        print(f"Process failed: {e}")
        # Final emergency cleanup
        subprocess.run(f"docker rm -f {CONTAINER_NAME}", shell=True, capture_output=True)

if __name__ == "__main__":
    main()