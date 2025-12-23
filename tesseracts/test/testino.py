import subprocess
import time
import requests
import signal
import os

# Configuration
IMAGE_NAME = "vlasov:latest"
PORT = "8080"
URL = f"http://localhost:{PORT}/apply"
PAYLOAD = {"inputs": {"dummy_in": 0.0}}

def run_dpc_workflow():
    # 1. Start the Tesseract Server in the background
    print(f"Starting Tesseract server for {IMAGE_NAME}...")
    server_process = subprocess.Popen(
        ["tesseract", "serve", "--gpus", "all", IMAGE_NAME, "--port", PORT],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid  # Creates a process group so we can kill it easily
    )

    try:
        # 2. Wait for the server to be ready (Health Check)
        print("⏳ Waiting for GPU and JAX to initialize...")
        max_retries = 30
        for i in range(max_retries):
            try:
                # Check if the swagger docs or a simple get works
                response = requests.get(f"http://localhost:{PORT}/docs", timeout=1)
                if response.status_code == 200:
                    print("Server is UP and ready!")
                    break
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        else:
            print("Server failed to start in time.")
            return

        # 3. Your DPC Optimization Loop
        print("🏃 Starting compute iterations...")
        for i in range(100):
            start_wall = time.perf_counter()
            
            response = requests.post(URL, json=PAYLOAD)
            result = response.json()
            
            end_wall = time.perf_counter()
            
            # Extract metrics
            data = result.get("outputs", result)
            gpu_time = data.get("compute_time_ms")
            wall_time = (end_wall - start_wall) * 1000
            
            print(f"Iteration {i:02d} | GPU: {gpu_time:7.2f}ms | Total Wall: {wall_time:7.2f}ms")

    finally:
        # 4. Cleanup: Kill the server process and its children
        print(f"Shutting down Tesseract server...")
        if server_process:
            # Kill the entire process group (including the container)
            os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
            server_process.wait()
            print("Done. GPU memory released.")

if __name__ == "__main__":
    run_dpc_workflow()