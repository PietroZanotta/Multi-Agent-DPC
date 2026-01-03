import jax
import tesseract_core
import tesseract_jax # if installed
import platform

print(f"JAX Version: {jax.__version__}")
print(f"JAX default backend: {jax.default_backend()}")
print(f"Tesseract-Core: {tesseract_core.__version__}")
try:
    print(f"Tesseract-JAX: {tesseract_jax.__version__}")
except:
    print("Tesseract-JAX: Not found via __version__")
print(f"OS: {platform.platform()}")