import os

# Avoid native OpenMP crashes when the two CPU models run together on macOS.
os.environ.setdefault("OMP_NUM_THREADS", "1")
