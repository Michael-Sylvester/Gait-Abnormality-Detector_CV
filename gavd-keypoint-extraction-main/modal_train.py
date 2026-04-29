"""
Modal script for running ST-GCN training on a GPU cloud instance.

Prerequisites:
    modal setup   # one-time authentication

Workflow:

  1. Upload keypoint data to a persistent Modal volume (run once):
        modal run modal_train.py --upload

  2. Train from scratch:
        modal run modal_train.py

  3. Resume an interrupted run:
        modal run modal_train.py --resume

  4. Override any training hyperparameter, e.g.:
        modal run modal_train.py --epochs 100 --batch-size 32

  5. Download results after training:
        modal volume get gavd-runs best_model.pt       stgcn/runs/best_model.pt
        modal volume get gavd-runs last_checkpoint.pt  stgcn/runs/last_checkpoint.pt
        modal volume get gavd-runs history.json        stgcn/runs/history.json
        modal volume get gavd-runs test_results.json   stgcn/runs/test_results.json
"""

from pathlib import Path
import modal

# ── container image ────────────────────────────────────────────────────────
# Install PyTorch with CUDA 12.4 (matches Modal's A10G driver stack),
# then bake the local source packages into the image so they're importable
# on the remote container without any mount overhead at run time.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .run_commands(
        "pip install torch --index-url https://download.pytorch.org/whl/cu124 -q"
    )
    .pip_install("scikit-learn", "pandas", "numpy")
    .add_local_python_source("stgcn", "baseline", "src")
)

# ── persistent volumes ─────────────────────────────────────────────────────
# gavd-keypoints : the .npy / .json / CSV dataset  (written once on upload)
# gavd-runs      : checkpoints and results          (written every epoch)
data_vol = modal.Volume.from_name("gavd-keypoints", create_if_missing=True)
runs_vol = modal.Volume.from_name("gavd-runs",      create_if_missing=True)

# Paths inside the remote container
DATA_DIR        = "/data/keypoints"
RUNS_DIR        = "/runs"
RUNS_DIR_ABLATE = "/runs/ablation"

app = modal.App("gavd-stgcn")


# ── shared training helper (plain function, runs inside the container) ─────
def _run_training(out_dir, in_channels, epochs, batch_size, lr, dropout, fixed_len, seed, resume):
    import subprocess, sys

    cmd = [
        sys.executable, "-m", "stgcn.train",
        "--keypoint_dir", DATA_DIR,
        "--out_dir",      out_dir,
        "--epochs",       str(epochs),
        "--batch_size",   str(batch_size),
        "--lr",           str(lr),
        "--dropout",      str(dropout),
        "--fixed_len",    str(fixed_len),
        "--seed",         str(seed),
        "--in_channels",  str(in_channels),
    ]
    if resume:
        cmd.append("--resume")

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    runs_vol.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")


# ── full training (in_channels=3) ──────────────────────────────────────────
@app.function(
    image=image,
    gpu="A10G",
    volumes={"/data": data_vol, "/runs": runs_vol},
    timeout=3600 * 8,
)
def train_remote(
    epochs:      int   = 80,
    batch_size:  int   = 32,
    lr:          float = 1e-3,
    dropout:     float = 0.5,
    fixed_len:   int   = 300,
    seed:        int   = 42,
    resume:      bool  = False,
):
    _run_training(RUNS_DIR, 3, epochs, batch_size, lr, dropout, fixed_len, seed, resume)
    print("\nResults saved to volume 'gavd-runs'. Download with:")
    for fname in ("best_model.pt", "last_checkpoint.pt", "history.json", "test_results.json"):
        print(f"  modal volume get gavd-runs {fname} stgcn/runs/{fname}")


# ── channel ablation function ──────────────────────────────────────────────
@app.function(
    image=image,
    gpu="A10G",
    volumes={
        "/data": data_vol,
        "/runs": runs_vol,
    },
    timeout=3600 * 8,
)
def ablate_remote(
    epochs:     int   = 80,
    batch_size: int   = 32,
    lr:         float = 1e-3,
    dropout:    float = 0.5,
    fixed_len:  int   = 300,
    seed:       int   = 42,
    resume:     bool  = False,
):
    """Train with in_channels=2 (x,y only — no confidence). Results go to
    a separate /runs/ablation directory so they never overwrite the main run."""
    _run_training(RUNS_DIR_ABLATE, 2, epochs, batch_size, lr, dropout, fixed_len, seed, resume)

    print("\nAblation results saved. Download with:")
    for fname in ("best_model.pt", "last_checkpoint.pt", "history.json", "test_results.json"):
        print(f"  modal volume get gavd-runs ablation/{fname} stgcn/runs/ablation/{fname}")


# ── local entrypoint ───────────────────────────────────────────────────────
@app.local_entrypoint()
def main(
    upload:     bool  = False,
    ablate:     bool  = False,
    resume:     bool  = False,
    epochs:     int   = 80,
    batch_size: int   = 32,
    lr:         float = 1e-3,
    dropout:    float = 0.5,
    fixed_len:  int   = 300,
    seed:       int   = 42,
):
    """
    --upload          Upload keypoint data to the Modal volume (run once).
    --ablate          Run the channel ablation (x,y only, no confidence).
    --resume          Resume from the last checkpoint of whichever run is selected.
    All other flags   Hyperparameters forwarded to the training script.
    """
    if upload:
        local_dir = Path("gavd_handoff/data/keypoints")
        files = list(local_dir.iterdir())
        print(f"Uploading {len(files)} files to Modal volume 'gavd-keypoints'...")
        with data_vol.batch_upload() as batch:
            batch.put_directory(str(local_dir), "/keypoints")
        print("Upload complete. Run without --upload to start training.")
        return

    kwargs = dict(epochs=epochs, batch_size=batch_size, lr=lr,
                  dropout=dropout, fixed_len=fixed_len, seed=seed, resume=resume)

    if ablate:
        print("Running channel ablation (in_channels=2, x,y only)...")
        ablate_remote.remote(**kwargs)
    else:
        print("Running full training (in_channels=3, x,y,conf)...")
        train_remote.remote(**kwargs)
