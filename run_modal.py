"""Run a stage of run_all.sh on a Modal T4 (alternative to free Colab).

    MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=... modal run run_modal.py --stage gate
    python run_modal.py factkg_seeds '^fk_rag_s1$'   # after modal deploy: one run per T4, in parallel
    modal volume get jefkv-runs / runs_modal        # fetch runs/*, incl. _outputs/ (table + plots)

runs/ lives on the Volume, so a re-run skips finished configs (run_all.sh checks eval.json).
"""

import os
import shutil
import subprocess

import modal

app = modal.App("jefkv")
vol = modal.Volume.from_name("jefkv-runs", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", "/root/jefkv", ignore=[".git", ".venv", "**/__pycache__", "runs*", "outputs*", "data/smoke",
                                               "data/processed", "data/fresh", "data/factkg/test_kg*", "docs", "*.log", "*.pt"])
)


@app.function(image=image, gpu="T4", timeout=24 * 3600, volumes={"/root/jefkv/runs": vol})
def run(stage: str = "gate", only: str = ""):
    try:  # only: ONLY regex of run_all.sh, so one stage can be spread over several GPUs
        subprocess.run(["bash", "run_all.sh", stage], cwd="/root/jefkv", check=True, env={**os.environ, "ONLY": only or "."})
        if os.path.isdir("/root/jefkv/outputs"):  # stages that exit before plotting (unf) have none
            shutil.copytree("/root/jefkv/outputs", "/root/jefkv/runs/_outputs", dirs_exist_ok=True)
    finally:
        vol.commit()


@app.local_entrypoint()
def main(stage: str = "gate"):
    run.remote(stage)


if __name__ == "__main__":
    # `modal run --detach` still stopped the app ~3 min after the local client lost its network.
    # Independent of the client:  modal deploy run_modal.py && python run_modal.py <stage>
    #   then `modal app stop jefkv` when the volume has the results.
    import sys

    # python run_modal.py <stage> [only-regex]: one spawn per call (one T4 each)
    print(modal.Function.from_name("jefkv", "run").spawn(*(sys.argv[1:3] or ["gate"])).object_id)
