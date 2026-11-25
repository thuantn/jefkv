"""Kaggle kernel entry point: copy the code dataset, install pinned deps, run one stage.

Needs: GPU accelerator + internet on (Kaggle requires a phone-verified account for both).
Results land in /kaggle/working/jefkv/{runs,outputs}; fetch them with `kaggle kernels output`.
"""

import glob
import os
import shutil
import subprocess
import sys

STAGE = os.environ.get("JEFKV_STAGE", "gate")  # any run_all.sh stage

src = os.path.dirname(glob.glob("/kaggle/input/**/run_all.sh", recursive=True)[0])
work = "/kaggle/working/jefkv"
shutil.copytree(src, work, dirs_exist_ok=True)
os.chdir(work)
# Kaggle's own torch is kept; only the pinned libraries are installed.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
subprocess.run([sys.executable, "-c", "import torch;print(torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"], check=True)
subprocess.run(["bash", "run_all.sh", STAGE], check=True)
# Keep the output small: adapters are not needed to read the results.
for f in glob.glob("runs/*/adapter.pt"):
    os.remove(f)
