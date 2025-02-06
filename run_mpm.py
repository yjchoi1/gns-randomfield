import subprocess
import os

def run_mpm(cfg: str, sim_dir: str, mpm_json: str):
    # Check if initial stress file exits
    if not os.path.exists(f"{cfg.save_dir}/particles-stresses.txt"):
        raise ValueError(f"Initial stress file should exist in {cfg.save_dir}/particles-stresses.txt")

    # Define the bash commands
    commands = f"""
    module reset
    module load intel
    module load libfabric
    mpm -i {mpm_json} -f {sim_dir}
    """

    # Execute the commands
    subprocess.run(commands, shell=True, executable='/bin/bash')
