import sys
import os
import hydra
from omegaconf import DictConfig

# Add both submodules and their parent directory to path
project_root = os.path.dirname(os.path.abspath(__file__))
cbgeopy_path = os.path.join(project_root, "cbgeopy")
gns_path = os.path.join(project_root, "gns")
sys.path.insert(0, cbgeopy_path)
sys.path.insert(0, gns_path)
sys.path.insert(0, project_root)

import genenrate_input

@hydra.main(version_base=None, config_path=".", config_name="config")
def run_le(cfg: DictConfig):
    genenrate_input.generate_input(cfg=cfg, save_dir=f'{cfg.save_dir}/sim_le/')

if __name__ == "__main__":
    run_le()