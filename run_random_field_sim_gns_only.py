import sys
import os
import shutil
import importlib

# Add both submodules and their parent directory to path
project_root = os.path.dirname(os.path.abspath(__file__))
cbgeopy_path = os.path.join(project_root, "cbgeopy")
gns_path = os.path.join(project_root, "gns")

# Add paths in order: submodules first, then project root
sys.path.insert(0, cbgeopy_path)
sys.path.insert(0, gns_path)
sys.path.insert(0, project_root)

import numpy as np
import glob
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import run_mpm
import pickle

from cbgeopy.gns_preprocess.h5_to_npz import convert_hd5_to_npz
from gns.gns import reading_utils
from gns.gns import particle_data_loader
import run_gns
import data_handle
from postprocess import PostProcesser


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path="./tmp", config_name="config")
def main(cfg):
    
    n_gns_trajectory_frames = cfg.gns_config.nsteps + 6
    results = []
    
    for i in np.arange(cfg.sim_id_range[0], cfg.sim_id_range[1]):
        result = {}
        
        # Set simulation directory
        sim_id = f"sim_{i}"
        sim_dir = f'{cfg.save_dir}/{sim_id}/'
        
        # Create save directory if it doesn't exist
        if not os.path.exists(sim_dir):
            os.makedirs(sim_dir)

        # Save config
        OmegaConf.save(cfg, f'{cfg.save_dir}/config.yaml')
        
        # Generate random field input
        input_generator = importlib.import_module(cfg.mpm_config.input_generator_module_name)
        # input_generator.generate_input(sim_dir, cfg)
        
        # Get MPM configuration without calling generate_input
        mpm = input_generator.create_mpm_config(cfg)

        # Save input generator used
        module_file = input_generator.__file__
        module_filename = os.path.basename(module_file)
        shutil.copy2(module_file, f'{cfg.save_dir}/{module_filename}')
            
        # Make x0
        npz_name = "trajectory_mpm_initial_steps.npz"
        x0 = data_handle.prepare_x0(
            cfg, 
            x0_option="from_same_5vels",
            npz_path=f'{sim_dir}/{npz_name}',
            material_params=torch.tensor([1.0, 1.0], device=device).unsqueeze(0),
            device=device,
            mpm=mpm,
        )
        
        # Run GNS -> save trajectory.npz
        result_dict = run_gns.forward(x0, cfg, device)
        
        # Save the results as npz
        run_gns.save_results(result_dict, f'{sim_dir}/trajectory_gns_full.npz')
        
        # Postprocess the results
        postprocesser = PostProcesser(
            gns_trajectory_path=f'{sim_dir}/trajectory_gns_full.npz',
            mpm_trajectory_path=f'{sim_dir}/trajectory_mpm_full.npz' if i % cfg.full_mpm_interval == 0 else None
        )
        runout_gns, runout_mpm = postprocesser.compute_runout()
        mse = postprocesser.compute_mse()
        postprocesser.plot_deposit(save_dir=sim_dir, domain=cfg.domain)
        
        # Save the results
        result["gns"] = {"runout": runout_gns}
        result["mpm"] = {"runout": runout_mpm}
        result["mse"] = mse
        
        # Append the result to the results list
        # results.append(result)
        
    # # Save the results
    # with open(f'{cfg.save_dir}/results.pkl', 'wb') as f:
    #     pickle.dump(results, f)


if __name__ == "__main__":
    main()