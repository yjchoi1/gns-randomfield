import sys
import os

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
import genenrate_input
import run_mpm
import pickle

from cbgeopy.gns_preprocess.h5_to_npz import convert_hd5_to_npz
from gns.gns import reading_utils
from gns.gns import particle_data_loader
import run_gns
import data_handle
from postprocess import PostProcesser


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    n_mpm_full_trajectory_frames = cfg.mpm_config.n_full_mpm_timesteps // 375
    n_gns_trajectory_frames = cfg.gns_config.nsteps + 6
    if n_mpm_full_trajectory_frames != n_gns_trajectory_frames:
        raise ValueError(f"Forward simulation n_frames for MPM and GNS should be the same")
    
    results = []
    
    for i in range(cfg.n_samples):
        result = {}
        
        # Set simulation directory
        sim_id = f"sim_{i}"
        sim_dir = f'{cfg.save_dir}/{sim_id}/'
        
        # Create save directory if it doesn't exist
        if not os.path.exists(sim_dir):
            os.makedirs(sim_dir)
        
        # Generate random field input
        genenrate_input.generate_input(sim_dir, cfg)
    
        # Run MPM for entire simulation if `i` is at specifed index
        if i % cfg.full_mpm_interval == 0:
            run_mpm.run_mpm(cfg, sim_dir, 'mpm-resume-full.json')
            
            # Convert h5 to npz for the full MPM simulation
            convert_hd5_to_npz(
                n_dims=cfg.n_dims,
                n_material_props=cfg.n_material_props,
                max_cohesion=cfg.max_cohesion,
                result_dir=f'{sim_dir}/results/sand2d',
                mpm_input_path=f'{sim_dir}/mpm-resume-full.json',
                save_path=f'{sim_dir}/trajectory_mpm_full.npz',
                use_material_feature=True,
                scale={"origin": [0, 0], "factor": [1.0, 1.0], "timestep_stride": int(1)},
                max_time=cfg.mpm_config.n_full_mpm_timesteps)
            
        # Run MPM for first few initial steps if `i` is not at specifed index
        else:
            run_mpm.run_mpm(cfg, sim_dir, 'mpm-resume-initial-steps.json')
            
            # Convert h5 to npz for the first few initial steps
            convert_hd5_to_npz(
                n_dims=cfg.n_dims,
                n_material_props=cfg.n_material_props,
                max_cohesion=cfg.max_cohesion,
                result_dir=f'{sim_dir}/results/sand2d',
                mpm_input_path=f'{sim_dir}/mpm-resume-initial-steps.json',
                save_path=f'{sim_dir}/trajectory_mpm_initial_steps.npz',
                use_material_feature=True,
                scale={"origin": [0, 0], "factor": [1.0, 1.0], "timestep_stride": int(1)},
                max_time=cfg.mpm_config.n_first_initial_timesteps)
            
        # Make x0
        npz_name = "trajectory_mpm_full.npz" if i % cfg.full_mpm_interval == 0 else "trajectory_mpm_initial_steps.npz"
        x0 = data_handle.prepare_x0(
            cfg, 
            x0_option="from_same_5vels",
            npz_path=f'{sim_dir}/{npz_name}',
            material_params=torch.tensor([1.0, 1.0], device=device).unsqueeze(0),
            device=device,
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
        
        # Save the results
        result["gns"] = {"runout": runout_gns}
        result["mpm"] = {"runout": runout_mpm}
        result["mse"] = mse
        
        # Append the result to the results list
        results.append(result)
        
    # Save the results
    with open(f'{cfg.save_dir}/results.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()