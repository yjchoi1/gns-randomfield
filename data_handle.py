import numpy as np
import torch
import json
import sys
import glob
from gns import reading_utils
from gns import particle_data_loader

LINEAR_ELASTIC_FEATURE_DEG = 45.0
COHESION_DEFAULT = 1000

def prepare_x0(
    cfg,
    x0_option: str, 
    npz_path: str,
    material_params: torch.Tensor,
    device: torch.device,
    mpm=None,
):
    """Prepare initial features (X0) for rollout on torch device.

    Args:
        x0_option (str): Option for preparing initial features (X0)
        npz_path (str): Path to the `.npz` file containing the MPM trajectory
        material_params (torch.Tensor): Material parameters for particle groups (frictions, cohesions) with shape (n_groups, 2)        device (torch.device): Device to move tensors to

    Raises:
        NotImplementedError: If the option for preparing initial features (X0) is not implemented
        ValueError: If the option for preparing initial features (X0) is invalid

    Returns:
        tuple: Initial features (X0)
    """
    # preprocess material parameters
    rigid_material_params = torch.tensor([1.0, 1.0], device=device).unsqueeze(0)
    material_params = torch.cat([rigid_material_params.detach(), material_params])    
    
    if x0_option == "from_same_5vels":
        mpm_trajectory = particle_data_loader.load_data(npz_path)
    elif x0_option == "from_mpm":
        raise NotImplementedError

        # run_mpm(
        #     mpm_path,
        #     mpm_json_path
        # )
        
        # # Make `.npz` to prepare initial state X_1 for rollout
        # convert_hd5_to_npz(
        #     h5_path,
        #     ndim,
        #     output_path,
        #     material_feature=True)
        
        # mpm_trajectory = particle_data_loader.load_data(output_path)
            
    else:
        raise ValueError(f"Invalid option: {x0_option}")
    
    # # Get particle positions from particle group txt file.
    # particle_files = sorted(glob.glob(f"{path}/particles*.txt"))
    # particle_groups = []
    # particle_group_idx_ranges = []
    # count = 0
    # for filename in particle_files:
    #     particle_group = torch.tensor(np.loadtxt(filename, skiprows=1))
    #     particle_groups.append(particle_group)
    #     particle_group_idx_range = np.arange(count, count+len(particle_group))
    #     count = count+len(particle_group)
    #     particle_group_idx_ranges.append(particle_group_idx_range)
    # initial_position = torch.concat(particle_groups).to(device)
    
    # Prepare initial features (X0) for rollout on torch device
    if len(mpm_trajectory[0][0]) < 6:
        # Compute velocity using the first two position sequences
        velocity = (mpm_trajectory[0][0][1] - mpm_trajectory[0][0][0]) / cfg.gns_config.physical_dt
        
        # Generate additional position sequences using the computed velocity
        additional_positions = [
            mpm_trajectory[0][0][0] + (i + 1) * velocity for i in range(4)
        ]
        
        # Concatenate the original and generated position sequences
        positions = torch.tensor(
            np.concatenate((mpm_trajectory[0][0], additional_positions), axis=0),
            device=device,
            dtype=torch.float32
        )
    else:
        positions = torch.tensor(
            mpm_trajectory[0][0][:6], device=device, dtype=torch.float32
        )
    
    initial_positions = positions.permute(1, 0, 2).contiguous()
    particle_types = torch.tensor(
        mpm_trajectory[0][1], device=device, dtype=torch.long
        ).contiguous()

    # If mpm is provided, build material_properties from mpm.particle_groups
    if mpm is not None and hasattr(mpm, "particle_groups"):
        n_particles = mpm_trajectory[0][1].shape[0]
        max_cohesion = cfg.max_cohesion
        default_cohesion = cfg.devault_cohesion
        # two columns: [tan(friction), cohesion_normalized]
        material_feature = np.zeros((n_particles, 2), dtype=np.float32)

        for group_id, group in mpm.particle_groups.items():
            ids = np.asarray(group.get("id", []), dtype=int)
            if ids.size == 0:
                continue

            if int(group_id) == 0:
                phi_deg = np.full(ids.size, LINEAR_ELASTIC_FEATURE_DEG, dtype=np.float32)
                coh_raw = np.full(ids.size, max_cohesion, dtype=np.float32)
            else:
                phi_deg = np.asarray(group.get("field_values", []), dtype=np.float32)
                if phi_deg.shape[0] != ids.shape[0]:
                    n = min(phi_deg.shape[0], ids.shape[0])
                    ids = ids[:n]
                    phi_deg = phi_deg[:n]
                coh_raw = np.full(ids.size, default_cohesion, dtype=np.float32)

            # Shift to 0-based if needed and mask to bounds
            if ids.max() >= n_particles:
                ids = ids - 1
            mask = (ids >= 0) & (ids < n_particles)
            ids = ids[mask]
            phi_deg = phi_deg[mask]
            coh_raw = coh_raw[mask]

            material_feature[ids, 0] = np.tan(np.deg2rad(phi_deg))
            material_feature[ids, 1] = coh_raw / max_cohesion

        material_properties = torch.tensor(
            material_feature, device=device, dtype=torch.float32
        ).contiguous()
    else:
        material_properties = torch.tensor(
            mpm_trajectory[0][2], device=device, dtype=torch.float32
            ).contiguous()

    # Get number of particles per example (we only consider one example)
    n_particles_per_example = torch.tensor(
        [mpm_trajectory[0][1].shape[0]],
        ).to(device)
    
    x0 = (
        initial_positions,
        particle_types,
        material_properties,
        n_particles_per_example
    )
    
    # Return initial features (X0)
    return x0