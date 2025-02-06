from gns.gns import reading_utils
from gns.gns import train
import torch
from tqdm import tqdm
import os
from typing import List
import numpy as np


@torch.no_grad()
def forward(
    x0: torch.Tensor, 
    cfg, 
    device: torch.device, 
    gradient_checkpoint_interval: int = 1,
    is_gradient_checkpointing: bool = False
):
    # Load simulator
    simulator, metadata = load_simulator(cfg, device)
    simulator.to(device)
    
    # Unpack initial features (X0)
    initial_positions, particle_types, material_property, n_particles_per_example = x0

    current_positions = initial_positions
    predictions: List[torch.Tensor] = []

    def predict_step(use_checkpoint: bool = False):
        if use_checkpoint:
            next_position = torch.utils.checkpoint.checkpoint(
                simulator.predict_positions,
                current_positions,
                [n_particles_per_example],
                particle_types,
                material_property
            )
        else:
            next_position = simulator.predict_positions(
                current_positions,
                nparticles_per_example=[n_particles_per_example],
                particle_types=particle_types,
                material_property=material_property
            )
        return next_position

    for step in tqdm(range(cfg.gns_config.nsteps), desc="Simulation progress"):
        next_position = predict_step(
            use_checkpoint=is_gradient_checkpointing and step % gradient_checkpoint_interval == 0
        )

        # Fix the positions of rigid particles
        next_position[particle_types == cfg.gns_config.stationary_particle] = current_positions[:, -1][
            particle_types == cfg.gns_config.stationary_particle]

        # Store predictions
        predictions.append(next_position)

        # Update positions for next step:
        # Shift `current_positions`, removing the oldest position in the sequence
        # and appending the next position at the end.
        current_positions = torch.cat(
            [current_positions[:, 1:], next_position[:, None, :]], dim=1
        )
        
    rollout = torch.cat(
        (initial_positions.permute(1, 0, 2), torch.stack(predictions))
    )
    
    result_dict = {
        "rollout": rollout.cpu().numpy(),
        "material_property": material_property.cpu().numpy(),
        "particle_types": particle_types.cpu().numpy(),
    }

    return result_dict


def load_simulator(cfg, device):
    metadata = reading_utils.read_metadata(
        cfg.gns_config.data_path, "rollout", cfg.gns_config.meta_data
    )
    simulator = train._get_simulator(
        metadata,
        cfg.gns_config.num_particle_types,
        cfg.gns_config.noise_std,
        cfg.gns_config.noise_std,
        device,
    )

    if os.path.exists(cfg.gns_config.model_path + cfg.gns_config.model_file):
        simulator.load(cfg.gns_config.model_path + cfg.gns_config.model_file)
    else:
        raise Exception(f"Model does not exist at {cfg.gns_config.model_path + cfg.gns_config.model_file}")

    return simulator, metadata


def save_results(result_dict,save_path):
    # Make npz
    trajectories = {}
    
    trajectories[save_path] = (
        result_dict["rollout"].astype("float32"),  # position sequence (timesteps, particles, dims)
        result_dict["particle_types"].astype("int32"),  # particle type (particles, )
        result_dict["material_property"].astype("float32"))  # particle type (particles, n_particle_features)

    # Create structured array to hold the data
    structured_data = np.empty(len(trajectories), dtype=object)
    for i, value in enumerate(trajectories.values()):
        structured_data[i] = value

    np.savez_compressed(save_path, gns_data=structured_data)
    
    

    