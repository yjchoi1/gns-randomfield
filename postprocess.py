from matplotlib import pyplot as plt
from gns.gns import particle_data_loader
import numpy as np

class PostProcesser:
    def __init__(self, gns_trajectory_path, mpm_trajectory_path=None):
        self.gns_trajectory_path = gns_trajectory_path
        self.gns_data = particle_data_loader.load_data(self.gns_trajectory_path)
        # Read MPM data if exists
        if mpm_trajectory_path is not None:
            self.mpm_trajectory_path = mpm_trajectory_path
            self.mpm_data = particle_data_loader.load_data(self.mpm_trajectory_path)
        else:
            self.mpm_data = None

    def compute_runout(self):        
        # GNS runout
        positions_gns = self.gns_data[0][0]
        particle_types_gns = self.gns_data[0][1]
        # Get positions at final timestep before masking
        final_positions_gns = positions_gns[-1]  # shape: (n_particles, n_dims)
        kinematic_particle_positions_gns = final_positions_gns[particle_types_gns == 6]
        runout_gns = kinematic_particle_positions_gns[:, 0].max()
        
        # MPM runout
        if self.mpm_data is not None:
            positions_mpm = self.mpm_data[0][0]
            particle_types_mpm = self.mpm_data[0][1]
            final_positions_mpm = positions_mpm[-1]  # shape: (n_particles, n_dims)
            kinematic_particle_positions_mpm = final_positions_mpm[particle_types_mpm == 6]
            runout_mpm = kinematic_particle_positions_mpm[:, 0].max()
        else:
            runout_mpm = None
        
        return runout_gns, runout_mpm
    
    def compute_mse(self):
        if self.mpm_data is not None:
            mse = np.mean((self.gns_data[0][0] - self.mpm_data[0][0]) ** 2)
            return mse
        else:
            return None

    def plot_deposit(self, domain, save_dir=None):
        """
        Plot the final deposit state of particles for both GNS and MPM (if available).
        Returns the figure and axes objects.
        """
        # Create figure with appropriate subplots
        if self.mpm_data is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 5))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

        # Get final timestep particle positions
        gns_final_positions = self.gns_data[0][0][-1, :, :]  # shape: (n_particles, n_dims)

        # Plot GNS results
        ax1.scatter(gns_final_positions[:, 0], gns_final_positions[:, 1],
                    s=10, alpha=0.6, label='GNS')
        ax1.set_title('GNS Final Deposit')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot MPM results if available
        if self.mpm_data is not None:
            mpm_final_positions = self.mpm_data[0][0][-1, :, :]
            ax2.scatter(mpm_final_positions[:, 0], mpm_final_positions[:, 1],
                        s=10, alpha=0.6, label='MPM')
            ax2.set_title('MPM Final Deposit')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.grid(True, alpha=0.3)
            ax2.legend()

        # Set domain limits if provided
        if domain is not None:
            ax1.set_xlim(domain[0][0], domain[0][1])
            ax1.set_ylim(domain[1][0], domain[1][1])
            if self.mpm_data is not None:
                ax2.set_xlim(domain[0][0], domain[0][1])
                ax2.set_ylim(domain[1][0], domain[1][1])

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.savefig(f"{save_dir}/final_deposit.png")
