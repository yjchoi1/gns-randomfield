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
        runout_gns = self.gns_data[0][0][-1, :, 0].max()
        if self.mpm_data is not None:
            runout_mpm = self.mpm_data[0][0][-1, :, 0].max()
        else:
            runout_mpm = None
        
        return runout_gns, runout_mpm
    
    def compute_mse(self):
        if self.mpm_data is not None:
            mse = np.mean((self.gns_data[0][0] - self.mpm_data[0][0]) ** 2)
            return mse
        else:
            return None