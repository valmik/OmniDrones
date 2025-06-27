import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

class RandomLissajous:
    def __init__(self, 
                 num_trajs: int,
                 T: float = 2.0,
                 origin: torch.Tensor = torch.zeros(3),
                 # Amplitude ranges
                 ax_range: Tuple[float, float] = (1.0, 3.0),
                 ay_range: Tuple[float, float] = (1.0, 3.0),
                 az_range: Tuple[float, float] = (1.0, 3.0),
                 # Frequency ranges
                 fx_range: Tuple[float, float] = (0.5, 2.0),
                 fy_range: Tuple[float, float] = (0.5, 2.0),
                 fz_range: Tuple[float, float] = (0.5, 2.0),
                 # Phase delay ranges (in radians)
                 del_x_range: Tuple[float, float] = (0.0, 2*np.pi),
                 del_y_range: Tuple[float, float] = (0.0, 2*np.pi),
                 del_z_range: Tuple[float, float] = (0.0, 2*np.pi),
                 device: str = 'cpu'
                 ):
        self.num_trajs = num_trajs
        self.T = T
        self.device = device
        self.origin = origin.to(device) if isinstance(origin, torch.Tensor) else torch.tensor(origin, device=device)
        
        self.ax_range = ax_range
        self.ay_range = ay_range
        self.az_range = az_range
        self.fx_range = fx_range
        self.fy_range = fy_range
        self.fz_range = fz_range
        self.del_x_range = del_x_range
        self.del_y_range = del_y_range
        self.del_z_range = del_z_range
        
        self.ax = torch.zeros(num_trajs, device=device)
        self.ay = torch.zeros(num_trajs, device=device)
        self.az = torch.zeros(num_trajs, device=device)
        self.fx = torch.zeros(num_trajs, device=device)
        self.fy = torch.zeros(num_trajs, device=device)
        self.fz = torch.zeros(num_trajs, device=device)
        self.del_x = torch.zeros(num_trajs, device=device)
        self.del_y = torch.zeros(num_trajs, device=device)
        self.del_z = torch.zeros(num_trajs, device=device)
        
        self.reset()
        
    def _sample_uniform(self, range_tuple: Tuple[float, float], size: int) -> torch.Tensor:
        """Sample uniformly from a range."""
        low, high = range_tuple
        return torch.rand(size, device=self.device) * (high - low) + low
        
    def reset(self, idx: Optional[torch.Tensor] = None):
        """Reset trajectory parameters. If idx is None, reset all trajectories."""
        if idx is None:
            idx = torch.arange(self.num_trajs, device=self.device)
        
        num_reset = len(idx)
        
        # Sample new parameters
        self.ax[idx] = self._sample_uniform(self.ax_range, num_reset)
        self.ay[idx] = self._sample_uniform(self.ay_range, num_reset)
        self.az[idx] = self._sample_uniform(self.az_range, num_reset)
        
        self.fx[idx] = self._sample_uniform(self.fx_range, num_reset)
        self.fy[idx] = self._sample_uniform(self.fy_range, num_reset)
        self.fz[idx] = self._sample_uniform(self.fz_range, num_reset)
        
        self.del_x[idx] = self._sample_uniform(self.del_x_range, num_reset)
        self.del_y[idx] = self._sample_uniform(self.del_y_range, num_reset)
        self.del_z[idx] = self._sample_uniform(self.del_z_range, num_reset)
        
    def pos(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute position for single time point across all trajectories.
        Args:
            t: torch.Tensor of shape [num_trajs] - time for each trajectory
        Returns:
            torch.Tensor of shape [num_trajs, 3] - positions
        """
        assert t.shape == (self.num_trajs,), f"Expected t.shape to be ({self.num_trajs},), got {t.shape}"
        
        x = self.ax * torch.sin(self.fx * t / self.T + self.del_x)
        y = self.ay * torch.sin(self.fy * t / self.T + self.del_y)
        z = self.az * torch.sin(self.fz * t / self.T + self.del_z)

        pos = torch.stack([x, y, z], dim=-1)  # [num_trajs, 3]
        return (pos + self.origin).to(self.device)

    def batch_pos(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute positions for batched time inputs.
        Args:
            t: torch.Tensor of shape [num_trajs, num_time_points]
        Returns:
            torch.Tensor of shape [num_trajs, num_time_points, 3]
        """
        assert t.ndim == 2 and t.shape[0] == self.num_trajs, f"Expected t.shape to be ({self.num_trajs}, num_time_points), got {t.shape}"
        
        # Expand parameters to match time dimensions
        ax = self.ax.unsqueeze(1)  # [num_trajs, 1]
        ay = self.ay.unsqueeze(1)
        az = self.az.unsqueeze(1)
        fx = self.fx.unsqueeze(1)
        fy = self.fy.unsqueeze(1)
        fz = self.fz.unsqueeze(1)
        del_x = self.del_x.unsqueeze(1)
        del_y = self.del_y.unsqueeze(1)
        del_z = self.del_z.unsqueeze(1)
        
        x = ax * torch.sin(fx * t / self.T + del_x)
        y = ay * torch.sin(fy * t / self.T + del_y)
        z = az * torch.sin(fz * t / self.T + del_z)

        pos = torch.stack([x, y, z], dim=-1)  # [num_trajs, num_time_points, 3]
        origin = self.origin.view(1, 1, -1).expand(t.shape[0], t.shape[1], -1)
        return (pos + origin).to(self.device)

    def vel(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity for single time point across all trajectories.
        Args:
            t: torch.Tensor of shape [num_trajs]
        Returns:
            torch.Tensor of shape [num_trajs, 3]
        """
        assert t.shape == (self.num_trajs,)
        
        # Derivatives of sin(ft + del) = f * cos(ft + del)
        vx = (self.ax * self.fx / self.T) * torch.cos(self.fx * t / self.T + self.del_x)
        vy = (self.ay * self.fy / self.T) * torch.cos(self.fy * t / self.T + self.del_y)
        vz = (self.az * self.fz / self.T) * torch.cos(self.fz * t / self.T + self.del_z)

        return torch.stack([vx, vy, vz], dim=-1)

    def acc(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute acceleration for single time point across all trajectories.
        Args:
            t: torch.Tensor of shape [num_trajs]
        Returns:
            torch.Tensor of shape [num_trajs, 3]
        """
        assert t.shape == (self.num_trajs,)
        
        # Second derivatives of sin(ft + del) = -f^2 * sin(ft + del)
        ax = -(self.ax * (self.fx / self.T)**2) * torch.sin(self.fx * t / self.T + self.del_x)
        ay = -(self.ay * (self.fy / self.T)**2) * torch.sin(self.fy * t / self.T + self.del_y)
        az = -(self.az * (self.fz / self.T)**2) * torch.sin(self.fz * t / self.T + self.del_z)

        return torch.stack([ax, ay, az], dim=-1)

    def jerk(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute jerk for single time point across all trajectories.
        Args:
            t: torch.Tensor of shape [num_trajs]
        Returns:
            torch.Tensor of shape [num_trajs, 3]
        """
        assert t.shape == (self.num_trajs,)
        
        # Third derivatives of sin(ft + del) = -f^3 * cos(ft + del)
        jx = -(self.ax * (self.fx / self.T)**3) * torch.cos(self.fx * t / self.T + self.del_x)
        jy = -(self.ay * (self.fy / self.T)**3) * torch.cos(self.fy * t / self.T + self.del_y)
        jz = -(self.az * (self.fz / self.T)**3) * torch.cos(self.fz * t / self.T + self.del_z)

        return torch.stack([jx, jy, jz], dim=-1)

    def snap(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute snap (fourth derivative) for single time point across all trajectories.
        Args:
            t: torch.Tensor of shape [num_trajs]
        Returns:
            torch.Tensor of shape [num_trajs, 3]
        """
        assert t.shape == (self.num_trajs,)
        
        # Fourth derivatives of sin(ft + del) = f^4 * sin(ft + del)
        sx = (self.ax * (self.fx / self.T)**4) * torch.sin(self.fx * t / self.T + self.del_x)
        sy = (self.ay * (self.fy / self.T)**4) * torch.sin(self.fy * t / self.T + self.del_y)
        sz = (self.az * (self.fz / self.T)**4) * torch.sin(self.fz * t / self.T + self.del_z)

        return torch.stack([sx, sy, sz], dim=-1)

    def get_parameters(self, idx: Optional[torch.Tensor] = None) -> dict:
        """
        Get trajectory parameters for specified indices.
        Args:
            idx: indices to get parameters for. If None, returns all.
        Returns:
            dict containing parameter arrays
        """
        if idx is None:
            idx = torch.arange(self.num_trajs, device=self.device)
            
        return {
            'ax': self.ax[idx].cpu().numpy(),
            'ay': self.ay[idx].cpu().numpy(),
            'az': self.az[idx].cpu().numpy(),
            'fx': self.fx[idx].cpu().numpy(),
            'fy': self.fy[idx].cpu().numpy(),
            'fz': self.fz[idx].cpu().numpy(),
            'del_x': self.del_x[idx].cpu().numpy(),
            'del_y': self.del_y[idx].cpu().numpy(),
            'del_z': self.del_z[idx].cpu().numpy(),
        }


if __name__ == "__main__":
    import time
    
    # Test the RandomLissajous class
    num_trajs = 5
    device = 'cpu'
    
    # Create random Lissajous trajectory generator
    random_liss = RandomLissajous(
        num_trajs=num_trajs,
        T=10.0,
        origin=torch.zeros(3),
        ax_range=(0.5, 3.0),
        ay_range=(0.5, 3.0), 
        az_range=(0.5, 3.0),
        fx_range=(0.3, 2.0),
        fy_range=(0.3, 2.0),
        fz_range=(0.3, 2.0),
        del_x_range=(0.0, 2*np.pi),
        del_y_range=(0.0, 2*np.pi),
        del_z_range=(0.0, 2*np.pi),
        device=device
    )
    
    # Print initial parameters
    params = random_liss.get_parameters()
    print("Random Lissajous Parameters:")
    for key, val in params.items():
        print(f"{key}: {val}")
    
    # Generate time points
    t = torch.stack([torch.linspace(0, 20, 1000) for _ in range(num_trajs)], dim=0)
    
    # Get positions using batch method
    positions = random_liss.batch_pos(t)
    
    # Test single time point methods
    t_single = torch.ones(num_trajs) * 5.0  # t=5 for all trajectories
    pos_single = random_liss.pos(t_single)
    vel_single = random_liss.vel(t_single)
    acc_single = random_liss.acc(t_single)
    jerk_single = random_liss.jerk(t_single)
    snap_single = random_liss.snap(t_single)
    
    print(f"\nAt t=5.0:")
    print(f"Position shape: {pos_single.shape}")
    print(f"Velocity shape: {vel_single.shape}")
    print(f"Acceleration shape: {acc_single.shape}")
    print(f"Jerk shape: {jerk_single.shape}")
    print(f"Snap shape: {snap_single.shape}")
    
    # Plot some trajectories
    fig = plt.figure(figsize=(15, 10))
    
    # 3D trajectories
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    for i in range(min(3, num_trajs)):
        ax1.plot(positions[i, :, 0], positions[i, :, 1], positions[i, :, 2], 
                label=f'Traj {i}', alpha=0.7)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Random Lissajous Trajectories')
    ax1.legend()
    
    # X-Y projections
    ax2 = fig.add_subplot(2, 3, 2)
    for i in range(min(3, num_trajs)):
        ax2.plot(positions[i, :, 0], positions[i, :, 1], 
                label=f'Traj {i}', alpha=0.7)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('X-Y Projection')
    ax2.legend()
    ax2.grid(True)
    
    # X-Z projections
    ax3 = fig.add_subplot(2, 3, 3)
    for i in range(min(3, num_trajs)):
        ax3.plot(positions[i, :, 0], positions[i, :, 2], 
                label=f'Traj {i}', alpha=0.7)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('X-Z Projection')
    ax3.legend()
    ax3.grid(True)
    
    # Y-Z projections
    ax4 = fig.add_subplot(2, 3, 4)
    for i in range(min(3, num_trajs)):
        ax4.plot(positions[i, :, 1], positions[i, :, 2], 
                label=f'Traj {i}', alpha=0.7)
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.set_title('Y-Z Projection')
    ax4.legend()
    ax4.grid(True)
    
    # Time series for one trajectory
    ax5 = fig.add_subplot(2, 3, 5)
    traj_idx = 0
    ax5.plot(t[traj_idx], positions[traj_idx, :, 0], label='X', alpha=0.7)
    ax5.plot(t[traj_idx], positions[traj_idx, :, 1], label='Y', alpha=0.7)
    ax5.plot(t[traj_idx], positions[traj_idx, :, 2], label='Z', alpha=0.7)
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Position')
    ax5.set_title(f'Time Series - Trajectory {traj_idx}')
    ax5.legend()
    ax5.grid(True)
    
    # Parameter histogram
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.hist(params['fx'], alpha=0.5, label='fx', bins=10)
    ax6.hist(params['fy'], alpha=0.5, label='fy', bins=10)
    ax6.hist(params['fz'], alpha=0.5, label='fz', bins=10)
    ax6.set_xlabel('Frequency')
    ax6.set_ylabel('Count')
    ax6.set_title('Frequency Distribution')
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    datetime = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    plt.savefig(f'random_lissajous-{datetime}.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as: random_lissajous-{datetime}.png")
    
    # Test reset functionality
    print(f"\nTesting reset functionality...")
    old_params = random_liss.get_parameters(torch.tensor([0]))
    random_liss.reset(torch.tensor([0]))  # Reset only trajectory 0
    new_params = random_liss.get_parameters(torch.tensor([0]))
    
    print(f"Old parameters for trajectory 0: ax={old_params['ax'][0]:.3f}, fx={old_params['fx'][0]:.3f}")
    print(f"New parameters for trajectory 0: ax={new_params['ax'][0]:.3f}, fx={new_params['fx'][0]:.3f}")
    
    plt.show()