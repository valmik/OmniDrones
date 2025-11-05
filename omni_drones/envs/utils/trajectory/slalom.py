import torch

class Slalom():
    def __init__(self,
                 T: float = 10.0,
                 origin: torch.Tensor = torch.zeros(3),
                 dist_x: float = 10.0,
                 ay: float = 2.0,
                 fy: float = 4.0,
                 az: float = 1.0,
                 fz: float = 1.0,
                 device: str = 'cpu'
                 ):
        self.T = T
        self.device = device
        self.origin = origin
        self.dist_x = dist_x
        self.ay = ay
        self.fy = fy
        self.az = az
        self.fz = fz

    def reset(self, idx: torch.Tensor = None):
        pass

    def pos(self, t: torch.Tensor):

        freq_y = 2 * torch.pi * self.fy / self.T
        freq_z = 2 * torch.pi * self.fz / self.T

        vx = self.dist_x / self.T
        x = vx * t
        y = self.ay * torch.sin(freq_y * t)
        z = self.az * torch.sin(freq_z * t)

        pos = torch.stack([x, y, z], dim=-1)
        return (pos + self.origin).to(self.device)

    def batch_pos(self, t: torch.Tensor):
        """
        Compute positions for batched time inputs.
        Args:
            t: torch.Tensor of shape [num_trajs, num_time_points]
        Returns:
            torch.Tensor of shape [num_trajs, num_time_points, 3]
        """
        assert t.ndim == 2, "t must be of shape [num_trajs, num_time_points]"
        
        freq_y = 2 * torch.pi * self.fy / self.T
        freq_z = 2 * torch.pi * self.fz / self.T

        vx = self.dist_x / self.T
        x = vx * t
        y = self.ay * torch.sin(freq_y * t)
        z = self.az * torch.sin(freq_z * t)

        pos = torch.stack([x, y, z], dim=-1)
        origin = self.origin.view(1, 1, -1).expand(t.shape[0], t.shape[1], -1)
        return (pos + origin).to(self.device)

    def vel(self, t: torch.Tensor):
        freq_y = 2 * torch.pi * self.fy / self.T
        freq_z = 2 * torch.pi * self.fz / self.T

        vx = self.dist_x / self.T
        dx = vx * torch.ones_like(t)
        dy = self.ay * freq_y * torch.cos(freq_y * t)
        dz = self.az * freq_z * torch.cos(freq_z * t)

        vel = torch.stack([dx, dy, dz], dim=-1)
        return vel.to(self.device)
    
    def acc(self, t: torch.Tensor): 
        freq_y = 2 * torch.pi * self.fy / self.T
        freq_z = 2 * torch.pi * self.fz / self.T

        ddx = torch.zeros_like(t)
        ddy = -self.ay * freq_y**2 * torch.sin(freq_y * t)
        ddz = -self.az * freq_z**2 * torch.sin(freq_z * t)

        acc = torch.stack([ddx, ddy, ddz], dim=-1)
        return acc.to(self.device)

    def jerk(self, t: torch.Tensor):
        freq_y = 2 * torch.pi * self.fy / self.T
        freq_z = 2 * torch.pi * self.fz / self.T

        d3x = torch.zeros_like(t)
        d3y = -self.ay * freq_y**3 * torch.cos(freq_y * t)
        d3z = -self.az * freq_z**3 * torch.cos(freq_z * t)

        jerk = torch.stack([d3x, d3y, d3z], dim=-1)
        return jerk.to(self.device)
        
    def snap(self, t: torch.Tensor):
        freq_y = 2 * torch.pi * self.fy / self.T
        freq_z = 2 * torch.pi * self.fz / self.T

        d4x = torch.zeros_like(t)
        d4y = self.ay * freq_y**4 * torch.sin(freq_y * t)
        d4z = self.az * freq_z**4 * torch.sin(freq_z * t)

        snap = torch.stack([d4x, d4y, d4z], dim=-1)
        return snap.to(self.device)