import torch

class Lissajous():
    def __init__(self, 
                 T: float = 2.0,
                 origin: torch.Tensor = torch.zeros(3),
                 ax: float = 3.0,
                 ay: float = 1.0,
                 az: float = 2.0,
                 fx: float = 1.0,
                 fy: float = 0.5,
                 fz: float = 1.0,
                 del_x: float = 0.0,
                 del_y: float = 0.0,
                 del_z: float = 0.0,
                 device: str = 'cpu'
                 ):
        self.T = T
        self.device = device
        self.origin = origin
        self.ax = ax
        self.ay = ay
        self.az = az
        self.fx = fx
        self.fy = fy
        self.fz = fz
        self.del_x = del_x
        self.del_y = del_y
        self.del_z = del_z
        
    def reset(self, idx: torch.Tensor = None):
        pass
        
    def pos(self, t):
        x = self.ax * torch.sin(self.fx * t / self.T + self.del_x)
        y = self.ay * torch.sin(self.fy * t / self.T + self.del_y)
        z = self.az * torch.sin(self.fz * t / self.T + self.del_z)

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
        x = self.ax * torch.sin(self.fx * t / self.T + self.del_x)
        y = self.ay * torch.sin(self.fy * t / self.T + self.del_y)
        z = self.az * torch.sin(self.fz * t / self.T + self.del_z)

        pos = torch.stack([x, y, z], dim=-1)
        # Ensure origin has correct shape for broadcasting
        origin = self.origin.view(1, 1, -1).expand(t.shape[0], t.shape[1], -1)
        return (pos + origin).to(self.device)

    def vel(self, t: torch.Tensor):

        dx = self.ax * self.fx / self.T * torch.cos(self.fx * t / self.T + self.del_x)
        dy = self.ay * self.fy / self.T * torch.cos(self.fy * t / self.T + self.del_y)
        dz = self.az * self.fz / self.T * torch.cos(self.fz * t / self.T + self.del_z)

        vel = torch.stack([dx, dy, dz], dim=-1)
        return vel.to(self.device)

    def acc(self, t: torch.Tensor):
        ddx = -self.ax * self.fx**2 / self.T**2 * torch.sin(self.fx * t / self.T + self.del_x)
        ddy = -self.ay * self.fy**2 / self.T**2 * torch.sin(self.fy * t / self.T + self.del_y)
        ddz = -self.az * self.fz**2 / self.T**2 * torch.sin(self.fz * t / self.T + self.del_z)

        acc = torch.stack([ddx, ddy, ddz], dim=-1)
        return acc.to(self.device)

    def jerk(self, t: torch.Tensor):
        d3x = -self.ax * self.fx**3 / self.T**3 * torch.cos(self.fx * t / self.T + self.del_x)
        d3y = -self.ay * self.fy**3 / self.T**3 * torch.cos(self.fy * t / self.T + self.del_y)
        d3z = -self.az * self.fz**3 / self.T**3 * torch.cos(self.fz * t / self.T + self.del_z)

        jerk = torch.stack([d3x, d3y, d3z], dim=-1)
        return jerk.to(self.device)

    def snap(self, t: torch.Tensor):
        d4x = self.ax * self.fx**4 / self.T**4 * torch.sin(self.fx * t / self.T + self.del_x)
        d4y = self.ay * self.fy**4 / self.T**4 * torch.sin(self.fy * t / self.T + self.del_y)
        d4z = self.az * self.fz**4 / self.T**4 * torch.sin(self.fz * t / self.T + self.del_z)

        snap = torch.stack([d4x, d4y, d4z], dim=-1)
        return snap.to(self.device)