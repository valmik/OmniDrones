import torch

class Lissajous():
    def __init__(self, 
                 T: float = 15.0,
                 origin: torch.Tensor = torch.zeros(3),
                 ax: float = 1.0,
                 ay: float = 1.0,
                 az: float = 1.0,
                 fx: float = 1.0,
                 fy: float = 1.0,
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

        self.z_offset = 1 + 2* torch.abs(self.az)
        
    def reset(self, idx: torch.Tensor = None):
        pass
        
    def pos(self, t):
        x = self.ax * torch.sin(self.fx * t / self.T + self.del_x)
        y = self.ay * torch.sin(self.fy * t / self.T + self.del_y)
        z = self.z_offset + self.az * torch.sin(self.fz * t / self.T + self.del_z)
        pos = torch.stack([x, y, z], dim=-1)
        return (pos + self.origin).to(self.device)