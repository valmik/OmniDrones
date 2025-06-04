import torch
from typing import List
class Constant():
    def __init__(self, pose: List[float], origin: torch.Tensor = torch.zeros(3), device: str = 'cpu'):
        self.device = device
        self.origin = origin
        self.pose = pose
    
    def reset(self, idx: torch.Tensor = None):
        pass
        
    def pos(self, t):

        x = torch.ones_like(t) * self.pose[0]
        y = torch.ones_like(t) * self.pose[1]
        z = torch.ones_like(t) * self.pose[2]

        pos = torch.stack([
            x, y, z
        ], dim=-1)
        
        return (pos + self.origin).to(self.device)