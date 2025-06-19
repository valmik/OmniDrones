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

    def batch_pos(self, t: torch.Tensor):
        """
        Compute positions for batched time inputs.
        Args:
            t: torch.Tensor of shape [num_trajs, num_time_points]
        Returns:
            torch.Tensor of shape [num_trajs, num_time_points, 3]
        """
        assert t.ndim == 2, "t must be of shape [num_trajs, num_time_points]"
        pos = torch.tensor(self.pose, device=self.device).expand(t.shape[0], t.shape[1], 3)
        # Ensure origin has correct shape for broadcasting
        origin = self.origin.view(1, 1, -1).expand(t.shape[0], t.shape[1], -1)
        return (pos + origin).to(self.device)