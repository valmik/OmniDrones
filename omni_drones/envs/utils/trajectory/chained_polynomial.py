from typing import List, Union

import torch
try:
    from .base import BaseTrajectory
except:
    from omni_drones.envs.utils.trajectory.base import BaseTrajectory
try:
    from ...utils import math_utils as mu 
except:
    import omni_drones.envs.utils.math_utils as mu


class ChainedPolynomial(BaseTrajectory):
    def __init__(self,
                 num_trajs: int,
                 scale: float = 1.5,
                 use_y: bool = True,
                 use_z: bool = True,
                 min_dt: float = 1.5,
                 max_dt: float = 4.0,
                 degree: int = 5,
                 origin: torch.Tensor = torch.zeros(3),
                 device: str = 'cpu'):
        super().__init__(num_trajs, origin, device)
        assert degree % 2 == 1

        self.use_y = use_y
        self.use_z = use_z
        self.degree = degree
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.scale = scale

        self.num_segments = 100

        self.x_coeffs = torch.zeros(self.num_trajs, self.degree + 1, self.num_segments, 
                                    dtype=torch.float32, device=self.device)
        self.T_x = torch.zeros(self.num_trajs, self.num_segments + 1, 
                               dtype=torch.float32, device=self.device)
        if self.use_y:
            self.y_coeffs = torch.zeros(self.num_trajs, self.degree + 1, self.num_segments, 
                                        dtype=torch.float32, device=self.device)
            self.T_y = torch.zeros(self.num_trajs, self.num_segments + 1, 
                                   dtype=torch.float32, device=self.device)
        if self.use_z:
            self.z_coeffs = torch.zeros(self.num_trajs, self.degree + 1, self.num_segments, 
                                        dtype=torch.float32, device=self.device)
            self.T_z = torch.zeros(self.num_trajs, self.num_segments + 1, 
                                   dtype=torch.float32, device=self.device)
        self.reset()

    def generate_coeffs(self, dt: torch.Tensor):
        b_values = torch.rand(self.num_trajs, (self.degree + 1) // 2, self.num_segments + 1, 
                              dtype=torch.float32, device=self.device) * self.scale * 2 - self.scale # (num_trajs, (degree + 1) // 2, num_segments + 1)
        b_values[:, 0] = 0.
        b_values = torch.cat([b_values[..., :-1], b_values[..., 1:]], dim=1) # (num_trajs, degree + 1, num_segments)

        A_values = torch.zeros(self.num_trajs, self.degree + 1, self.degree + 1, self.num_segments, dtype=torch.float32, device=self.device)
        coeffs = torch.zeros(self.num_trajs, self.degree + 1, self.num_segments, dtype=torch.float32, device=self.device)

        for i in range(self.num_segments):
            A_values[..., i] = self.deriv_fitting_matrix(dt[:, i])
            coeffs[..., i] = torch.linalg.solve(A_values[..., i], b_values[..., i])

        coeffs = coeffs.flip(dims=[-2])

        return coeffs

    def deriv_fitting_matrix(self, dt: torch.Tensor, degree: int = None):
        if degree is None:
            degree = self.degree + 1

        A = torch.zeros(self.num_trajs, degree, degree, device=self.device)

        ts = dt[:, None] ** torch.arange(degree, device=self.device) # (num_trajs, degree)
        constant_term = 1.
        poly = torch.ones(self.num_trajs, degree, device=self.device) # (num_trajs, degree)

        for i in range(degree // 2):
            A[:, i, i] = constant_term
            A[:, i + degree // 2, :] = torch.cat([
                torch.zeros(self.num_trajs, i, device=self.device),
                ts[:, :degree - i] * poly[:, :degree - i],
            ], dim=1)
            poly = mu.polyder(poly, increasing_order=True)
            constant_term *= i + 1

        return A

    def reset(self, 
              idx: torch.Tensor = None, 
              origin: torch.Tensor = None,
              verbose: bool = False):
        idx = super().reset(idx, origin, verbose)

        dt_x = torch.rand(self.num_trajs, self.num_segments, device=self.device
                          ) * (self.max_dt - self.min_dt) + self.min_dt # (num_trajs, num_segments)
        T_x = torch.cat([torch.cumsum(dt_x, dim=1),
                         torch.zeros(self.num_trajs, 1, device=self.device),
                         ], dim=1).contiguous()
        x_coeffs = self.generate_coeffs(dt_x) # (num_trajs, degree + 1, num_segments)
        self.x_coeffs[idx] = x_coeffs[idx]
        self.T_x[idx] = T_x[idx]

        if self.use_y:
            dt_y = torch.rand(self.num_trajs, self.num_segments, device=self.device
                              ) * (self.max_dt - self.min_dt) + self.min_dt
            T_y = torch.cat([torch.cumsum(dt_y, dim=1),
                             torch.zeros(self.num_trajs, 1, device=self.device),
                             ], dim=1).contiguous()
            y_coeffs = self.generate_coeffs(dt_y)
            self.y_coeffs[idx] = y_coeffs[idx]
            self.T_y[idx] = T_y[idx]

        if self.use_z:
            dt_z = torch.rand(self.num_trajs, self.num_segments, device=self.device
                              ) * (self.max_dt - self.min_dt) + self.min_dt
            T_z = torch.cat([torch.cumsum(dt_z, dim=1),
                             torch.zeros(self.num_trajs, 1, device=self.device),
                             ], dim=1).contiguous()
            z_coeffs = self.generate_coeffs(dt_z)
            self.z_coeffs[idx] = z_coeffs[idx]
            self.T_z[idx] = T_z[idx]

    def pos(self, t: torch.Tensor):
        assert t.shape == (self.num_trajs,)

        idx_x = torch.searchsorted(self.T_x, t[:, None]).squeeze(-1) # (num_trajs,)
        offset = self.T_x[torch.arange(self.num_trajs, device=self.device), idx_x - 1] # (num_trajs,)
        x_coeffs = self.x_coeffs[torch.arange(self.num_trajs, device=self.device), :, idx_x]
        x = mu.poly(x_coeffs, (t - offset)[:, None],)

        if self.use_y:
            idx_y = torch.searchsorted(self.T_y, t[:, None]).squeeze(-1)
            offset = self.T_y[torch.arange(self.num_trajs, device=self.device), idx_y - 1]
            y = mu.poly(self.y_coeffs[torch.arange(self.num_trajs, device=self.device), :, idx_y], 
                        (t - offset)[:, None],)
        else:
            y = x * 0.

        if self.use_z:
            idx_z = torch.searchsorted(self.T_z, t[:, None]).squeeze(-1)
            offset = self.T_z[torch.arange(self.num_trajs, device=self.device), idx_z - 1]
            z = mu.poly(self.z_coeffs[torch.arange(self.num_trajs, device=self.device), :, idx_z], 
                        (t - offset)[:, None],)
        else:
            z = x * 0.

        return torch.cat([x, y, z], dim=-1) + self.origin

    # implement the parallel version for trajectory generation
    def batch_pos(self, t: torch.Tensor):
        assert t.ndim == 2 and t.shape[0] == self.num_trajs, "t: [num_trajs, num_time_points]"

        batch_size, num_t = t.shape
        
        idx_x = torch.searchsorted(self.T_x, t).squeeze(-1)
        if idx_x.ndim == 1:
            idx_x = idx_x.unsqueeze(1)
        
        offset = self.T_x.gather(1, (idx_x - 1) % self.T_x.shape[1])

        # expand idx_x
        idx_x_expanded = idx_x.unsqueeze(1).expand(-1, self.x_coeffs.shape[1], -1)
        # self.x_coeffs: [num_traj, degree + 1, 100], 
        # idx_x_expanded: [num_traj, degree + 1, num_timepoints], 
        # x_coeffs: [num_traj, degree + 1, num_timepoints] + transpose
        x_coeffs = self.x_coeffs.gather(2, idx_x_expanded).transpose(1, 2)
        x = mu.poly(x_coeffs.reshape(-1, self.degree + 1), (t - offset).reshape(-1, 1),).reshape(self.num_trajs, -1, 1)

        if self.use_y:
            idx_y = torch.searchsorted(self.T_y, t).squeeze(-1)
            if idx_y.ndim == 1:
                idx_y = idx_y.unsqueeze(1)
            offset = self.T_y.gather(1, (idx_y - 1) % self.T_y.shape[1])
            # expand idx_y
            idx_y_expanded = idx_y.unsqueeze(1).expand(-1, self.y_coeffs.shape[1], -1)
            y_coeffs = self.y_coeffs.gather(2, idx_y_expanded).transpose(1, 2)
            y = mu.poly(y_coeffs.reshape(-1, self.degree + 1), (t - offset).reshape(-1, 1),).reshape(self.num_trajs, -1, 1)
        else:
            y = x * 0.

        if self.use_z:
            idx_z = torch.searchsorted(self.T_z, t).squeeze(-1)
            if idx_z.ndim == 1:
                idx_z = idx_z.unsqueeze(1)
            offset = self.T_z.gather(1, (idx_z - 1) % self.T_z.shape[1])
            z_coeffs = self.z_coeffs.gather(2, idx_z.unsqueeze(1).expand(-1, self.z_coeffs.shape[1], -1)).transpose(1, 2)
            z = mu.poly(z_coeffs.reshape(-1, self.degree + 1), (t - offset).reshape(-1, 1),).reshape(self.num_trajs, -1, 1)
        else:
            z = x * 0.

        return torch.cat([x, y, z], dim=-1) + self.origin # [num_traj, num_timepoints, 3]

    def vel(self, t: torch.Tensor):
        assert t.shape == (self.num_trajs,)

        idx_x = torch.searchsorted(self.T_x, t[:, None]).squeeze(-1)
        offset = self.T_x[torch.arange(self.num_trajs, device=self.device), idx_x - 1]
        x = mu.poly(mu.polyder(
            self.x_coeffs[torch.arange(self.num_trajs, device=self.device), :, idx_x]), 
            (t - offset)[:, None])

        if self.use_y:
            idx_y = torch.searchsorted(self.T_y, t[:, None]).squeeze(-1)
            offset = self.T_y[torch.arange(self.num_trajs, device=self.device), idx_y - 1]
            y = mu.poly(mu.polyder(
                self.y_coeffs[torch.arange(self.num_trajs, device=self.device), :, idx_y]), 
                (t - offset)[:, None])
        else:
            y = x * 0.

        if self.use_z:
            idx_z = torch.searchsorted(self.T_z, t[:, None]).squeeze(-1)
            offset = self.T_z[torch.arange(self.num_trajs, device=self.device), idx_z - 1]
            z = mu.poly(mu.polyder(
                self.z_coeffs[torch.arange(self.num_trajs, device=self.device), :, idx_z]), 
                (t - offset)[:, None])
        else:
            z = x * 0.

        return torch.cat([x, y, z], dim=-1)
    
    def acc(self, t: torch.Tensor):
        assert t.shape == (self.num_trajs,)

        idx_x = torch.searchsorted(self.T_x, t[:, None]).squeeze(-1)
        offset = self.T_x[torch.arange(self.num_trajs, device=self.device), idx_x - 1]
        x = mu.poly(mu.polyder(
            self.x_coeffs[torch.arange(self.num_trajs, device=self.device), :, idx_x], 2), 
            (t - offset)[:, None])

        if self.use_y:
            idx_y = torch.searchsorted(self.T_y, t[:, None]).squeeze(-1)
            offset = self.T_y[torch.arange(self.num_trajs, device=self.device), idx_y - 1]
            y = mu.poly(mu.polyder(
                self.y_coeffs[torch.arange(self.num_trajs, device=self.device), :, idx_y], 2), 
                (t - offset)[:, None])
        else:
            y = x * 0.

        if self.use_z:
            idx_z = torch.searchsorted(self.T_z, t[:, None]).squeeze(-1)
            offset = self.T_z[torch.arange(self.num_trajs, device=self.device), idx_z - 1]
            z = mu.poly(mu.polyder(
                self.z_coeffs[torch.arange(self.num_trajs, device=self.device), :, idx_z], 2), 
                (t - offset)[:, None])
        else:
            z = x * 0.  

        return torch.cat([x, y, z], dim=-1)

    def jerk(self, t: torch.Tensor):
        assert t.shape == (self.num_trajs,)

        idx_x = torch.searchsorted(self.T_x, t[:, None]).squeeze(-1)
        offset = self.T_x[torch.arange(self.num_trajs, device=self.device), idx_x - 1]
        x = mu.poly(mu.polyder(
            self.x_coeffs[torch.arange(self.num_trajs, device=self.device), :, idx_x], 3), 
            (t - offset)[:, None])

        if self.use_y:
            idx_y = torch.searchsorted(self.T_y, t[:, None]).squeeze(-1)
            offset = self.T_y[torch.arange(self.num_trajs, device=self.device), idx_y - 1]
            y = mu.poly(mu.polyder(
                self.y_coeffs[torch.arange(self.num_trajs, device=self.device), :, idx_y], 3), 
                (t - offset)[:, None])
        else:
            y = x * 0.

        if self.use_z:
            idx_z = torch.searchsorted(self.T_z, t[:, None]).squeeze(-1)
            offset = self.T_z[torch.arange(self.num_trajs, device=self.device), idx_z - 1]
            z = mu.poly(mu.polyder(
                self.z_coeffs[torch.arange(self.num_trajs, device=self.device), :, idx_z], 3), 
                (t - offset)[:, None])
        else:
            z = x * 0.

        return torch.cat([x, y, z], dim=-1)
    
    def snap(self, t: torch.Tensor):
        assert t.shape == (self.num_trajs,)

        idx_x = torch.searchsorted(self.T_x, t[:, None]).squeeze(-1)
        offset = self.T_x[torch.arange(self.num_trajs, device=self.device), idx_x - 1]
        x = mu.poly(mu.polyder(
            self.x_coeffs[torch.arange(self.num_trajs, device=self.device), :, idx_x], 4), 
            (t - offset)[:, None])

        if self.use_y:
            idx_y = torch.searchsorted(self.T_y, t[:, None]).squeeze(-1)
            offset = self.T_y[torch.arange(self.num_trajs, device=self.device), idx_y - 1]
            y = mu.poly(mu.polyder(
                self.y_coeffs[torch.arange(self.num_trajs, device=self.device), :, idx_y], 4), 
                (t - offset)[:, None])
        else:
            y = x * 0.

        if self.use_z:
            idx_z = torch.searchsorted(self.T_z, t[:, None]).squeeze(-1)
            offset = self.T_z[torch.arange(self.num_trajs, device=self.device), idx_z - 1]
            z = mu.poly(mu.polyder(
                self.z_coeffs[torch.arange(self.num_trajs, device=self.device), :, idx_z], 4), 
                (t - offset)[:, None])
        else:
            z = x * 0.

        return torch.cat([x, y, z], dim=-1)
    

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    
    # Create test trajectory
    num_traj = 2000
    ref = ChainedPolynomial(num_traj, scale=1.0, min_dt=1.5, max_dt=4.0, degree=5)

    # Generate time points
    t = torch.stack([torch.arange(0, 10, 0.004) for _ in range(num_traj)], dim=0)

    # Get positions
    pos = ref.batch_pos(t).cpu().numpy()  # [num_trajs, num_time_points, 3]

    # Get velocities, accelerations, jerks and snaps
    vel = []
    acc = []
    jerk = []
    snap = []
    for ti in range(t.shape[1]):
        vel.append(ref.vel(t[:, ti]))
        acc.append(ref.acc(t[:, ti]))
        jerk.append(ref.jerk(t[:, ti]))
        snap.append(ref.snap(t[:, ti]))

    vel = torch.stack(vel, dim=1).cpu().numpy()
    acc = torch.stack(acc, dim=1).cpu().numpy()
    jerk = torch.stack(jerk, dim=1).cpu().numpy()
    snap = torch.stack(snap, dim=1).cpu().numpy()

    # Plot results
    plot_idx = 1
    datetime = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))

    # Plot 3D trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos[plot_idx, :,0], pos[plot_idx, :,1], pos[plot_idx, :,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig(f'chainedpoly-{datetime}.png')

    # Plot x,y,z components
    fig, axs = plt.subplots(3, 5, figsize=(50, 10))
    for i in range(3):
        axs[i,0].plot(t[plot_idx], pos[plot_idx, :,i])
        axs[i,0].set_xlabel('t')
        axs[i,0].set_ylabel('x')
        axs[i,1].plot(t[plot_idx], vel[plot_idx, :,i])
        axs[i,1].set_xlabel('t')
        axs[i,1].set_ylabel('v')
        axs[i,2].plot(t[plot_idx], acc[plot_idx, :,i])
        axs[i,2].set_xlabel('t')
        axs[i,2].set_ylabel('a')
        axs[i,3].plot(t[plot_idx], jerk[plot_idx, :,i])
        axs[i,3].set_xlabel('t')
        axs[i,3].set_ylabel('j')
        axs[i,4].plot(t[plot_idx], snap[plot_idx, :,i])
        axs[i,4].set_xlabel('t')
        axs[i,4].set_ylabel('s')
    plt.tight_layout()
    plt.savefig(f'chainedpoly_xyz-{datetime}.png')