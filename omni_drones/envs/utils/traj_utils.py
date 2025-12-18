import importlib.util
from pathlib import Path

import torch

try:
    from .trajectory.base import BaseTrajectory
except ImportError:
    # Use importlib to load the module directly without initializing parent packages
    base_path = Path(__file__).parent / "trajectory" / "base.py"
    spec = importlib.util.spec_from_file_location("base", base_path)
    base_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base_module)
    BaseTrajectory = base_module.BaseTrajectory

def find_closest_point_on_trajectory(
    trajectory: BaseTrajectory, 
    point: torch.Tensor, 
    t_guess: float, 
    tol: float = 1e-6, 
    max_iter: int = 1000
) -> float:
    """
    Find the closest point on a trajectory to a given point.
    """
    assert point.shape == (3,)

    # Convert t_guess to tensor if it's not already
    if isinstance(t_guess, torch.Tensor):
        t = t_guess.clone().detach()
    else:
        t = torch.tensor(t_guess, dtype=torch.float32)

    t = torch.clamp(t, 0.0, trajectory.T)
    
    for _ in range(max_iter):
        pos = trajectory.pos(t)
        vel = trajectory.vel(t)
        acc = trajectory.acc(t)

        error = pos - point

        f = torch.dot(error, vel)

        if abs(f.item()) < tol:
            break

        df = torch.dot(vel, vel) + torch.dot(error, acc)
        if df.item() == 0:
            return t.item()

        t = t - f / df

        t = torch.clamp(t, 0.0, trajectory.T)

    return t

def find_closest_point_on_trajectory_brute_force(
    trajectory: BaseTrajectory, 
    point: torch.Tensor, 
    tol: float = 1e-6, 
    num_samples: int = 10000,
) -> torch.Tensor:
    """
    Find the closest point on a trajectory to a given point or vector of points.
    
    Args:
        trajectory: The trajectory to search on
        point: A single point of shape (3,) or a vector of points of shape (N, 3)
        tol: Tolerance (unused, kept for compatibility)
    
    Returns:
        If point is shape (3,): returns a scalar tensor (or float)
        If point is shape (N, 3): returns a tensor of shape (N,) with closest times for each point
    """
    point = point.squeeze()
    # Handle both single point and vector of points
    is_single_point = point.ndim == 1
    if is_single_point:
        point = point.unsqueeze(0)  # Shape: (1, 3)
    
    num_points = point.shape[0]
    times = torch.linspace(0.0, trajectory.T, num_samples, device=point.device)
    
    # Compute all trajectory positions at once: shape (num_times, 3)
    traj_positions = trajectory.pos(times)  # Shape: (num_times, 3)
    
    # Compute distances for all points and all times
    # point: (num_points, 3), traj_positions: (num_times, 3)
    # We want: (num_points, num_times)
    point_expanded = point.unsqueeze(1)  # Shape: (num_points, 1, 3)
    traj_expanded = traj_positions.unsqueeze(0)  # Shape: (1, num_times, 3)
    distances = torch.norm(point_expanded - traj_expanded, dim=-1)  # Shape: (num_points, num_times)
    
    # Find closest time for each point
    closest_indices = torch.argmin(distances, dim=1)  # Shape: (num_points,)
    closest_times = times[closest_indices]  # Shape: (num_points,)
    
    # Return scalar tensor if single point, otherwise return tensor
    if is_single_point:
        return closest_times.squeeze(0)  # Returns a scalar tensor
    return closest_times


def _test_point(traj, point, t_guess = 0.0):
    # print(f"point: {point}")
    t0 = find_closest_point_on_trajectory(traj, point, t_guess)
    # print(f"t_from_original_guess: {t0}")
    # print(f"pos_from_original_guess: {traj.pos(t0)}")
    # working_ts = [t0]
    # for t_guess in torch.linspace(0.0, traj.T, 100):
    #     t = find_closest_point_on_trajectory(traj, point, t_guess)
    #     if torch.abs(t - t0) > 1e-2:
    #         print(f"t: {t}")
    #         print(f"pos: {traj.pos(t)}")
    #         working_ts.append(t)
    # working_ts = torch.stack(working_ts)
    return t0
    

def _test_traj(traj):
    import matplotlib.pyplot as plt
    import time
    datetime = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))

    times = torch.linspace(0.0, traj.T, 100)
    points = traj.pos(times) + torch.rand(100, 3)*0.2
    closest_points = []
    for time, point in zip(times, points):
        t = find_closest_point_on_trajectory(traj, point, time)
        tbf = find_closest_point_on_trajectory_brute_force(traj, point)
        if torch.abs(t - tbf) > 1e-2:
            distance = torch.norm(traj.pos(t) - point)
            distancebf = torch.norm(traj.pos(tbf) - point)
            if distance > distancebf:
                print(f"t: {t}")
                print(f"tbf: {tbf}")
                print(f"distance: {distance}")
                print(f"distancebf: {distancebf}")
                print(f"point: {point}")
                print(f"pos: {traj.pos(t)}")
                print(f"posbf: {traj.pos(tbf)}")

        closest_points.append(traj.pos(t))
    closest_points = torch.stack(closest_points)

    t = torch.linspace(0.0, traj.T, 1000)
    pos = traj.pos(t).cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos[:,0], pos[:,1], pos[:,2], 'b-')
    for point, closest_point in zip(points, closest_points):
        ax.plot([point[0], closest_point[0]], [point[1], closest_point[1]], [point[2], closest_point[2]], 'r-')
    plt.savefig(f'closest_points-{datetime}.png')
    return closest_points



if __name__ == "__main__":
    trajectory_dir = Path(__file__).parent / "trajectory"
    trajectory_files = [
        f for f in trajectory_dir.glob("*.py")
        if not f.name.startswith("_") and f.name != "base.py"
    ]

    for tf in trajectory_files:
        module_name = tf.stem
        spec = importlib.util.spec_from_file_location(module_name, tf)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        globals().update({k: v for k, v in mod.__dict__.items() if not k.startswith("_")})


    slalom = globals()["Slalom"](origin=torch.zeros(3), device="cpu")

    _test_traj(slalom)
    