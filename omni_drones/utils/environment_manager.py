import logging
from typing import Dict, Any, Optional
import torch
import numpy as np
import time
from omegaconf import OmegaConf
from torchrl.envs.transforms import TransformedEnv, Compose
from omni_drones.robots.robot import RobotBase
from omni_drones.envs.isaac_env import IsaacEnv

# Configure logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

class EvalEnvironmentManager:
    """Manages switching between different evaluation environments by clearing and rebuilding scenes."""
    
    def __init__(self, base_cfg, env_creator, device, training_base_env: IsaacEnv = None, training_env: TransformedEnv = None):
        self.base_cfg = base_cfg
        self.env_creator = env_creator
        self.device = device
        
        # Keep track of current environment
        self.current_env_name: Optional[str] = None
        self.current_env = None
        self.current_base_env = None

        if training_base_env is not None and training_env is not None:
            self.current_env_name = 'training'
            self.current_env = training_env
            self.current_base_env = training_base_env

    def get_training_env(self):
        """Get the training environment."""
        return self.get_env('training', OmegaConf.create({}))
        
    def get_env(self, eval_name: str, eval_cfg_changes: Dict[str, Any]):
        """Get an evaluation environment, rebuilding the scene if necessary."""
        
        if self.current_env_name == eval_name:
            # Already have the right environment loaded
            return self.current_env, self.current_base_env
            
        # Need to switch environments
        logging.info(f"Switching to eval environment: {eval_name}")
        
        # Clean up current environment
        self._cleanup_current_env()
        
        # Create new evaluation config
        eval_cfg = OmegaConf.merge(self.base_cfg, eval_cfg_changes)
        
        # Clear the scene and robot registry
        self._clear_scene()

        # Create new environment
        try:
            env, base_env = self.env_creator(eval_cfg)
            
            self.current_env = env
            self.current_base_env = base_env
            self.current_env_name = eval_name

            for i in range(10):
                try:
                    env.reset()
                    base_env.sim.step(render=False)
                except:
                    pass
            
            logging.info(f"Successfully created eval environment: {eval_name}")
            return env, base_env
            
        except Exception as e:
            logging.error(f"Failed to create eval environment {eval_name}: {e}")
            raise
    
    def _cleanup_current_env(self):
        """Clean up the current environment."""
        if self.current_env is not None:
            try:
                # Note: We avoid calling .close() due to the issues you mentioned
                # Instead we'll rely on scene clearing
                pass
            except Exception as e:
                logging.warning(f"Error during environment cleanup: {e}")
            
        # Clear CUDA cache before scene cleanup
        try:
            torch.cuda.empty_cache()
            logging.info("CUDA cache cleared")
        except Exception as e:
            logging.warning(f"Error clearing CUDA cache: {e}")
            
        self.current_env = None
        self.current_base_env = None
        self.current_env_name = None

        import gc
        gc.collect()
        time.sleep(0.1)
    
    def _clear_scene(self):
        """Clear the Isaac Sim scene and robot registry."""
        try:
            # Clear robot registry to allow new instances
            RobotBase._robots.clear()
            
            # Clear the stage (this removes all prims)
            import omni.usd
            from omni.isaac.core.simulation_context import SimulationContext
            import time

            simulation_context = SimulationContext.instance()
            if simulation_context:
                simulation_context.stop()
                time.sleep(0.1)
                # Clear the instance so the new environment can create a fresh one
                SimulationContext.clear_instance()

            # Clear the stage
            omni.usd.get_context().new_stage()
            
            logging.info("Scene cleared successfully")
            
        except Exception as e:
            logging.warning(f"Error clearing scene: {e}")
            raise

    def _reset_physics_views(self, drone):
        """Reset all physics-related views that cache backend references"""
        try:
            if hasattr(drone, '_view'):
                drone._view = None  # Force recreation
            if hasattr(drone, '_physics_view'):
                drone._physics_view = None
            if hasattr(drone, 'rotors_view'):
                drone.rotors_view = None
            if hasattr(drone, '_articulation_view'):
                drone._articulation_view = None

            for attr_name in dir(drone):
                if attr_name.endswith('_view') and not attr_name.startswith('_'):
                    try:
                        setattr(drone, attr_name, None)
                    except Exception as e:
                        logging.warning(f"Error resetting physics view {attr_name}: {e}")
        except Exception as e:
            logging.warning(f"Error resetting physics views: {e}")


# Modified evaluation function
@torch.no_grad()
def evaluate_with_manager(
    eval_env_manager: EvalEnvironmentManager,
    eval_name: str,
    eval_diffs: Dict[str, Any],
    policy,
    seed: int = 0,
    exploration_type=None
):
    """Evaluate using the environment manager."""
    
    # try:
    # Get the evaluation environment (may trigger scene rebuild)
    env, base_env = eval_env_manager.get_env(eval_name, eval_diffs)
    
    # Prepare for evaluation
    base_env.enable_render(True)
    base_env.eval()
    env.eval()
    env.set_seed(seed)
    
    # Run evaluation
    info, traj_data = evaluate_single_env_direct(env, base_env, policy, eval_name, seed, exploration_type)
    
    # Clean up after evaluation
    base_env.enable_render(not eval_env_manager.base_cfg.get("headless", True))
        
    # except Exception as e:
    #     logging.error(f"Error evaluating environment {eval_name}: {e}")
    #     # Continue with other environments
    #     continue
    
    return info, traj_data

@torch.no_grad()
def evaluate_current_env(manager, policy, seed, exploration_type):
    """Evaluate the current environment."""
    env = manager.current_env
    base_env = manager.current_base_env
    eval_name = manager.current_env_name
    if not env or not base_env:
        raise ValueError("No environment or base environment found")
    
    # Prepare for evaluation
    base_env.enable_render(True)
    base_env.eval()
    env.eval()
    env.set_seed(seed)

    info, traj_data = evaluate_single_env_direct(env, base_env, policy, eval_name, seed, exploration_type)

    base_env.enable_render(not manager.base_cfg.get("headless", True))

    env.train()
    base_env.train()

    return info, traj_data

@torch.no_grad()
def evaluate_single_env_direct(env, base_env, policy, eval_name, seed, exploration_type):
    """Direct evaluation without environment creation."""
    from omni_drones.utils.torchrl import RenderCallback
    from torchrl.envs.utils import set_exploration_type
    import wandb

    render_callback = RenderCallback(interval=2)
    
    with set_exploration_type(exploration_type):
        trajs = env.rollout(
            max_steps=base_env.max_episode_length,
            policy=policy,
            callback=render_callback,
            auto_reset=True,
            break_when_any_done=False,
            return_contiguous=False,
        )
    
    env.reset()

    central_env_idx = base_env.central_env_idx
    
    done = trajs.get(("next", "done"))
    first_done = torch.argmax(done.long(), dim=1).cpu()
    
    def take_first_episode(tensor: torch.Tensor):
        indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
        return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)
    
    traj_stats = {
        k: take_first_episode(v)
        for k, v in trajs[("next", "stats")].cpu().items()
    }

    try:
        traj_data = {
            k: v[:, :first_done.max()+1]
            for k, v in trajs[("next", "traj_stats")].cpu().items()
        }
        
    except Exception as e:
        logging.error(f"Error processing trajectory data: {e}")
        traj_data = {}
    
    info = {
        f"eval_{eval_name}/stats." + k: torch.mean(v.float()).item()
        for k, v in traj_stats.items()
    }
    
    # Add video if needed
    info[f"eval_{eval_name}/recording"] = wandb.Video(
        render_callback.get_video_array(axes="t c h w"),
        fps=0.5 / (base_env.dt * base_env.substeps),
        format="mp4"
    )

    # Make trajectory plots
    plot_info = make_trajectory_plots(traj_data, eval_name, central_env_idx)
    info.update(plot_info)
    
    return info, traj_data


def make_trajectory_plots(traj_data: Dict[str, torch.Tensor], eval_name: str, central_env_idx: int = 0):
    """Plot the trajectory data."""
    from omni_drones.utils.math import quaternion_to_euler

    try:
        position = traj_data["state"][central_env_idx, 1:, 0, :3].numpy()  # [time, 3]
        quaternion = traj_data["state"][central_env_idx, 1:, 0, 3:7]  # [time, 4]
        euler = quaternion_to_euler(quaternion).numpy()  # [time, 3]
    except Exception as e:
        logging.error(f"Error plotting trajectory: {e}")
        return {}
    
    try:
        target_position = traj_data["target_position"][central_env_idx, 1:, 0, :3].numpy()  # [time, 3]
    except Exception as e:
        target_position = None

    try:
        time = traj_data["time"][central_env_idx, 1:, 0].numpy()  # [time]
    except Exception as e:
        time = None

    try:
        wind_acceleration = traj_data["wind_acceleration"][central_env_idx, 1:, :3].numpy()  # [time, 3]
    except Exception as e:
        wind_acceleration = None

    plot_info = {}
    try:
        plot_info.update(plot_3d_trajectory(position, target_position, eval_name))
    except Exception as e:
        logging.error(f"Error plotting 3D trajectory: {e}")
    try:
        plot_info.update(plot_trajectory_position(position, target_position, time, wind_acceleration, eval_name))
    except Exception as e:
        logging.error(f"Error plotting position trajectory: {e}")
    try:
        plot_info.update(plot_trajectory_rpy(euler, time, eval_name))
    except Exception as e:
        logging.error(f"Error plotting RPY trajectory: {e}")
    
    return plot_info

    

def plot_3d_trajectory(position: np.ndarray, target_position: np.ndarray | None, eval_name: str):
    """Plot the 3D trajectory."""   
    import matplotlib.pyplot as plt
    import wandb

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot position trajectory
    ax.plot(position[:, 0], position[:, 1], position[:, 2], 
            label='Position', linewidth=2)
    
    # Plot target position trajectory
    if target_position is not None:
        ax.plot(target_position[:, 0], target_position[:, 1], target_position[:, 2],
                label='Target Position', linewidth=2, linestyle='--')
    
    # Add labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    plt.title(f'3D Trajectory Plot - {eval_name}')

    plot_info = {
        f"eval_{eval_name}/3d_plot": wandb.Image(fig)
    }

    plt.close(fig)
    return plot_info

def plot_trajectory_position(
        position: np.ndarray, 
        target_position: np.ndarray | None, 
        time: np.ndarray | None, 
        wind_acceleration: np.ndarray | None,
        eval_name: str
        ):
    """Plot the trajectory position with wind acceleration on secondary y-axis."""
    import matplotlib.pyplot as plt
    import wandb    
    
    fig = plt.figure(figsize=(12, 10))
    
    # Create time array if not provided
    if time is None:
        time = np.arange(len(position))
        time_label = 'Step'
    else:
        time_label = 'Time'

    if wind_acceleration is not None and np.isclose(wind_acceleration, 0).all():
        wind_acceleration = None
    
    # Create subplots for x, y, z positions
    ax_x = fig.add_subplot(311)
    ax_y = fig.add_subplot(312)
    ax_z = fig.add_subplot(313)
    
    # Plot x position
    line_x_pos = ax_x.plot(time, position[:, 0], label='Position', linewidth=2, color='blue')
    if target_position is not None:
        line_x_target = ax_x.plot(time, target_position[:, 0], label='Target', linewidth=2, linestyle='--', color='red')
    ax_x.set_ylabel('X Position', color='blue')
    ax_x.tick_params(axis='y', labelcolor='blue')
    ax_x.grid(True)
    
    # Add wind acceleration for x-axis if available
    if wind_acceleration is not None:
        ax_x_wind = ax_x.twinx()
        line_x_wind = ax_x_wind.plot(time, wind_acceleration[:, 0], label='Wind Accel', linewidth=1.5, color='green', alpha=0.7)
        ax_x_wind.set_ylabel('Wind Acceleration X', color='green')
        ax_x_wind.tick_params(axis='y', labelcolor='green')
        
        # Combine legends
        lines_x = line_x_pos + (line_x_target if target_position is not None else []) + line_x_wind
        labels_x = [l.get_label() for l in lines_x]
        ax_x.legend(lines_x, labels_x, loc='upper right')
    else:
        ax_x.legend(loc='upper right')
    
    # Plot y position
    line_y_pos = ax_y.plot(time, position[:, 1], label='Position', linewidth=2, color='blue')
    if target_position is not None:
        line_y_target = ax_y.plot(time, target_position[:, 1], label='Target', linewidth=2, linestyle='--', color='red')
    ax_y.set_ylabel('Y Position', color='blue')
    ax_y.tick_params(axis='y', labelcolor='blue')
    ax_y.grid(True)
    
    # Add wind acceleration for y-axis if available
    if wind_acceleration is not None:
        ax_y_wind = ax_y.twinx()
        line_y_wind = ax_y_wind.plot(time, wind_acceleration[:, 1], label='Wind Accel', linewidth=1.5, color='green', alpha=0.7)
        ax_y_wind.set_ylabel('Wind Acceleration Y', color='green')
        ax_y_wind.tick_params(axis='y', labelcolor='green')
        
        # Combine legends
        lines_y = line_y_pos + (line_y_target if target_position is not None else []) + line_y_wind
        labels_y = [l.get_label() for l in lines_y]
        ax_y.legend(lines_y, labels_y, loc='upper right')
    else:
        ax_y.legend(loc='upper right')
    
    # Plot z position
    line_z_pos = ax_z.plot(time, position[:, 2], label='Position', linewidth=2, color='blue')
    if target_position is not None:
        line_z_target = ax_z.plot(time, target_position[:, 2], label='Target', linewidth=2, linestyle='--', color='red')
    ax_z.set_xlabel(time_label)
    ax_z.set_ylabel('Z Position', color='blue')
    ax_z.tick_params(axis='y', labelcolor='blue')
    ax_z.grid(True)
    
    # Add wind acceleration for z-axis if available
    if wind_acceleration is not None:
        ax_z_wind = ax_z.twinx()
        line_z_wind = ax_z_wind.plot(time, wind_acceleration[:, 2], label='Wind Accel', linewidth=1.5, color='green', alpha=0.7)
        ax_z_wind.set_ylabel('Wind Acceleration Z', color='green')
        ax_z_wind.tick_params(axis='y', labelcolor='green')
        
        # Combine legends
        lines_z = line_z_pos + (line_z_target if target_position is not None else []) + line_z_wind
        labels_z = [l.get_label() for l in lines_z]
        ax_z.legend(lines_z, labels_z, loc='upper right')
    else:
        ax_z.legend(loc='upper right')
    
    plt.suptitle(f'Position vs Time - {eval_name}')
    plt.tight_layout()
    
    plot_info = {
        f"eval_{eval_name}/position_plot": wandb.Image(fig)
    }
    
    plt.close(fig)
    return plot_info

def plot_trajectory_rpy(rpy: np.ndarray, time: np.ndarray | None, eval_name: str):
    """Plot the trajectory rpy."""
    import matplotlib.pyplot as plt
    import wandb

    fig = plt.figure(figsize=(12, 8))

    # Create time array if not provided
    if time is None:
        time = np.arange(len(rpy))
        time_label = 'Step'
    else:
        time_label = 'Time'
    
    ax_roll = fig.add_subplot(311)
    ax_pitch = fig.add_subplot(312)
    ax_yaw = fig.add_subplot(313)   

    ax_roll.plot(time, rpy[:, 0], label='Roll', linewidth=2)
    ax_pitch.plot(time, rpy[:, 1], label='Pitch', linewidth=2)
    ax_yaw.plot(time, rpy[:, 2], label='Yaw', linewidth=2)

    ax_roll.set_ylabel('Roll')
    ax_pitch.set_ylabel('Pitch')
    ax_yaw.set_ylabel('Yaw')

    ax_yaw.set_xlabel(time_label) 

    plt.suptitle(f'RPY vs Time - {eval_name}')
    plt.tight_layout()

    plot_info = {
        f"eval_{eval_name}/rpy_plot": wandb.Image(fig)
    }   

    plt.close(fig)
    return plot_info

    
