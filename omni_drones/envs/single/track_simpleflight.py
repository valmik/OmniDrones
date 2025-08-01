# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion, quat_rotate
import omni.isaac.core.utils.prims as prim_utils
import torch
import torch.distributions as D
from torch.func import vmap
from collections import deque

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec

from isaacsim.util.debug_draw import _debug_draw
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_from_matrix
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.kit.viewport.utility import get_active_viewport_window



import omegaconf
from ..utils.trajectory import ChainedPolynomial, RandomZigzag, NPointedStar, Lemniscate, Lissajous, Constant, RandomLissajous

class TrackSimpleFlight(IsaacEnv):
    r"""
    A basic control task. The goal for the agent is to track a reference
    lemniscate trajectory in the 3D space.

    ## Observation

    - `rpos` (3 * `future_traj_steps`): The relative position of the drone to the
      reference positions in the future `future_traj_steps` time steps.
    - `drone_state` (16 + `num_rotors`): The basic information of the drone (except its position),
      containing its rotation (in quaternion), velocities (linear and angular),
      heading and up vectors, and the current throttle.
    - `time_encoding` (optional): The time encoding, which is a 4-dimensional
      vector encoding the current progress of the episode.

    ## Reward

    - `pos`: Reward for tracking the trajectory, computed from the position
      error as {math}`\exp(-a * \text{pos_error})`.
    - `up`: Reward computed from the uprightness of the drone to discourage
      large tilting.
    - `spin`: Reward computed from the spin of the drone to discourage spinning.
    - `effort`: Reward computed from the effort of the drone to optimize the
      energy consumption.
    - `action_smoothness`: Reward that encourages smoother drone actions, computed based on the throttle difference of the drone.

    The total reward is computed as follows:

    ```{math}
        r = r_\text{pos} + r_\text{pos} * (r_\text{up} + r_\text{heading}) + r_\text{effort} + r_\text{action_smoothness}
    ```

    ## Episode End

    The episode ends when the tracking error is larger than `reset_thres`, or
    when the drone is too close to the ground, or when the episode reaches
    the maximum length.

    ## Config

    | Parameter               | Type  | Default       | Description                                                                                                                                                                                                                             |
    | ----------------------- | ----- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `drone_model`           | str   | "hummingbird" | Specifies the model of the drone being used in the environment.                                                                                                                                                                         |
    | `reset_thres`           | float | 0.5           | Threshold for the distance between the drone and its target, upon exceeding which the episode will be reset.                                                                                                                            |
    | `future_traj_steps`     | int   | 4             | Number of future trajectory steps the drone needs to predict.                                                                                                                                                                           |
    | `reward_distance_scale` | float | 1.2           | Scales the reward based on the distance between the drone and its target.                                                                                                                                                               |
    | `time_encoding`         | bool  | True          | Indicates whether to include time encoding in the observation space. If set to True, a 4-dimensional vector encoding the current progress of the episode is included in the observation. If set to False, this feature is not included. |
    """
    def __init__(self, cfg, headless):
        self.reset_thres = cfg.task.reset_thres
        
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.reward_distance_exp = cfg.task.reward_distance_exp
        self.reward_spin_weight = cfg.task.reward_spin_weight
        self.reward_up_weight = cfg.task.reward_up_weight

        self.time_encoding = cfg.task.time_encoding
        self.future_traj_steps = int(cfg.task.future_traj_steps)
        assert self.future_traj_steps > 0
        self.intrinsics = cfg.task.intrinsics
        self.wind = cfg.task.wind

        self.traj_type = cfg.task.traj_type

        super().__init__(cfg, headless)

        self.drone.initialize()
        randomization = self.cfg.task.get("randomization", None)
        if randomization is not None:
            if "drone" in self.cfg.task.randomization:
                self.drone.setup_randomization(self.cfg.task.randomization["drone"])

        if self.wind:
            if randomization is not None:
                if "wind" in self.cfg.task.randomization:
                    cfg = self.cfg.task.randomization["wind"]
                    # for phase in ("train", "eval"):
                    wind_intensity_scale = cfg['train'].get("intensity", None)
                    self.wind_intensity_low = wind_intensity_scale[0]
                    self.wind_intensity_high = wind_intensity_scale[1]
            else:
                self.wind_intensity_low = 0
                self.wind_intensity_high = 2
            self.wind_w = torch.zeros(self.num_envs, 3, 8, device=self.device)
            self.wind_i = torch.zeros(self.num_envs, 1, device=self.device)

        self.training_init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
        )
        self.origin = torch.tensor([0., 0., 5], device=self.device)

        if isinstance(self.traj_type, list) or isinstance(self.traj_type, omegaconf.ListConfig):
            self.ref, self.traj_t0 = zip(*[self.get_ref(traj_name) for traj_name in self.traj_type])
        else:
            self.ref, self.traj_t0 = zip(*[self.get_ref(self.traj_type)])

        self.ref_style_seq = torch.randint(0, len(self.ref), (self.num_envs,)).to(self.device)
        
        self.eval_init_rpy_dist = D.Uniform(
            torch.tensor([-.0, -.0, 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi
        )

        self.target_pos = torch.zeros(self.num_envs, self.future_traj_steps, 3, device=self.device)

        self.alpha = 0.8

        self.draw = _debug_draw.acquire_debug_draw_interface()

        if self.wind:
            wind_marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/WindMarkers",
                markers={
                    "wind_arrow": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                        scale=(0.3, 0.3, 1.0),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
                    ),
                },
            )
            self.wind_markers = VisualizationMarkers(wind_marker_cfg)

            self._wind_overlay_window = None
            self._wind_label = None

        self.vapor_trail_length = cfg.task.get("vapor_trail_length", 200)
        if self.vapor_trail_length > 0:
            self.vapor_trails = deque(maxlen=self.vapor_trail_length)
            alpha_decay = 0.95
            self.vapor_trail_sizes = []
            self.vapor_trail_colors = []
            for i in range(self.vapor_trail_length):
                alpha = alpha_decay ** i
                self.vapor_trail_sizes.append(max(1, int(3 * alpha)))
                self.vapor_trail_colors.append((0.2, 0.6, 1.0, alpha))


    def get_ref(self, traj_name):
        traj_t0 = torch.zeros(self.num_envs, 1, device=self.device)
        if traj_name == 'poly':
            ref = ChainedPolynomial(num_trajs=self.num_envs,
                                scale=2.5,
                                use_y=True,
                                use_z=True,
                                min_dt=1.5,
                                max_dt=4.0,
                                degree=5,
                                origin=self.origin,
                                device=self.device)
        elif traj_name == 'zigzag':
            ref = RandomZigzag(num_trajs=self.num_envs,
                                max_D=[1.0, 1.0, 1.0],
                                min_dt=1.0,
                                max_dt=1.5,
                                diff_axis=True,
                                origin=self.origin,
                                device=self.device)
        elif traj_name == 'random_lissajous':
            ref = RandomLissajous(num_trajs=self.num_envs,
                                  T=5.0,
                                origin=self.origin,
                                device=self.device)
        elif traj_name == 'pentagram':
            ref = NPointedStar(num_trajs=self.num_envs,
                                num_points=5,
                                origin=self.origin,
                                speed=1.0,
                                radius=0.7,
                                device=self.device)
        elif traj_name == 'slow':
            ref = Lemniscate(T=15.0, origin=self.origin, device=self.device)
            traj_t0 = torch.ones(self.num_envs, 1, device=self.device) * 15.0 / 4
        elif traj_name == 'normal':
            ref = Lemniscate(T=5.5, origin=self.origin, device=self.device)
            traj_t0 = torch.ones(self.num_envs, 1, device=self.device) * 5.5 / 4
        elif traj_name == 'fast':
            ref = Lemniscate(T=3.5, origin=self.origin, device=self.device)
            traj_t0 = torch.ones(self.num_envs, 1, device=self.device) * 3.5 / 4
        elif traj_name == 'lissajous':
            ref = Lissajous(origin=self.origin, device=self.device)
        elif traj_name == 'lissajous_fancy':
            ref = Lissajous(T = 3.0,origin=self.origin, device=self.device, ax=3, ay=2, az=2, fx=0.5, fy=1.0, fz=1.0, del_y=0.5)
        elif traj_name == 'constant':
            ref = Constant(pose=[1.0, 2.0, 3.0], origin=self.origin, device=self.device)
        else:
            raise ValueError(f"Invalid trajectory type: {traj_name}")
        return ref, traj_t0

    def _design_scene(self):
        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller
        )

        global_prim_list = []

        use_ground_plane = self.cfg.task.get("use_ground_plane", True)
        if use_ground_plane:
            kit_utils.create_ground_plane(
                "/World/defaultGroundPlane",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
                color=(0.2, 0.2, 0.2)
            )
            global_prim_list.append("/World/defaultGroundPlane")

        self.drone.spawn(translations=[(0.0, 0.0, 1.5)])
        return global_prim_list

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        obs_dim = drone_state_dim + 3 * (self.future_traj_steps-1)
        if self.time_encoding:
            self.time_encoding_dim = 4
            obs_dim += self.time_encoding_dim
        if self.intrinsics:
            obs_dim += sum(spec.shape[-1] for name, spec in self.drone.info_spec.items())

        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": UnboundedContinuousTensorSpec((1, obs_dim))
            }
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": {
                "action": self.drone.action_spec.unsqueeze(0),
            }
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": {
                "reward": UnboundedContinuousTensorSpec((1, 1))
            }
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
        )
        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "tracking_error": UnboundedContinuousTensorSpec(1),
            "tracking_error_ema": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec(1),
            "reward_pos": UnboundedContinuousTensorSpec(1),
            "reward_up": UnboundedContinuousTensorSpec(1),
            "reward_spin": UnboundedContinuousTensorSpec(1),
            "reward_action_smoothness": UnboundedContinuousTensorSpec(1),
            "reward_effort": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((1, drone_state_dim)),
            # "prev_action": self.drone.action_spec.unsqueeze(0),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

        traj_spec = CompositeSpec({
            "state": UnboundedContinuousTensorSpec((1, drone_state_dim)),
            "target_position": UnboundedContinuousTensorSpec((1, 3)),
            "time": UnboundedContinuousTensorSpec(1),
            "wind_acceleration": UnboundedContinuousTensorSpec((3,)),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["traj_stats"] = traj_spec
        self.traj_stats = traj_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)

        for ref in self.ref:
            ref.reset(env_ids)

        if self.training:
            self.ref_style_seq[env_ids] = torch.randint(0, len(self.ref), (len(env_ids),)).to(self.device)
            rot = euler_to_quaternion(self.training_init_rpy_dist.sample(env_ids.shape))
        else:
            self.ref_style_seq[env_ids] = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
            rot = euler_to_quaternion(self.eval_init_rpy_dist.sample(env_ids.shape))

        pos = self._compute_traj(2)
        pos = pos[env_ids, 0, :]
        vel = torch.zeros(len(env_ids), 1, 6, device=self.device)
        self.drone.set_world_poses(
            pos + self.envs_positions[env_ids], rot, env_ids
        )
        self.drone.set_velocities(vel, env_ids)

        self.stats[env_ids] = 0.

        if self._should_render(0) and (env_ids == self.central_env_idx).any() :
            # visualize the trajectory
            self.draw.clear_lines()

            traj_vis = self._compute_traj(self.max_episode_length, self.central_env_idx.unsqueeze(0))[0]
            traj_vis = traj_vis + self.envs_positions[self.central_env_idx]
            point_list_0 = traj_vis[:-1].tolist()
            point_list_1 = traj_vis[1:].tolist()
            colors = [(1.0, 1.0, 1.0, 1.0) for _ in range(len(point_list_0))]
            sizes = [1 for _ in range(len(point_list_0))]
            self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)

            if self.vapor_trail_length > 0:
                self.vapor_trails.clear()

        if self.wind:
            self.wind_i[env_ids] = torch.rand(*env_ids.shape, 1, device=self.device) * (self.wind_intensity_high-self.wind_intensity_low) + self.wind_intensity_low
            self.wind_w[env_ids] = torch.randn(*env_ids.shape, 3, 8, device=self.device)


    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)

        if self.wind:
            t = (self.progress_buf * self.dt).reshape(-1, 1, 1)
            self.wind_force = self.wind_i * torch.sin(t * self.wind_w).sum(-1)
            wind_forces = self.drone.MASS_0 * self.wind_force
            wind_forces = wind_forces.unsqueeze(1).expand(*self.drone.shape, 3)
            self.drone.base_link.apply_forces(wind_forces, is_global=True)

            self._visualize_wind()

    def _post_sim_step(self, tensordict: TensorDictBase):
        
        if self._should_render(0):
            if self.vapor_trail_length > 0:
                self._update_vapor_trail()
                self._visualize_vapor_trail()

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state()
        self.info["drone_state"] = self.drone_state

        self.target_pos[:] = self._compute_traj(self.future_traj_steps, step_size=5)

        self.rpos = self.target_pos - self.drone_state[..., :3]
        obs = [
            self.rpos.flatten(1).unsqueeze(1),
            self.drone_state[..., 3:],
        ]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        if self.intrinsics:
            obs.append(self.drone.get_info())

        obs = torch.cat(obs, dim=-1)

        self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference, (1-self.alpha))

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                },
                "stats": self.stats.clone(),
                "info": self.info.clone(),
                "traj_stats": self.traj_stats.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        # pos reward
        distance = torch.norm(self.rpos[:, [0]], dim=-1)
        self.stats["tracking_error"].add_(-distance)
        self.stats["tracking_error_ema"].lerp_(distance, (1-self.alpha))

        self.traj_stats["state"].copy_(self.drone_state)
        self.traj_stats["target_position"].copy_(self.target_pos[:, [0]])
        if self.wind:
            self.traj_stats["wind_acceleration"].copy_(self.wind_force)
        else:
            self.traj_stats["wind_acceleration"].copy_(torch.zeros(self.num_envs, 3, device=self.device))

        reward_pos = self.reward_distance_scale * torch.exp(-self.reward_distance_exp * distance)

        # uprightness
        tiltage = torch.abs(1 - self.drone.up[..., 2])
        reward_up = self.reward_up_weight * 0.5 / (1.0 + torch.square(tiltage))

        # effort
        reward_effort = self.reward_effort_weight * torch.exp(-self.effort)
        reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.drone.throttle_difference)

        # spin reward
        spin = torch.square(self.drone.vel[..., -1])
        reward_spin = self.reward_spin_weight * 0.5 / (1.0 + torch.square(spin))

        reward = (
            reward_pos
            + reward_pos * (reward_up + reward_spin)
            + reward_effort
            + reward_action_smoothness
        )

        self.stats["reward_pos"].add_(reward_pos)
        self.stats["reward_up"].add_(reward_pos * reward_up)
        self.stats["reward_spin"].add_(reward_pos * reward_spin)
        self.stats["reward_effort"].add_(reward_effort)
        self.stats["reward_action_smoothness"].add_(reward_action_smoothness)

        misbehave = (
            (self.drone.pos[..., 2] < 0.1)
            | (distance > self.reset_thres)
        )
        hasnan = torch.isnan(self.drone_state).any(-1)

        terminated = misbehave | hasnan
        truncated = (self.progress_buf >= self.max_episode_length - 1).unsqueeze(-1)

        done = terminated | truncated

        ep_len = self.progress_buf.unsqueeze(-1)
        self.stats["tracking_error"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["reward_pos"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["reward_up"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )   
        self.stats["reward_spin"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["reward_effort"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )   
        self.stats["reward_action_smoothness"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )

        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        self.traj_stats["time"][:] = self.progress_buf.unsqueeze(1)*self.dt

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1),
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )

    def _compute_traj(self, steps: int, env_ids=None, step_size: float=1.):
        assert steps > 1
        if env_ids is None:
            env_ids = ...
        t = self.progress_buf[env_ids].unsqueeze(1) + step_size * torch.arange(steps, device=self.device)
        
        if len(self.ref) == 1:
            t = self.traj_t0[0] + t * self.dt
            target_pos = self.ref[0].batch_pos(t)
        elif len(self.ref) == 2:
            t = self.traj_t0[0] + t * self.dt
            pos_1 = self.ref[0].batch_pos(t)
            pos_2 = self.ref[1].batch_pos(t)
            target_pos = pos_1 * (1 - self.ref_style_seq[env_ids].unsqueeze(1).unsqueeze(1)) + pos_2 * self.ref_style_seq[env_ids].unsqueeze(1).unsqueeze(1)
        else:
            one_hot = torch.nn.functional.one_hot(self.ref_style_seq[env_ids], len(self.ref)).float()

            traj_t0 = sum(one_hot[:, i] * self.traj_t0[i].squeeze(1) for i in range(len(self.ref)))
            t = traj_t0.unsqueeze(1) + t * self.dt
            target_pos = sum(one_hot[:, i].unsqueeze(-1).unsqueeze(-1) * self.ref[i].batch_pos(t) for i in range(len(self.ref)))

        return target_pos
    
    def _update_vapor_trail(self):
        pos = self.drone.pos[self.central_env_idx].squeeze()
        self.vapor_trails.append(pos.tolist())

    def _visualize_vapor_trail(self):
        # if not self._should_render(0) or not self.vapor_trail_length > 0:
        #     return
        
        trail_points = list(self.vapor_trails)
        if len(trail_points) < 2:
            return
        
        point_list_0 = trail_points[:-1]
        point_list_1 = trail_points[1:]

        num_segments = len(point_list_0)
        colors = self.vapor_trail_colors[:num_segments]
        sizes = self.vapor_trail_sizes[:num_segments]

        colors = colors[::-1]
        sizes = sizes[::-1]

        self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)
        
    def _visualize_wind(self):
        """Visualize wind vector using Isaac Lab's 3D arrow markers"""
        if not self.wind or not self._should_render(0):
            return
        
        central_idx = self.central_env_idx
        
        # Get current wind force for the central environment
        current_wind_force = self.wind_force[central_idx]  # Shape: (3,)
        wind_magnitude = torch.norm(current_wind_force).item()

        self._create_wind_overlay(wind_magnitude)
        
        if wind_magnitude < 0.01:  # Don't draw very small wind vectors
            return
        
        # Normalize wind direction
        wind_direction = current_wind_force / (wind_magnitude + 1e-8)
        
        # Position the arrow in a fixed location relative to the drone
        # drone_pos = self.drone.pos[central_idx].squeeze()  # Shape: (3,)
        arrow_position = self.origin + torch.tensor([3.0, 3.0, 2.0], device=self.device)  # Shape: (3,)
        
        # Create rotation matrix to align arrow with wind direction
        # The arrow_x.usd points in the +X direction by default
        x_axis = wind_direction  # Wind direction becomes the X axis
        
        # Create a reasonable Y axis (perpendicular to wind in horizontal plane if possible)
        if abs(wind_direction[2].item()) < 0.9:  # Not mostly vertical
            y_axis = torch.tensor([-wind_direction[1], wind_direction[0], 0], device=self.device)
            y_axis = y_axis / (torch.norm(y_axis) + 1e-8)
        else:  # Mostly vertical wind, choose arbitrary horizontal Y
            y_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        
        # Z axis is cross product of X and Y
        z_axis = torch.linalg.cross(x_axis, y_axis)
        z_axis = z_axis / (torch.norm(z_axis) + 1e-8)
        
        # Recompute Y to ensure orthogonality
        y_axis = torch.linalg.cross(z_axis, x_axis)
        
        # Create rotation matrix and convert to quaternion
        rotation_matrix = torch.stack([x_axis, y_axis, z_axis], dim=1)
        arrow_orientation = quat_from_matrix(rotation_matrix.unsqueeze(0))  # Shape: (1, 4)
        
        # Scale the arrow based on wind intensity
        scale_factor = wind_magnitude / self.wind_intensity_high
        scale_factor = max(0.1, min(3.0, scale_factor))
        diam_scale = max(0.1, min(3.0, scale_factor))

        arrow_scale = torch.tensor([[0.3*diam_scale, 0.3*diam_scale, 1.0*scale_factor]], device=self.device)
        
        # Prepare data for visualization
        arrow_positions = arrow_position.unsqueeze(0)  # Shape: (1, 3)
        marker_indices = torch.tensor([0], device=self.device)  # Use first (and only) marker type
        
        # Visualize the wind arrow
        self.wind_markers.visualize(
            translations=arrow_positions,
            orientations=arrow_orientation,
            marker_indices=marker_indices,
            scales=arrow_scale
        )

    def _create_wind_overlay(self, wind_magnitude):
        """Create a text label to display wind information"""
        try:
            import omni.ui as ui
            
            overlay_text = f"Wind Acceleration: {wind_magnitude:.2f} m/s²"
            
            # Create a simple overlay window without relying on viewport dimensions
            if not hasattr(self, '_wind_overlay_window') or self._wind_overlay_window is None:
                self._wind_overlay_window = ui.Window(
                    "Wind Info", 
                    width=250, 
                    height=60,
                    flags=ui.WINDOW_FLAGS_NO_TITLE_BAR | ui.WINDOW_FLAGS_NO_RESIZE
                )
                # Position in top-right corner (these coordinates work for most screen sizes)
                self._wind_overlay_window.position_x = 1400  # Adjust based on your screen
                self._wind_overlay_window.position_y = 50
                
                with self._wind_overlay_window.frame:
                    with ui.VStack(style={"background_color": 0x22000000}):  # Semi-transparent background
                        self._wind_label = ui.Label(
                            overlay_text,
                            style={
                                "color": 0xFFFFFFFF, 
                                "font_size": 16,
                                "margin": 5
                            }
                        )
            elif hasattr(self, '_wind_label') and self._wind_label is not None:
                # Update existing label
                self._wind_label.text = overlay_text
                    
        except (ImportError, Exception) as e:
            pass