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

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
from omni.isaac.debug_draw import _debug_draw

from ..utils.trajectory import ChainedPolynomial, RandomZigzag, NPointedStar, Lemniscate

class Track(IsaacEnv):
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

        self.use_eval = cfg.task.use_eval
        self.eval_traj = cfg.task.eval_traj

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

        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
        )
        self.origin = torch.tensor([0., 0., 2.], device=self.device)

        self.ref = [ChainedPolynomial(num_trajs=self.num_envs,
                                scale=2.5,
                                use_y=True,
                                min_dt=1.5,
                                max_dt=4.0,
                                degree=5,
                                origin=self.origin,
                                device=self.device),
                    RandomZigzag(num_trajs=self.num_envs,
                                max_D=[1.0, 1.0, 0.0],
                                min_dt=1.0,
                                max_dt=1.5,
                                diff_axis=True,
                                origin=self.origin,
                                device=self.device)]
        
        self.ref_style_seq = torch.randint(0, 2, (self.num_envs,)).to(self.device)
        self.traj_t0 = torch.zeros(self.num_envs, 1, device=self.device)
        
        if self.use_eval:
            self.init_rpy_dist = D.Uniform(
                torch.tensor([-.0, -.0, 0.], device=self.device) * torch.pi,
                torch.tensor([0., 0., 0.], device=self.device) * torch.pi
            )
            if self.eval_traj == 'poly':
                self.ref = [ChainedPolynomial(num_trajs=self.num_envs,
                                scale=2.5,
                                use_y=True,
                                min_dt=1.5,
                                max_dt=4.0,
                                degree=5,
                                origin=self.origin,
                                device=self.device)]
            elif self.eval_traj == 'zigzag':
                self.ref = [RandomZigzag(num_trajs=self.num_envs,
                                max_D=[1.0, 1.0, 0.0],
                                min_dt=1.0,
                                max_dt=1.5,
                                diff_axis=True,
                                origin=self.origin,
                                device=self.device)]
            elif self.eval_traj == 'pentagram':
                self.ref = [NPointedStar(num_trajs=self.num_envs,
                                num_points=5,
                                origin=self.origin,
                                speed=1.0,
                                radius=0.7,
                                device=self.device)]
            elif self.eval_traj == 'slow':
                self.ref = Lemniscate(T=15.0, origin=self.origin, device=self.device)
                self.traj_t0 = torch.ones(self.num_envs, 1, device=self.device) * 15.0 / 4
            elif self.eval_traj == 'normal':
                self.ref = Lemniscate(T=5.5, origin=self.origin, device=self.device)
                self.traj_t0 = torch.ones(self.num_envs, 1, device=self.device) * 5.5 / 4
            elif self.eval_traj == 'fast':
                self.ref = Lemniscate(T=3.5, origin=self.origin, device=self.device)
                self.traj_t0 = torch.ones(self.num_envs, 1, device=self.device) * 3.5 / 4

        self.target_pos = torch.zeros(self.num_envs, self.future_traj_steps, 3, device=self.device)

        self.alpha = 0.8

        self.draw = _debug_draw.acquire_debug_draw_interface()

    def _design_scene(self):
        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller
        )

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        self.drone.spawn(translations=[(0.0, 0.0, 1.5)])
        return ["/World/defaultGroundPlane"]

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
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)

        if not self.use_eval:
            self.ref[0].reset(env_ids)
            self.ref[1].reset(env_ids)
            self.ref_style_seq[env_ids] = torch.randint(0, 2, (len(env_ids),)).to(self.device)

        if self.use_eval:
            self.ref.reset(env_ids)

        pos = torch.zeros(len(env_ids), 3, device=self.device)
        pos = pos + self.origin # init: (0, 0, 1)
        rot = euler_to_quaternion(self.init_rpy_dist.sample(env_ids.shape))
        vel = torch.zeros(len(env_ids), 1, 6, device=self.device)
        self.drone.set_world_poses(
            pos + self.envs_positions[env_ids], rot, env_ids
        )
        self.drone.set_velocities(vel, env_ids)


        # self.traj_c[env_ids] = self.traj_c_dist.sample(env_ids.shape)
        # self.traj_rot[env_ids] = euler_to_quaternion(self.traj_rpy_dist.sample(env_ids.shape))
        # self.traj_scale[env_ids] = self.traj_scale_dist.sample(env_ids.shape)
        # traj_w = self.traj_w_dist.sample(env_ids.shape)
        # self.traj_w[env_ids] = torch.randn_like(traj_w).sign() * traj_w

        # t0 = torch.zeros(len(env_ids), device=self.device)
        # pos = lemniscate(t0 + self.traj_t0, self.traj_c[env_ids]) + self.origin
        # rot = euler_to_quaternion(self.init_rpy_dist.sample(env_ids.shape))
        # vel = torch.zeros(len(env_ids), 1, 6, device=self.device)
        # self.drone.set_world_poses(
        #     pos + self.envs_positions[env_ids], rot, env_ids
        # )
        # self.drone.set_velocities(vel, env_ids)

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

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state()

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
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        # pos reward
        distance = torch.norm(self.rpos[:, [0]], dim=-1)
        self.stats["tracking_error"].add_(-distance)
        self.stats["tracking_error_ema"].lerp_(distance, (1-self.alpha))

        reward_pos = torch.exp(-self.reward_distance_scale * distance)

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
        if env_ids is None:
            env_ids = ...
        t = self.progress_buf[env_ids].unsqueeze(1) + step_size * torch.arange(steps, device=self.device)
        
        t = self.traj_t0 + t * self.dt
        
        if not self.use_eval:
            smooth = self.ref[0].batch_pos(t)
            zigzag = self.ref[1].batch_pos(t)
            target_pos = smooth * (1 - self.ref_style_seq[env_ids].unsqueeze(1)) + zigzag * self.ref_style_seq[env_ids].unsqueeze(1)
        else:
            target_pos = []
            for ti in range(t.shape[1]):
                target_pos.append(self.ref.pos(t[:, ti]))
            target_pos = torch.stack(target_pos, dim=1)[env_ids]

        return target_pos
        # t = self.traj_t0 + scale_time(self.traj_w[env_ids].unsqueeze(1) * t * self.dt)
        # traj_rot = self.traj_rot[env_ids].unsqueeze(1).expand(-1, t.shape[1], 4)

        # target_pos = vmap(lemniscate)(t, self.traj_c[env_ids])
        # target_pos = vmap(quat_rotate)(traj_rot, target_pos) * self.traj_scale[env_ids].unsqueeze(1)

        # return self.origin + target_pos
