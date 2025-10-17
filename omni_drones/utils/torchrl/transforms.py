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


from typing import Any, Dict, Optional, Sequence, Union, Tuple, List

import torch
from tensordict.tensordict import TensorDictBase, TensorDict
from torch.optim.optimizer import required
from torchrl.data.tensor_specs import TensorSpec
from torchrl.envs.common import EnvBase
from torchrl.envs.transforms import (
    TransformedEnv,
    Transform,
    Compose,
    FlattenObservation,
    CatTensors
)
from torchrl.data import (
    TensorSpec,
    BoundedTensorSpec,
    UnboundedContinuousTensorSpec,
    DiscreteTensorSpec,
    MultiDiscreteTensorSpec,
    CompositeSpec,
)
from .env import AgentSpec
from dataclasses import replace


def _transform_agent_spec(self: Transform, agent_spec: AgentSpec) -> AgentSpec:
    return agent_spec
Transform.transform_agent_spec = _transform_agent_spec


def _transform_agent_spec(self: Compose, agent_spec: AgentSpec) -> AgentSpec:
    for transform in self.transforms:
        agent_spec = transform.transform_agent_spec(agent_spec)
    return agent_spec
Compose.transform_agent_spec = _transform_agent_spec


def _agent_spec(self: TransformedEnv) -> AgentSpec:
    agent_spec = self.transform.transform_agent_spec(self.base_env.agent_spec)
    return {name: replace(spec, _env=self) for name, spec in agent_spec.items()}
TransformedEnv.agent_spec = property(_agent_spec)


class FromDiscreteAction(Transform):
    def __init__(
        self,
        action_key: Tuple[str] = ("agents", "action"),
        nbins: Union[int, Sequence[int]] = None,
    ):
        if nbins is None:
            nbins = 2
        super().__init__([], in_keys_inv=[action_key])
        if not isinstance(action_key, tuple):
            action_key = (action_key,)
        self.nbins = nbins
        self.action_key = action_key

    def transform_input_spec(self, input_spec: CompositeSpec) -> CompositeSpec:
        action_spec = input_spec[("full_action_spec", *self.action_key)]
        if isinstance(action_spec, BoundedTensorSpec):
            if isinstance(self.nbins, int):
                nbins = [self.nbins] * action_spec.shape[-1]
            elif len(self.nbins) == action_spec.shape[-1]:
                nbins = self.nbins
            else:
                raise ValueError(
                    "nbins must be int or list of length equal to the last dimension of action space."
                )
            self.minimum = action_spec.space.minimum.unsqueeze(-2)
            self.maximum = action_spec.space.maximum.unsqueeze(-2)
            self.mapping = torch.cartesian_prod(
                *[torch.linspace(0, 1, dim_nbins) for dim_nbins in nbins]
            ).to(action_spec.device)  # [prod(nbins), len(nbins)]
            n = self.mapping.shape[0]
            spec = DiscreteTensorSpec(
                n, shape=[*action_spec.shape[:-1], 1], device=action_spec.device
            )
        else:
            NotImplementedError("Only BoundedTensorSpec is supported.")
        input_spec[("full_action_spec", *self.action_key)] = spec
        return input_spec

    def _inv_apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        mapping = self.mapping * (self.maximum - self.minimum) + self.minimum
        action = action.unsqueeze(-1)
        action = torch.take_along_dim(mapping, action, dim=-2).squeeze(-2)
        return action


class FromMultiDiscreteAction(Transform):
    def __init__(
        self,
        action_key: Tuple[str] = ("agents", "action"),
        nbins: Union[int, Sequence[int]] = 2,
    ):
        if action_key is None:
            action_key = "action"
        super().__init__([], in_keys_inv=[action_key])
        if not isinstance(action_key, tuple):
            action_key = (action_key,)
        self.nbins = nbins
        self.action_key = action_key

    def transform_input_spec(self, input_spec: CompositeSpec) -> CompositeSpec:
        action_spec = input_spec[("full_action_spec", *self.action_key)]
        if isinstance(action_spec, BoundedTensorSpec):
            if isinstance(self.nbins, int):
                nbins = [self.nbins] * action_spec.shape[-1]
            elif len(self.nbins) == action_spec.shape[-1]:
                nbins = self.nbins
            else:
                raise ValueError(
                    "nbins must be int or list of length equal to the last dimension of action space."
                )
            spec = MultiDiscreteTensorSpec(
                nbins, shape=action_spec.shape, device=action_spec.device
            )
            self.nvec = spec.nvec.to(action_spec.device)
            self.minimum = action_spec.space.minimum
            self.maximum = action_spec.space.maximum
        else:
            NotImplementedError("Only BoundedTensorSpec is supported.")
        input_spec[("full_action_spec", *self.action_key)] = spec
        return input_spec

    def _inv_apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        action = action / (self.nvec - 1) * (self.maximum - self.minimum) + self.minimum
        return action

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return super()._inv_call(tensordict)


class DepthImageNorm(Transform):
    def __init__(
        self,
        in_keys: Sequence[str],
        min_range: float,
        max_range: float,
        inverse: bool=False
    ):
        super().__init__(in_keys=in_keys)
        self.max_range = max_range
        self.min_range = min_range
        self.inverse = inverse

    def _apply_transform(self, obs: torch.Tensor) -> None:
        obs = torch.nan_to_num(obs, posinf=self.max_range, neginf=self.min_range)
        obs = obs.clip(self.min_range, self.max_range)
        if self.inverse:
            obs = (obs - self.min_range) / (self.max_range - self.min_range)
        else:
            obs = (self.max_range - obs) / (self.max_range - self.min_range)
        return obs


def ravel_composite(
    spec: CompositeSpec, key: str, start_dim: int=-2, end_dim: int=-1
):
    r"""

    Examples:
    >>> obs_spec = CompositeSpec({
    ...     "obs_self": UnboundedContinuousTensorSpec((1, 19)),
    ...     "obs_others": UnboundedContinuousTensorSpec((3, 13)),
    ... })
    >>> spec = CompositeSpec({
            "agents": {
                "observation": obs_spec
            }
    ... })
    >>> t = ravel_composite(spec, ("agents", "observation"))

    """
    composite_spec = spec[key]
    if not isinstance(key, tuple):
        key = (key,)
    if isinstance(composite_spec, CompositeSpec):
        in_keys = [k for k in spec.keys(True, True) if k[:len(key)] == key]
        return Compose(
            FlattenObservation(start_dim, end_dim, in_keys),
            CatTensors(in_keys, out_key=key, del_keys=False)
        )
    else:
        raise TypeError
    
class ControllerWrapper(Transform):

    REGISTRY: Dict[str, "ControllerWrapper"] = {}
    action_shape: Tuple[int, ...] = None
    required_keys_inv: List[Tuple[str, ...]] = [("info", "drone_state")]

    @classmethod
    def __init_subclass__(cls, **kwargs):
        if cls.__name__ in ControllerWrapper.REGISTRY:
            raise ValueError("")
        super().__init_subclass__(**kwargs)
        ControllerWrapper.REGISTRY[cls.__name__] = cls
        ControllerWrapper.REGISTRY[cls.__name__.lower()] = cls

    def __init__(
        self,
        controller,
        cfg,
        action_key: str = ("agents", "action"),
        additional_keys_inv: List[Tuple[str, ...]] = [],
    ):
        all_keys_inv = self.required_keys_inv.copy() + (additional_keys_inv or [])
        super().__init__([], in_keys_inv=all_keys_inv)
        self.controller = controller
        self.cfg = cfg
        self.action_key = action_key
        
        # Call _post_init if it exists in the subclass
        if hasattr(self, '_post_init'):
            self._post_init()

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        action_shape = getattr(self, "action_shape", None)
        if action_shape is None:
            raise NotImplementedError("action_shape must be specified for ControllerWrapper subclasses")
        
        action_spec = input_spec[("full_action_spec", *self.action_key)]
        spec = UnboundedContinuousTensorSpec(action_spec.shape[:-1]+action_shape, device=action_spec.device)
        input_spec[("full_action_spec", *self.action_key)] = spec
        return input_spec

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError("ControllerWrapper subclasses must implement _inv_call")

class HybridControllerBase(ControllerWrapper):
    action_shape = (4,)
    required_keys_inv = [("info", "drone_state"), ("traj_stats", "future_targets"), ("traj_stats", "future_times")]

    def _get_data(self, tensordict: TensorDictBase):
        drone_state = tensordict[("info", "drone_state")][..., :13]
        future_targets = tensordict[("traj_stats", "future_targets")]
        future_times = tensordict[("traj_stats", "future_times")]
        return drone_state, future_targets, future_times

    def _process_data(self, tensordict: TensorDictBase):
        drone_state, future_targets, future_times = self._get_data(tensordict)
        if future_times.shape[-1] < 4:
            raise ValueError(f"future_times must have at least 4 elements, got {future_times.shape[-1]}")
        pos_array = future_targets[..., :3]
        time_array = future_times
        target_pos = future_targets[..., 0, :]
        return drone_state, target_pos, pos_array, time_array

    def _estimate_derivatives_difference(self, pos_array, time_array):
        # Ensure we have at least 3 points
        if pos_array.shape[-2] < 3:
            # Fallback to zero derivatives if insufficient points
            return torch.zeros_like(pos_array[..., 0, :]), torch.zeros_like(pos_array[..., 0, :])

        d1 = pos_array[..., 1, :] - pos_array[..., 0, :]
        d2 = pos_array[..., 2, :] - pos_array[..., 1, :]

        dt1 = time_array[..., 1] - time_array[..., 0]
        dt2 = time_array[..., 2] - time_array[..., 1]
        
        # Avoid division by zero
        dt1 = torch.clamp(dt1, min=1e-6)
        dt2 = torch.clamp(dt2, min=1e-6)
        
        v0 = d1 / dt1
        a0 = (d2 / dt2 - v0) / (dt1 + dt2)

        return v0, a0

    def _estimate_derivatives_polynomial(self, pos_array, time_array, poly_degree=2):
        num_points = min(pos_array.shape[-2], max(poly_degree + 4, 10))
        pos = pos_array[..., :num_points, :]
        t = time_array[..., :num_points]

        t_min = t[..., 0:1]
        t_max = t[..., -1:]
        time_scale = t_max - t_min + 1e-6
        t_norm = (t - t_min) / time_scale

        powers = torch.arange(poly_degree + 1, device=pos_array.device, dtype=pos_array.dtype)
        A = t_norm.unsqueeze(-1) ** powers

        try:
            # Try least squares with regularization
            AtA = A.transpose(-2, -1) @ A
            reg = 1e-4 * torch.eye(poly_degree + 1, device=pos_array.device, dtype=pos_array.dtype)
            AtA_reg = AtA + reg
            Atb = A.transpose(-2, -1) @ pos
            coeffs = torch.linalg.solve(AtA_reg, Atb)
        except:
            # Fallback to simple finite differences if polynomial fitting fails
            return self._estimate_derivatives_difference(pos_array, time_array)

        velocity = coeffs[..., 1, :] / time_scale if poly_degree > 0 else torch.zeros_like(pos[..., 0, :])
        acceleration = 2 * coeffs[..., 2, :] / time_scale**2 if poly_degree > 1 else torch.zeros_like(pos[..., 0, :])

        return velocity, acceleration

class OnlyPositionController(HybridControllerBase):
    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        
        drone_state, target_pos, pos_array, time_array = self._process_data(tensordict)
        target_velocity, target_acceleration = self._estimate_derivatives_polynomial(pos_array, time_array, poly_degree=2)

        cmds = self.controller(
            drone_state, 
            target_pos=target_pos,  # First future target position (absolute)
            target_vel=target_velocity,  # Estimated velocity
            target_acc=target_acceleration,  # Estimated acceleration
        )
        torch.nan_to_num_(cmds, 0.)
        tensordict.set(self.action_key, cmds)
        return tensordict

class OnlyPosControllerNoFF(HybridControllerBase):
    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        
        drone_state, target_pos, pos_array, time_array = self._process_data(tensordict)

        cmds = self.controller(
            drone_state, 
            target_pos=target_pos,  # First future target position (absolute)
        )
        torch.nan_to_num_(cmds, 0.)
        tensordict.set(self.action_key, cmds)
        return tensordict

class HybridPositionControllerVelocity(HybridControllerBase):

    def _post_init(self):
        self.neural_yaw_scale = self.cfg.task.neural_yaw_scale
        self.neural_vel_scale = self.cfg.task.neural_vel_scale

        self.scaled_init_vel_std = self.cfg.task.init_vel_std / self.neural_vel_scale
        self.scaled_init_yaw_std = self.cfg.task.init_yaw_std / self.neural_yaw_scale

        self.init_action_std = torch.as_tensor([
            self.scaled_init_vel_std, self.scaled_init_vel_std, self.scaled_init_vel_std,
            self.scaled_init_yaw_std
        ], device=self.controller.device).float()

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        drone_state, target_pos, pos_array, time_array = self._process_data(tensordict) 
        target_velocity, target_acceleration = self._estimate_derivatives_polynomial(pos_array, time_array, poly_degree=2)

        action = tensordict[self.action_key]
        target_vel_adj, target_yaw = action.split([3, 1], -1)

        cmds = self.controller(
            drone_state, 
            target_pos=target_pos,  # First future target position (absolute)
            target_vel=target_velocity + target_vel_adj * self.neural_vel_scale,  # Estimated velocity + adjustment
            target_acc=target_acceleration,  # Estimated acceleration
            target_yaw=target_yaw*self.neural_yaw_scale
        )
        torch.nan_to_num_(cmds, 0.)
        tensordict.set(self.action_key, cmds)
        return tensordict

class HybridPositionControllerAcceleration(HybridControllerBase):

    def _post_init(self):
        self.neural_yaw_scale = self.cfg.task.neural_yaw_scale
        self.neural_acc_scale = self.cfg.task.neural_acc_scale

        self.scaled_init_acc_std = self.cfg.task.init_acc_std / self.neural_acc_scale
        self.scaled_init_yaw_std = self.cfg.task.init_yaw_std / self.neural_yaw_scale

        self.init_action_std = torch.as_tensor([
            self.scaled_init_acc_std, self.scaled_init_acc_std, self.scaled_init_acc_std,
            self.scaled_init_yaw_std
        ], device=self.controller.device).float()

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        drone_state, target_pos, pos_array, time_array = self._process_data(tensordict) 
        target_velocity, target_acceleration = self._estimate_derivatives_polynomial(pos_array, time_array, poly_degree=2)

        action = tensordict[self.action_key]
        target_acc_adj, target_yaw = action.split([3, 1], -1)

        cmds = self.controller(
            drone_state, 
            target_pos=target_pos,  # First future target position (absolute)
            target_vel=target_velocity,  # Estimated velocity
            target_acc=target_acceleration + target_acc_adj * self.neural_acc_scale,  # Estimated acceleration
            target_yaw=target_yaw*self.neural_yaw_scale
        )
        torch.nan_to_num_(cmds, 0.)
        tensordict.set(self.action_key, cmds)
        return tensordict

class HybridPositionControllerYaw(HybridControllerBase):
    action_shape = (1,)


    def _post_init(self):
        self.neural_yaw_scale = self.cfg.task.neural_yaw_scale

        self.scaled_init_yaw_std = self.cfg.task.init_yaw_std / self.neural_yaw_scale

        self.init_action_std = torch.as_tensor([
            self.scaled_init_yaw_std
        ], device=self.controller.device).float()

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        drone_state, target_pos, pos_array, time_array = self._process_data(tensordict) 
        target_velocity, target_acceleration = self._estimate_derivatives_polynomial(pos_array, time_array, poly_degree=2)

        action = tensordict[self.action_key]
        target_yaw = action

        cmds = self.controller(
            drone_state, 
            target_pos=target_pos,  # First future target position (absolute)
            target_vel=target_velocity,  # Estimated velocity
            target_acc=target_acceleration,  # Estimated acceleration
            target_yaw=target_yaw*self.neural_yaw_scale
        )
        torch.nan_to_num_(cmds, 0.)
        tensordict.set(self.action_key, cmds)
        return tensordict

class HybridPositionControllerMotor(HybridControllerBase):
    
    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        drone_state, target_pos, pos_array, time_array = self._process_data(tensordict) 
        target_velocity, target_acceleration = self._estimate_derivatives_polynomial(pos_array, time_array, poly_degree=2)

        action = tensordict[self.action_key]
        _, dummy_yaw = action.split([3, 1], -1)
        target_yaw = torch.zeros_like(dummy_yaw)

        cmds = action + self.controller(
            drone_state, 
            target_pos=target_pos,  # First future target position (absolute)
            target_vel=target_velocity,  # Estimated velocity
            target_acc=target_acceleration,  # Estimated acceleration
            target_yaw=None
        )
        torch.nan_to_num_(cmds, 0.)
        tensordict.set(self.action_key, cmds)
        return tensordict

class PositionController(ControllerWrapper):
    action_shape = (7,)

    def _post_init(self):
        self.neural_pos_scale = self.cfg.task.neural_pos_scale
        self.neural_vel_scale = self.cfg.task.neural_vel_scale
        self.neural_yaw_scale = self.cfg.task.neural_yaw_scale

        self.scaled_init_pos_std = self.cfg.task.init_pos_std / self.neural_pos_scale
        self.scaled_init_vel_std = self.cfg.task.init_vel_std / self.neural_vel_scale
        self.scaled_init_yaw_std = self.cfg.task.init_yaw_std / self.neural_yaw_scale

        self.init_action_std = torch.as_tensor([
            self.scaled_init_pos_std, self.scaled_init_pos_std, self.scaled_init_pos_std,
            self.scaled_init_vel_std, self.scaled_init_vel_std, self.scaled_init_vel_std,
            self.scaled_init_yaw_std
        ], device=self.controller.device).float()

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        drone_state = tensordict[("info", "drone_state")][..., :13]
        action = tensordict[self.action_key]
        target_pos, target_vel, target_yaw = action.split([3, 3, 1], -1)
        cmds = self.controller(
            drone_state, 
            target_pos=target_pos * self.neural_pos_scale + drone_state[..., :3], # we should send relative position
            target_vel=target_vel * self.neural_vel_scale, 
            target_yaw=target_yaw*self.neural_yaw_scale
        )
        torch.nan_to_num_(cmds, 0.)
        tensordict.set(self.action_key, cmds)
        return tensordict
    
class VelocityController(ControllerWrapper):
    action_shape = (4,)

    def _post_init(self):
        self.neural_vel_scale = self.cfg.task.neural_vel_scale
        self.neural_yaw_scale = self.cfg.task.neural_yaw_scale

        self.scaled_init_vel_std = self.cfg.task.init_vel_std / self.neural_vel_scale
        self.scaled_init_yaw_std = self.cfg.task.init_yaw_std / self.neural_yaw_scale

        self.init_action_std = torch.as_tensor([
            self.scaled_init_vel_std, self.scaled_init_vel_std, self.scaled_init_vel_std,
            self.scaled_init_yaw_std
        ], device=self.controller.device).float()

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        drone_state = tensordict[("info", "drone_state")][..., :13]
        action = tensordict[self.action_key]
        target_vel, target_yaw = action.split([3, 1], -1)
        cmds = self.controller(
            drone_state, 
            target_vel=target_vel * self.neural_vel_scale, 
            target_yaw=target_yaw * self.neural_yaw_scale
        )
        torch.nan_to_num_(cmds, 0.)
        tensordict.set(self.action_key, cmds)
        return tensordict
    
class RateController(ControllerWrapper):
    action_shape = (4,)

    def _post_init(self):
        self.neural_body_rate_scale = self.cfg.task.neural_body_rate_scale

        self.scaled_init_body_rate_std = self.cfg.task.init_body_rate_std / self.neural_body_rate_scale

        self.init_action_std = torch.as_tensor([
            self.scaled_init_body_rate_std, self.scaled_init_body_rate_std, self.scaled_init_body_rate_std,
            1.0
        ], device=self.controller.device).float()

    def __init__(
        self, 
        controller, 
        action_key: str = ("agents", "action"), 
    ):
        super().__init__(controller, action_key)
        self.max_thrust = self.controller.max_thrusts.sum(-1)

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        drone_state = tensordict[("info", "drone_state")][..., :13]
        action = tensordict[self.action_key]
        target_rate, target_thrust = action.split([3, 1], -1)
        target_thrust = ((target_thrust + 1) / 2).clip(0.) * self.max_thrust
        cmds = self.controller(
            drone_state,
            target_rate=target_rate * self.neural_body_rate_scale,
            target_thrust=target_thrust
        )
        torch.nan_to_num_(cmds, 0.)
        tensordict.set(self.action_key, cmds)
        return tensordict


class AttitudeController(ControllerWrapper):
    action_shape = (4,)

    def __init__(
        self, 
        controller, 
        action_key: str = ("agents", "action"), 
    ):
        super().__init__(controller, action_key)
        self.max_thrust = self.controller.max_thrusts.sum(-1)

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        drone_state = tensordict[("info", "drone_state")][..., :13]
        action = tensordict[self.action_key]
        target_thrust, target_yaw_rate, target_roll, target_pitch = action.split(1, dim=-1)
        cmds = self.controller(
            drone_state,
            target_thrust=((target_thrust+1)/2).clip(0.) * self.max_thrust,
            target_yaw_rate=target_yaw_rate * torch.pi,
            target_roll=target_roll * torch.pi,
            target_pitch=target_pitch * torch.pi
        )
        torch.nan_to_num_(cmds, 0.)
        tensordict.set(self.action_key, cmds)
        return tensordict


class History(Transform):
    def __init__(
        self,
        in_keys: Sequence[str],
        out_keys: Sequence[str]=None,
        steps: int = 32,
    ):
        if out_keys is None:
            out_keys = [
                f"{key}_h" if isinstance(key, str) else key[:-1] + (f"{key[-1]}_h",)
                for key in in_keys
            ]
        if any(key in in_keys for key in out_keys):
            raise ValueError
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.steps = steps

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            is_tuple = isinstance(in_key, tuple)
            if in_key in observation_spec.keys(include_nested=is_tuple):
                spec = observation_spec[in_key]
                spec = spec.unsqueeze(-1).expand(*spec.shape, self.steps)
                observation_spec[out_key] = spec
        return observation_spec

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            item = tensordict.get(in_key)
            item_history = tensordict.get(out_key)
            item_history[..., :-1] = item_history[..., 1:]
            item_history[..., -1] = item
        return tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            item = tensordict.get(in_key)
            item_history = tensordict.get(out_key).clone()
            item_history[..., :-1] = item_history[..., 1:]
            item_history[..., -1] = item
            tensordict.set(("next", out_key), item_history)
        return tensordict

    def reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        _reset = tensordict.get("_reset", None)
        if _reset is None:
            _reset = torch.ones(tensordict.batch_size, dtype=bool, device=tensordict.device)
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            if out_key not in tensordict.keys(True, True):
                item = tensordict.get(in_key)
                item_history = (
                    item.unsqueeze(-1)
                    .expand(*item.shape, self.steps)
                    .clone()
                    .zero_()
                )
                tensordict.set(out_key, item_history)
            else:
                item_history = tensordict.get(out_key)
                item_history[_reset] = 0.
        return tensordict

