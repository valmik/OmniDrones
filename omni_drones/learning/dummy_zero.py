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


import torch
from torchrl.data import CompositeSpec, TensorSpec
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

class DummyZeroPolicy(TensorDictModuleBase):

    def __init__(
        self,
        cfg,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.n_agents, self.action_dim = action_spec.shape[-2:]
        self.reward_spec = reward_spec

    def __call__(self, tensordict: TensorDict):
        # Get batch shape from input tensordict
        batch_shape = tensordict.batch_size
        
        # Create zero actions with shape (*batch_shape, n_agents, action_dim)
        zero_actions = torch.zeros(
            *batch_shape, self.n_agents, self.action_dim,
            device=self.device,
            dtype=torch.float32
        )
        tensordict.set(("agents", "action"), zero_actions)
        
        # Create zero state values with shape matching reward_spec
        # state_value typically has shape (*batch_shape, *reward_spec.shape[-2:])
        state_value_shape = (*batch_shape, *self.reward_spec.shape[-2:])
        zero_state_value = torch.zeros(
            *state_value_shape,
            device=self.device,
            dtype=torch.float32
        )
        tensordict.set("state_value", zero_state_value)
        
        return tensordict

    def train_op(self, tensordict: TensorDict):
        # Return empty dict as no-op for training
        return {}

