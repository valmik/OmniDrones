import abc
import torch
import torch.nn as nn
from torchrl.data import TensorSpec
from typing import Dict

class ControllerBase(nn.Module):

    action_spec: TensorSpec
    REGISTRY: Dict[str, "ControllerBase"] = {}

    @classmethod
    def __init_subclass__(cls, **kwargs):
        if cls.__name__ in ControllerBase.REGISTRY:
            raise ValueError("")
        super().__init_subclass__(**kwargs)
        ControllerBase.REGISTRY[cls.__name__] = cls
        ControllerBase.REGISTRY[cls.__name__.lower()] = cls

    @abc.abstractmethod
    def compute(self, *args, **kwargs) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def process_rl_actions(self, actions) -> torch.Tensor:
        ...

    @property
    def device(self) -> torch.device:
        """Return the device where this module's parameters live.

        This mirrors common practice in PyTorch where moving a module with
        ``.to(device)`` migrates parameters/buffers but doesn't set an explicit
        ``.device`` attribute. Providing this property ensures downstream code
        can reliably query the controller's device.
        """
        try:
            return next(self.parameters()).device
        except StopIteration:
            # Fallback if the module has no parameters: check buffers or default to CPU
            for buffer in self.buffers():
                return buffer.device
            return torch.device("cpu")
