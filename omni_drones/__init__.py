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


import os

import torch
from isaacsim import SimulationApp
from tensordict import TensorDict

CONFIG_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir, "cfg")


def init_simulation_app(cfg):

    # No display in headless mode, which avoids the xcb error
    if cfg["headless"]:
        os.environ["DISPLAY"] = ""
    # launch the simulator
    config = {"headless": cfg["headless"], "anti_aliasing": 1}
    # load cheaper kit config in headless
    # if cfg.headless:
    #     app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
    # else:
    #     app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"
    # app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"
    app_experience = ""
    simulation_app = SimulationApp(config, experience=app_experience)
    # simulation_app = SimulationApp(config)


    # Add Isaac assets to the asset registry
    add_isaac_assets()


    return simulation_app

def add_isaac_assets():
    import carb
    settings = carb.settings.get_settings()

    cloud_asset_root = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5"
    settings.set("/persistent/isaac/asset_root/default", cloud_asset_root)
    
    # Also set the cloud setting (backup)
    settings.set("/persistent/isaac/asset_root/cloud", cloud_asset_root)


def _get_shapes(self: TensorDict):
    return {
        k: v.shape if isinstance(v, torch.Tensor) else v.shapes for k, v in self.items()
    }


def _get_devices(self: TensorDict):
    return {
        k: v.device if isinstance(v, torch.Tensor) else v.devices
        for k, v in self.items()
    }


TensorDict.shapes = property(_get_shapes)
TensorDict.devices = property(_get_devices)
