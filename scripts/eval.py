import logging
import os
import time

import hydra
import torch
import wandb
import numpy as np

from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from omni_drones.utils.wandb import init_wandb
from setproctitle import setproctitle
from omni_drones.learning import ALGOS


from train import evaluate, create_env


@hydra.main(version_base=None, config_path=".", config_name="eval")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    hydra_dir = hydra.core.hydra_config.HydraConfig.get().run.dir
    run = init_wandb(cfg, save_dir=hydra_dir)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.utils.environment_manager import EvalEnvironmentManager


    env, base_env = create_env(cfg)

    env_manager = EvalEnvironmentManager(cfg, create_env, base_env.device, base_env, env)
    major_eval_configs = OmegaConf.create({'base_eval': OmegaConf.create({})})
    more_configs = cfg.get('major_eval_configs', {})
    if more_configs:
        major_eval_configs.update(more_configs)

    try:
        policy = ALGOS[cfg.algo.name.lower()](
            cfg.algo,
            env.observation_spec,
            env.action_spec,
            env.reward_spec,
            device=base_env.device
        )
    except KeyError:
        raise NotImplementedError(f"Unknown algorithm: {cfg.algo.name}")
    
     # Load the saved policy weights
    if cfg.checkpoint_path is None:
        raise ValueError("checkpoint_path must be specified in config for evaluation")
    
    state_dict = torch.load(cfg.checkpoint_path)
    policy.load_state_dict(state_dict)
    policy.eval()  # Set policy to evaluation mode

    render_during_major_eval=cfg.get("render_during_major_eval", True)

    seed = cfg.get('seed', 0)
    if isinstance(seed, int):
        num_episodes = cfg.get('num_episodes', 1)
        seeds = np.arange(seed, seed + num_episodes)
    else:
        seeds = seed

    for i, seed in enumerate(seeds):
        info = {"episode": i}
        _, _ = evaluate(env_manager, major_eval_configs, policy, seed, 0, run, hydra_dir, render=render_during_major_eval)
        run.log(info)

    time.sleep(5)
    try:
        wandb.finish()
    except:
        pass
    simulation_app.close()

if __name__ == "__main__":
    main()