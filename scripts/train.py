import logging
import os
import time

import hydra
import torch
import wandb

from torch.func import vmap
from tqdm import tqdm
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from torchrl.data import CompositeSpec
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl import SyncDataCollector
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
    AttitudeController,
    RateController,
)
from omni_drones.utils.wandb import init_wandb
from omni_drones.utils.torchrl import RenderCallback, EpisodeStats
from omni_drones.learning import ALGOS
# from omni_drones.envs.isaac_env import IsaacEnv
from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose

def evaluate(
        env_manager,
        eval_configs, 
        policy, 
        seed,
        current_frames,
        wandb_run,
        hydra_dir,
        exploration_type: ExplorationType = ExplorationType.MODE,
        render: bool = True
    ):
    from omni_drones.utils.environment_manager import evaluate_with_manager

    for eval_name, eval_diff in eval_configs.items():
        info = {"env_frames": current_frames}
        eval_info, traj_data = evaluate_with_manager(env_manager, eval_name, eval_diff, policy, seed=seed, exploration_type=exploration_type, render=render)
        info.update(eval_info)
        wandb_run.log(info, commit=False)
        save_traj_data(traj_data, eval_name, seed, current_frames, hydra_dir)
    
    env, base_env = env_manager.get_training_env()
    
    base_env.train()
    env.train()
    
    return env, base_env

def save_traj_data(traj_data, eval_name, seed, current_frames, hydra_dir):
    import pickle
    import os

    # Create directory for saving trajectory data
    save_dir = os.path.join(hydra_dir, "trajectories")
    os.makedirs(save_dir, exist_ok=True)
    
    # Create directory for current eval
    eval_dir = os.path.join(save_dir, eval_name)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Save trajectory data as pickle file
    traj_file = eval_dir + f"/traj_data_{eval_name}_seed{seed}_frames{current_frames}.pkl"
    with open(traj_file, 'wb') as f:
        pickle.dump(traj_data, f)
    logging.info(f"Saved trajectory data to {traj_file}")
    
    # Add pickle file to wandb
    wandb.save(traj_file, base_path=hydra_dir)
    
def run_training_segment(
        env_manager, 
        policy, 
        cfg, 
        episode_stats, 
        hydra_dir,
        wandb_run, 
        start_iteration=0,
        render_during_minor_eval: bool = True
        ):
    
    from omni_drones.utils.environment_manager import evaluate_current_env

    env, base_env = env_manager.get_training_env()

    env.train()
    base_env.train()

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    major_eval_interval = cfg.get("major_eval_interval", -1)
    eval_interval = cfg.get("eval_interval", -1)
    save_interval = cfg.get("save_interval", -1)

    start_frame = start_iteration * frames_per_batch

    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

    iterations_until_major_eval = float('inf')
    if major_eval_interval > 0:
        iterations_until_major_eval = min(iterations_until_major_eval, major_eval_interval - start_iteration % major_eval_interval)
    if max_iters > 0:
        iterations_until_major_eval = min(iterations_until_major_eval, (max_iters - start_iteration))
    if total_frames > 0:
        iterations_until_major_eval = min(iterations_until_major_eval, (total_frames - start_frame) // frames_per_batch)

    assert iterations_until_major_eval > 0, "No iterations until major eval"
    iterations_until_major_eval = int(iterations_until_major_eval)

    pbar = tqdm(total=iterations_until_major_eval)

    for i, data in enumerate(collector):

        if i >= iterations_until_major_eval:
            break

        info = {"env_frames": start_frame + collector._frames, "rollout_fps": collector._fps}
        episode_stats.add(data.to_tensordict())

        if len(episode_stats) >= base_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

        info.update(policy.train_op(data.to_tensordict()))

        if eval_interval > 0 and (i + start_iteration) % eval_interval == 0 and (i + start_iteration) > 0:

            try:
                torch.cuda.empty_cache()
                logging.info("CUDA cache cleared")
            except Exception as e:
                logging.warning(f"Error clearing CUDA cache: {e}")

            eval_info, traj_data = evaluate_current_env(env_manager, policy, cfg.seed, ExplorationType.MODE, render=render_during_minor_eval)
            info.update(eval_info)
            save_traj_data(traj_data, "training", cfg.seed, start_frame + collector._frames, hydra_dir)
       
        if save_interval > 0 and (i + start_iteration) % save_interval == 0 and (i + start_iteration) > 0:
            try:
                ckpt_path = os.path.join(hydra_dir, f"checkpoint_{start_frame + collector._frames}.pt")
                torch.save(policy.state_dict(), ckpt_path)
                logging.info(f"Saved checkpoint to {str(ckpt_path)}")
            except AttributeError:
                logging.warning(f"Policy {policy} does not implement `.state_dict()`")

        wandb_run.log(info)
        print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))

        pbar.set_postfix({"rollout_fps": collector._fps, "frames": start_frame + collector._frames})
        pbar.update(1)

    pbar.close()
    final_frames = start_frame + collector._frames
    final_fps = collector._fps
    final_iteration = start_iteration + i

    try:
        collector.shutdown()
    except AttributeError:
        pass

    del collector
    import gc
    gc.collect()
    time.sleep(0.1)

    return final_frames, final_fps, final_iteration

def create_env(cfg):
    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    transforms = [InitTracker()]

    # a CompositeSpec is by default processed by a entity-based encoder
    # ravel it to use a MLP encoder instead
    if cfg.task.get("ravel_obs", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation"))
        transforms.append(transform)
    if cfg.task.get("ravel_obs_central", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation_central"))
        transforms.append(transform)

    # optionally discretize the action space or use a controller
    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromMultiDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromDiscreteAction(nbins=nbins)
            transforms.append(transform)
        else:
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    return env, base_env

@hydra.main(version_base=None, config_path=".", config_name="train")
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
    major_eval_configs = OmegaConf.create({'training': OmegaConf.create({})})
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

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    major_eval_interval = cfg.get("major_eval_interval", -1)
    render_during_major_eval = cfg.get("render_during_major_eval", True)
    render_during_minor_eval = cfg.get("render_during_minor_eval", True)

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)


    current_frames = 0
    current_iteration = 0

    logging.info(f"Eval at {current_frames} steps (iteration {current_iteration}).")
    _, _ = evaluate(env_manager, major_eval_configs, policy, cfg.seed, current_frames, run, hydra_dir, render=render_during_major_eval)


    def should_continue():
        if total_frames > 0 and current_frames >= total_frames:
            return False
        if max_iters > 0 and current_iteration >= max_iters:
            return False
        return True

    while should_continue():
        current_frames, fps, current_iteration = run_training_segment(
                env_manager, 
                policy, 
                cfg, 
                episode_stats, 
                hydra_dir, 
                run, 
                current_iteration,
                render=render_during_minor_eval
            )
                
        if should_continue() and major_eval_interval > 0:
            logging.info(f"Eval at {current_frames} steps (iteration {current_iteration}).")
            _, _ = evaluate(env_manager, major_eval_configs, policy, cfg.seed, current_frames, run, hydra_dir, render=render_during_major_eval)

    logging.info(f"Final Eval at {current_frames} steps (iteration {current_iteration}).")
    _, _ = evaluate(env_manager, major_eval_configs, policy, cfg.seed, current_frames, run, hydra_dir, render=render_during_major_eval)
    run.log({'env_frames': current_frames})

    try:
        ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
        torch.save(policy.state_dict(), ckpt_path)

        model_artifact = wandb.Artifact(
            f"{cfg.task.name}-{cfg.algo.name.lower()}",
            type="model",
            description=f"{cfg.task.name}-{cfg.algo.name.lower()}",
            metadata=dict(cfg))

        model_artifact.add_file(ckpt_path)
        wandb.save(ckpt_path)
        run.log_artifact(model_artifact)

        logging.info(f"Saved checkpoint to {str(ckpt_path)}")
    except AttributeError:
        logging.warning(f"Policy {policy} does not implement `.state_dict()`")

    wandb.finish()

    simulation_app.close()


if __name__ == "__main__":
    main()
