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

import argparse
import sys
import torch
import optuna
from omegaconf import OmegaConf
import hydra
from omni_drones import init_simulation_app
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose
from omni_drones.learning import ALGOS

# Global environment variable (will be set in main)
env = None
base_env = None
policy = None
episode_steps = None


def modify_controller_gains(controller, trial, drone_params):
    """
    Modify the Lee controller gains based on Optuna trial suggestions.
    
    Args:
        controller: LeePositionController instance
        trial: Optuna trial object
        drone_params: Drone parameters dict containing inertia information
    """
    # Suggest position gains (x, y, z)
    pos_gain_x = trial.suggest_float("pos_gain_x", 0.1, 20.0)
    pos_gain_y = trial.suggest_float("pos_gain_y", 0.1, 20.0)
    pos_gain_z = trial.suggest_float("pos_gain_z", 0.1, 20.0)
    
    # Suggest velocity gains (x, y, z)
    vel_gain_x = trial.suggest_float("vel_gain_x", 0.1, 15.0)
    vel_gain_y = trial.suggest_float("vel_gain_y", 0.1, 15.0)
    vel_gain_z = trial.suggest_float("vel_gain_z", 0.1, 15.0)
    
    # Suggest attitude gains (x, y, z)
    att_gain_x = trial.suggest_float("att_gain_x", 0.01, 10.0)
    att_gain_y = trial.suggest_float("att_gain_y", 0.01, 10.0)
    att_gain_z = trial.suggest_float("att_gain_z", 0.001, 1.0)
    
    # Suggest angular rate gains (x, y, z)
    ang_rate_gain_x = trial.suggest_float("ang_rate_gain_x", 0.01, 5.0)
    ang_rate_gain_y = trial.suggest_float("ang_rate_gain_y", 0.01, 5.0)
    ang_rate_gain_z = trial.suggest_float("ang_rate_gain_z", 0.001, 0.5)
    
    # Set position and velocity gains directly
    controller.pos_gain.data = torch.tensor([pos_gain_x, pos_gain_y, pos_gain_z], 
                                              device=controller.pos_gain.device, 
                                              dtype=controller.pos_gain.dtype)
    controller.vel_gain.data = torch.tensor([vel_gain_x, vel_gain_y, vel_gain_z], 
                                             device=controller.vel_gain.device, 
                                             dtype=controller.vel_gain.dtype)
    
    # For attitude and angular rate gains, need to apply inertia transformation
    # The controller applies: gain @ I[:3, :3].inverse()
    # So we need to do the same transformation
    inertia = drone_params["inertia"]
    I = torch.diag_embed(
        torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"], 1], 
                     device=controller.attitute_gain.device)
    )
    I_inv = I[:3, :3].inverse()
    
    att_gain_raw = torch.tensor([att_gain_x, att_gain_y, att_gain_z], 
                                device=controller.attitute_gain.device,
                                dtype=controller.attitute_gain.dtype)
    controller.attitute_gain.data = att_gain_raw @ I_inv
    
    ang_rate_gain_raw = torch.tensor([ang_rate_gain_x, ang_rate_gain_y, ang_rate_gain_z], 
                                     device=controller.ang_rate_gain.device,
                                     dtype=controller.ang_rate_gain.dtype)
    controller.ang_rate_gain.data = ang_rate_gain_raw @ I_inv


def eval_trial(trial):
    """
    Evaluate a single Optuna trial by running episodes with suggested gains.
    
    Returns:
        Average reward (Optuna maximizes, so we return reward directly)
    """
    global env, base_env, policy, episode_steps
    
    # Modify controller gains
    controller = base_env.controller
    modify_controller_gains(controller, trial, base_env.drone.params)
    
    # Ensure environment is in train mode for optimization
    env.train()
    base_env.train()
    
    # Run rollout - this handles stepping, auto-reset, and trajectory collection
    try:
        trajs = env.rollout(
            max_steps=episode_steps,
            policy=policy,
            auto_reset=True,
            break_when_any_done=False,
            return_contiguous=False,
        )
    except Exception as e:
        print(f"Error in rollout: {e}")
        import pdb; pdb.set_trace()
        trial.set_user_attr("error", f"Environment rollout error: {str(e)}")
        return float('-inf')  # Bad trial (return negative infinity since we maximize)
    
    # Extract rewards from trajectory
    rewards = trajs.get(("next", "reward"), torch.zeros(*trajs.batch_size, 1, device=base_env.device))
    
    # Calculate total reward and average
    total_reward = rewards.sum().item()
    num_steps = rewards.numel()
    
    if num_steps > 0:
        avg_reward = total_reward / num_steps
    else:
        avg_reward = float('-inf')
        trial.set_user_attr("warning", "No steps completed")
    
    # Count episodes (number of done flags)
    done = trajs.get(("next", "done"), torch.zeros(*trajs.batch_size, 1, dtype=torch.bool, device=base_env.device))
    num_episodes = done.sum().item()
    
    # Log additional metrics
    trial.set_user_attr("num_episodes", num_episodes)
    trial.set_user_attr("steps", num_steps)
    trial.set_user_attr("total_reward", total_reward)
    trial.set_user_attr("avg_reward", avg_reward)
    
    # Prune trial if reward is too low
    if avg_reward < -100.0:  # Threshold for very poor performance
        raise optuna.TrialPruned(f"Reward too low: {avg_reward}")
    
    # Return average reward (Optuna maximizes)
    return avg_reward


@hydra.main(version_base=None, config_path=".", config_name="tune_lee_controller")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    simulation_app = init_simulation_app(cfg)
    global env, base_env, episode_steps, policy
    
    print(f"Task: {cfg.task.name}")
    print(f"Number of environments: {cfg.env.num_envs}")
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs.isaac_env import IsaacEnv
    
    # Create environment
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)
    episode_steps = cfg.episode_steps

    # Create transforms with OnlyPositionController to simplify eval_trial
    from omni_drones.utils.torchrl.transforms import OnlyPositionController
    transforms = [InitTracker()]
    # Add OnlyPositionController so we can use env.step() directly
    controller_transform = OnlyPositionController(base_env.controller, cfg).to(base_env.device)
    transforms.append(controller_transform)
    env = TransformedEnv(base_env, Compose(*transforms)).train()

    try:
        policy = ALGOS[cfg.algo.name.lower()](
            cfg.algo, 
            env.observation_spec, 
            env.action_spec, 
            env.reward_spec, 
            device=base_env.device
        )
        checkpoint_path = cfg.get("checkpoint_path", None)
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location=base_env.device)
            policy.load_state_dict(state_dict)
            print(f"Loaded checkpoint from {checkpoint_path}")
    except Exception as e:
        print(f"Error in creating policy: {e}. Using dummy policy.")
        cfg.algo = OmegaConf.create({"name": "dummy"})
        policy = ALGOS["dummy"](
            cfg.algo, 
            env.observation_spec, 
            env.action_spec,
            env.reward_spec,
            device=base_env.device
        )
    
    
    if cfg.seed is not None:
        env.set_seed(cfg.seed)
    
    # Verify controller is LeePositionController
    if not hasattr(base_env.controller, 'pos_gain'):
        raise ValueError("Controller is not a LeePositionController. Expected LeePositionController.")
    
    # Create Optuna study
    # Storage: SQLite database file in current working directory
    # - Stores all trial data (parameters, objective values, states, user attributes)
    # - load_if_exists=True: resumes from existing trials if database exists
    # - Database file: database_lee_tuning.sqlite3 (can be viewed with SQLite tools)
    study_name = f"Lee Controller Tuning: {cfg.task.name}"
    
    # Configure sampler for better optimization
    # TPE (Tree-structured Parzen Estimator) is the default and works well
    # n_startup_trials: number of random trials before TPE kicks in (default: 10)
    # n_ei_candidates: number of candidates for expected improvement (default: 24)
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=10,  # Random exploration for first 10 trials
        n_ei_candidates=24,   # Number of candidates to evaluate
        seed=cfg.seed if cfg.seed is not None else None,  # For reproducibility
    )
    
    study = optuna.create_study(
        direction="maximize",  # We maximize reward
        study_name=study_name,
        storage="sqlite:///database_lee_tuning.sqlite3",
        load_if_exists=True,
        sampler=sampler,
    )
    
    print(f"\nStarting Optuna optimization...")
    print(f"Study name: {study_name}")
    print(f"Number of trials: {cfg.num_trials}")
    print(f"Steps per trial: {cfg.episode_steps}\n")
    
    # Run optimization
    try:
        study.optimize(eval_trial, n_trials=cfg.num_trials, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    
    # Print results
    print("\n" + "="*50)
    print("Optimization Results")
    print("="*50)
    print(f"Number of finished trials: {len(study.trials)}")
    
    if len(study.trials) > 0:
        print("\nBest trial:")
        trial = study.best_trial
        print(f"  Value (average reward): {trial.value}")
        print(f"  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        
        print("\nBest gains (for YAML file):")
        print(f"  position_gain: [{trial.params['pos_gain_x']:.3f}, {trial.params['pos_gain_y']:.3f}, {trial.params['pos_gain_z']:.3f}]")
        print(f"  velocity_gain: [{trial.params['vel_gain_x']:.3f}, {trial.params['vel_gain_y']:.3f}, {trial.params['vel_gain_z']:.3f}]")
        print(f"  attitude_gain: [{trial.params['att_gain_x']:.3f}, {trial.params['att_gain_y']:.3f}, {trial.params['att_gain_z']:.3f}]")
        print(f"  angular_rate_gain: [{trial.params['ang_rate_gain_x']:.3f}, {trial.params['ang_rate_gain_y']:.3f}, {trial.params['ang_rate_gain_z']:.3f}]")
        
        print("\nSummary DataFrame:")
        print(study.trials_dataframe())
    
    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
