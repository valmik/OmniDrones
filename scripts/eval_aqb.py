"""
Evaluation script to run OmniDrone-trained models on AdaptiveQuadBench test suite.

This script loads a trained OmniDrone model and uses OmniDroneBridge to run it
on AdaptiveQuadBench evaluation scenarios.
"""

import logging
import torch
import numpy as np

import hydra
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path=".", config_name="eval_aqb")
def main(cfg):
    """
    Main evaluation function.
    
    Loads OmniDrone model and transform, extracts quad_params from OmniDrone,
    patches them into AdaptiveQuadBench, creates OmniDroneBridge,
    and runs AdaptiveQuadBench evaluation.
    """
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    cfg["headless"] = True
    
    # Import init_simulation_app (minimal import to avoid conflicts)
    from omni_drones import init_simulation_app
    
    # Initialize OmniDrone simulation app FIRST (before importing anything else)
    # This prevents segfaults from module loading conflicts
    simulation_app = init_simulation_app(cfg)
    
    # Now import other OmniDrone components AFTER simulation app is initialized
    from omni_drones.learning import ALGOS
    from train import create_env
    
    print(OmegaConf.to_yaml(cfg))
    
    # Create OmniDrone environment to get policy, transform, and drone
    env, base_env = create_env(cfg)
    
    # Create policy
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
    
    # Load checkpoint
    if cfg.checkpoint_path is None:
        raise ValueError("checkpoint_path must be specified in config for evaluation")
    
    state_dict = torch.load(cfg.checkpoint_path, map_location=base_env.device)
    policy.load_state_dict(state_dict)
    policy.eval()
    
    # Extract transform from environment if it exists
    transform = None
    if hasattr(env, 'transform') and env.transform is not None:
        # Find controller transform in the transform chain
        transforms = getattr(env.transform, 'transforms', [])
        for t in transforms:
            # Check if this is a controller transform
            if hasattr(t, 'controller') or hasattr(t, '_inv_call'):
                # Check if it's a ControllerWrapper
                from omni_drones.utils.torchrl.transforms import ControllerWrapper
                if isinstance(t, ControllerWrapper):
                    transform = t
                    break

    from aqb.controller import OmniDroneBridge
    from aqb.experiments.config_manager import ExperimentConfig
    from aqb.experiments.experiment_runner import ExperimentRunner
    from aqb.experiments.config_manager import parse_experiment_args
    from aqb.quad_param.quadrotor import quad_params as default_quad_params
    import aqb.quad_param.quadrotor as quadrotor_module
    
    
    # Extract quad_params from OmniDrone drone
    # This will be used by the experiment runner instead of the default quad_params
    quad_params = _extract_quad_params_from_omni_drone(base_env.drone)
    
    # Patch quad_params into AdaptiveQuadBench module
    # This way experiment_runner will use our extracted params
    quadrotor_module.quad_params = quad_params
    
    # Create bridge controller (will be created per trial by switch_controller)
    # We store the components needed to create it
    bridge_components = {
        'policy': policy,
        'transform': transform,
        'device': str(base_env.device),
        'cfg': cfg
    }
    
    # Create a custom switch_controller function for AdaptiveQuadBench
    def switch_controller(controller_type, quad_params):
        if controller_type == 'omnidrone':
            # Create bridge with the current quad_params (which may have been modified by experiment runner)
            bridge = OmniDroneBridge(
                vehicle_params=quad_params,
                policy=bridge_components['policy'],
                transform=bridge_components['transform'],
                device=bridge_components['device'],
                cfg=bridge_components['cfg']
            )
            return bridge
        else:
            raise ValueError(f"Controller type {controller_type} not supported. Use 'omnidrone' for OmniDroneBridge.")
    
    # Parse AdaptiveQuadBench experiment arguments
    # These can be passed via command line or set in the OmniDrone config
    if hasattr(cfg, 'aqb_args') and cfg.aqb_args:
        # Use config from OmniDrone config file
        aqb_args = _create_aqb_args_from_config(cfg.aqb_args)
        aqb_config = ExperimentConfig.from_args(aqb_args)
    else:
        # Use command line arguments (if running from command line)
        # Note: This will fail if no command line args are provided
        # In that case, you should set aqb_args in the OmniDrone config
        try:
            aqb_config = parse_experiment_args()
        except SystemExit:
            # If no args provided, create a default config
            logging.warning("No AQB arguments provided. Using defaults. Set aqb_args in config for custom settings.")
            import argparse
            default_args = argparse.Namespace()
            default_args.controller = ['omnidrone']
            default_args.experiment = 'no'
            default_args.num_trials = 100
            default_args.seed = 42
            default_args.save_trials = False
            default_args.serial = False
            default_args.vis = False
            default_args.when2fail = False
            default_args.max_intensity = 10.0
            default_args.intensity_step = 1.0
            default_args.trajectory = 'random'
            default_args.delay_margin = False
            aqb_config = ExperimentConfig.from_args(default_args)
    
    # Override controller types to use omnidrone
    aqb_config.controller_types = ['omnidrone']
    
    # Create and run experiment runner
    runner = ExperimentRunner(aqb_config, switch_controller)
    runner.run()
    
    # Close OmniDrone simulation app
    simulation_app.close()


def _create_aqb_args_from_config(cfg_args):
    """Create argparse.Namespace from config for AQB arguments."""
    import argparse
    args = argparse.Namespace()
    # Handle both dict and OmegaConf
    if hasattr(cfg_args, 'get'):
        args.controller = cfg_args.get('controller', ['omnidrone'])
        args.experiment = cfg_args.get('experiment', 'no')
        args.num_trials = cfg_args.get('num_trials', 100)
        args.seed = cfg_args.get('seed', 42)
        args.save_trials = cfg_args.get('save_trials', False)
        args.serial = cfg_args.get('serial', False)
        args.vis = cfg_args.get('vis', False)
        args.when2fail = cfg_args.get('when2fail', False)
        args.max_intensity = cfg_args.get('max_intensity', 10.0)
        args.intensity_step = cfg_args.get('intensity_step', 1.0)
        args.trajectory = cfg_args.get('trajectory', 'random')
        args.delay_margin = cfg_args.get('delay_margin', False)
    else:
        # OmegaConf object
        args.controller = getattr(cfg_args, 'controller', ['omnidrone'])
        args.experiment = getattr(cfg_args, 'experiment', 'no')
        args.num_trials = getattr(cfg_args, 'num_trials', 100)
        args.seed = getattr(cfg_args, 'seed', 42)
        args.save_trials = getattr(cfg_args, 'save_trials', False)
        args.serial = getattr(cfg_args, 'serial', False)
        args.vis = getattr(cfg_args, 'vis', False)
        args.when2fail = getattr(cfg_args, 'when2fail', False)
        args.max_intensity = getattr(cfg_args, 'max_intensity', 10.0)
        args.intensity_step = getattr(cfg_args, 'intensity_step', 1.0)
        args.trajectory = getattr(cfg_args, 'trajectory', 'random')
        args.delay_margin = getattr(cfg_args, 'delay_margin', False)
    return args


def _extract_quad_params_from_omni_drone(drone):
    """
    Extract quad_params from OmniDrone drone object in AdaptiveQuadBench format.
    
    This function converts OmniDrone drone parameters to AdaptiveQuadBench format.
    The returned params will be used by the experiment runner, which may modify them
    for different experiment types (uncertainty, wind, etc.).
    """
    # Get parameters from drone
    params = drone.params
    
    # Extract rotor configuration
    rotor_config = params['rotor_configuration']
    num_rotors = rotor_config['num_rotors']
    
    # Get rotor positions from arm_lengths and rotor_angles
    arm_lengths = rotor_config.get('arm_lengths', [0.17] * num_rotors)
    rotor_angles = rotor_config.get('rotor_angles', [2 * np.pi * i / num_rotors for i in range(num_rotors)])
    
    # Compute rotor positions (using AQB naming convention: r1, r2, r3, r4)
    rotor_pos = {}
    rotor_names = ['r1', 'r2', 'r3', 'r4'][:num_rotors]
    for i, name in enumerate(rotor_names):
        arm_len = arm_lengths[i] if isinstance(arm_lengths, list) else arm_lengths
        angle = rotor_angles[i] if isinstance(rotor_angles, list) else rotor_angles
        rotor_pos[name] = np.array([
            arm_len * np.sin(angle),
            arm_len * np.cos(angle),
            0.0
        ])
    
    # Calculate arm_length (average or max, depending on configuration)
    if isinstance(arm_lengths, list):
        arm_length = float(np.mean(arm_lengths))
    else:
        arm_length = float(arm_lengths)
    
    # Get rotor directions
    rotor_directions = np.array(rotor_config.get('directions', [1, -1, 1, -1][:num_rotors]))
    
    # Extract force and moment constants (use first rotor's values, or average if they differ)
    force_constants = rotor_config.get('force_constants', [8.54858e-6] * num_rotors)
    moment_constants = rotor_config.get('moment_constants', [1.0e-6] * num_rotors)
    max_rot_vels = rotor_config.get('max_rotation_velocities', [838.0] * num_rotors)
    
    if isinstance(force_constants, list):
        k_eta = float(np.mean(force_constants))
    else:
        k_eta = float(force_constants)
    
    if isinstance(moment_constants, list):
        k_m = float(np.mean(moment_constants))
    else:
        k_m = float(moment_constants)
    
    if isinstance(max_rot_vels, list):
        rotor_speed_max = float(np.mean(max_rot_vels))
    else:
        rotor_speed_max = float(max_rot_vels)
    
    # Extract mass and inertia
    mass = float(drone.MASS_0.item() if torch.is_tensor(drone.MASS_0) else drone.MASS_0)
    
    # Extract drag coefficient
    drag_coef = float(params.get('drag_coef', 0.0))
    
    # Extract other parameters
    quad_params = {
        # Inertial properties
        'mass': mass,
        'Ixx': float(params['inertia']['xx']),
        'Iyy': float(params['inertia'].get('yy', params['inertia']['xx'])),  # Fallback to xx if not present
        'Izz': float(params['inertia']['zz']),
        'Ixy': float(params['inertia'].get('xy', 0.0)),
        'Iyz': float(params['inertia'].get('yz', 0.0)),
        'Ixz': float(params['inertia'].get('xz', 0.0)),
        'arm_length': arm_length,
        'com': np.array([0.0, 0.0, 0.0]),  # Center of mass (default, may need adjustment)
        
        # Geometric properties
        'num_rotors': num_rotors,
        'rotor_radius': 0.10,  # Default, may need to extract from model
        'rotor_pos': rotor_pos,
        'rotor_directions': rotor_directions,
        'rotor_efficiency': np.ones(num_rotors),  # Default, can be modified by experiments
        
        # IMU location
        'rI': np.array([0, 0, 0]),  # Default IMU location
        
        # Drag coefficients (using drag_coef for all axes if available)
        'cd1x': drag_coef,
        'cd1y': drag_coef,
        'cd1z': drag_coef,
        'cdz_h': 0.00,
        
        # Frame aerodynamic properties
        'c_Dx': drag_coef,
        'c_Dy': drag_coef,
        'c_Dz': drag_coef,
        
        # Rotor properties
        'k_eta': k_eta,  # Thrust coefficient
        'k_m': k_m,  # Moment coefficient
        'k_d': 1.19e-04,  # Rotor drag coefficient (default, may need adjustment)
        'k_z': 2.32e-04,  # Induced inflow coefficient (default, may need adjustment)
        'k_flap': 0.0,  # Flapping moment coefficient (default)
        
        # Motor properties
        'tau_m': 0.005,  # Motor response time (default, may need adjustment)
        'rotor_speed_min': 0.0,
        'rotor_speed_max': rotor_speed_max,
        'motor_noise_std': 50,  # Default motor noise
    }
    
    return quad_params


if __name__ == "__main__":
    main()

