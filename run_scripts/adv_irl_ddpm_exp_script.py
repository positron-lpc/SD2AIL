import yaml
import argparse
import numpy as np
import os, sys, inspect
import pickle

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

import gym
from rlkit.envs import get_env, get_envs
import random
# noinspection PyPackageRequirements
import rlkit.torch.utils.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.prioritized_replay_buffer import PrioritizedReplayBuffer
from rlkit.torch.common.networks import FlattenMlp
from rlkit.torch.common.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.algorithms.sac.sac_alpha import (
    SoftActorCritic,
)  # SAC Auto alpha version
from ddpm_disc.ddpm_disc_model import DDPM_Disc
from rlkit.torch.algorithms.adv_irl.adv_irl import AdvIRL
from rlkit.envs.wrappers import ProxyEnv, ScaledEnv, MinmaxEnv, EPS


def experiment(variant):
    with open("demos_listing.yaml", "r") as f:
        listings = yaml.safe_load(f.read())

    demos_path = listings[variant["expert_name"]]["file_paths"][variant["expert_idx"]]
    """
    Buffer input format
    """

    """
    PKL input format
    """
    print("demos_path", demos_path)

    with open(demos_path, "rb") as f:
        traj_list = pickle.load(f)
    traj_list = random.sample(traj_list, variant["traj_num"])

    obs = np.vstack([traj_list[i]["observations"] for i in range(len(traj_list))])
    obs_mean, obs_std = np.mean(obs, axis=0), np.std(obs, axis=0)
    acts_mean, acts_std = None, None
    obs_min, obs_max = np.min(obs, axis=0), np.max(obs, axis=0)

    env_specs = variant["env_specs"]
    env = get_env(env_specs)
    env.seed(env_specs["eval_env_seed"])

    print("\n\nEnv: {}".format(env_specs["env_name"]))  #
    print("kwargs: {}".format(env_specs["env_kwargs"]))
    print("Obs Space: {}".format(env.observation_space))
    print("Act Space: {}\n\n".format(env.action_space))
    # multi trajectories replay buffer
    if variant["replay_params"]["replay_model"] == "priority":
        capacity = variant["adv_irl_params"]["max_path_length"]
        expert_replay_buffer = [PrioritizedReplayBuffer(
            capacity,
            alpha=variant["replay_params"]["priority_alpha"],
            beta=variant["replay_params"]["priority_beta"],
            random_seed=np.random.randint(10000), ) for _ in range(variant["traj_num"])]
        fake_expert_replay_buffer = PrioritizedReplayBuffer(
            100000,
            alpha=variant["replay_params"]["priority_alpha"],
            beta=variant["replay_params"]["priority_beta"],
            random_seed=np.random.randint(10000))
        print("Using multiple replay buffers")
    else:
        expert_replay_buffer = EnvReplayBuffer(
            variant["adv_irl_params"]["replay_buffer_size"],
            env,
            random_seed=np.random.randint(10000),
        )
        print("Using normal replay buffers")
    # env_wrapper
    tmp_env_wrapper = env_wrapper = ProxyEnv  # Identical wrapper
    kwargs = {}
    wrapper_kwargs = {}

    if variant["scale_env_with_demo_stats"]:
        print("\nWARNING: Using scale env wrapper")
        tmp_env_wrapper = env_wrapper = ScaledEnv
        wrapper_kwargs = dict(
            obs_mean=obs_mean,
            obs_std=obs_std,
            acts_mean=acts_mean,
            acts_std=acts_std,
        )
        for i in range(len(traj_list)):
            traj_list[i]["observations"] = (traj_list[i]["observations"] - obs_mean) / (obs_std + EPS)
            traj_list[i]["next_observations"] = (traj_list[i]["next_observations"] - obs_mean) / (obs_std + EPS)

    elif variant["minmax_env_with_demo_stats"]:
        print("\nWARNING: Using min max env wrapper")
        tmp_env_wrapper = env_wrapper = MinmaxEnv
        wrapper_kwargs = dict(obs_min=obs_min, obs_max=obs_max)
        for i in range(len(traj_list)):
            traj_list[i]["observations"] = (traj_list[i]["observations"] - obs_min) / (obs_max - obs_min + EPS)
            traj_list[i]["next_observations"] = (traj_list[i]["next_observations"] - obs_min) / (
                    obs_max - obs_min + EPS)

    obs_space = env.observation_space
    act_space = env.action_space
    max_action = float(env.action_space.high[0])
    assert not isinstance(obs_space, gym.spaces.Dict)
    assert len(obs_space.shape) == 1
    assert len(act_space.shape) == 1

    env = env_wrapper(env, **wrapper_kwargs)
    training_env = get_envs(
        env_specs, env_wrapper, wrapper_kwargs=wrapper_kwargs, **kwargs
    )
    training_env.seed(env_specs["training_env_seed"])

    obs_dim = obs_space.shape[0]
    action_dim = act_space.shape[0]

    # build the policy models
    net_size = variant["policy_net_size"]
    num_hidden = variant["policy_num_hidden_layers"]
    qf1 = FlattenMlp(
        hidden_sizes=num_hidden * [net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=num_hidden * [net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    policy = ReparamTanhMultivariateGaussianPolicy(
        hidden_sizes=num_hidden * [net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    if variant["adv_irl_params"]["wrap_absorbing"]:
        obs_dim += 1
    input_dim = obs_dim + action_dim
    if variant["adv_irl_params"]["state_only"]:
        input_dim = obs_dim + obs_dim

    # build the discriminator model

    disc_model = DDPM_Disc(
        input_dim, action_dim, max_action,
        n_timesteps=variant["disc_ddpm_n_timesteps"],
        beta_schedule=variant["disc_ddpm_beta_schedule"],
        disc_hid_dim=exp_specs["disc_hid_dim"],
        device=ptu.device,
        lr=variant["adv_irl_params"]["disc_lr"],
        disc_momentum=variant["adv_irl_params"]["disc_momentum"],
        clamp_magnitude=variant["disc_clamp_magnitude"],
    )

    for i in range(len(traj_list)):
        if variant["replay_params"]["replay_model"] == "priority":
            expert_replay_buffer[i].add_path(
                traj_list[i], absorbing=variant["adv_irl_params"]["wrap_absorbing"], env=env,
            )
        else:
            expert_replay_buffer.add_path(
                traj_list[i], absorbing=variant["adv_irl_params"]["wrap_absorbing"], env=env,
            )

    # set up the algorithm
    trainer = SoftActorCritic(
        policy=policy, qf1=qf1, qf2=qf2, env=env, **variant["sac_params"]
    )
    algorithm = AdvIRL(
        env=env,
        training_env=training_env,
        exploration_policy=policy,
        discriminator=disc_model,
        policy_trainer=trainer,
        expert_replay_buffer=expert_replay_buffer,
        fake_expert_replay_buffer=fake_expert_replay_buffer,
        disc_ddpm=variant["env_specs"]["disc_ddpm"],
        **variant["adv_irl_params"]
    )

    if ptu.gpu_enabled():
        algorithm.to(ptu.device)
    algorithm.train()

    return 1


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", default=2, type=int)
    args = parser.parse_args()
    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.safe_load(spec_string)

    # make all seeds the same.
    exp_specs["env_specs"]["eval_env_seed"] = exp_specs["env_specs"][
        "training_env_seed"
    ] = exp_specs["seed"]

    exp_suffix = ""
    exp_suffix = "-trj-{}".format(
        format(exp_specs["traj_num"]),
    )

    if not exp_specs["adv_irl_params"]["no_terminal"]:
        exp_suffix = "--terminal" + exp_suffix

    if exp_specs["adv_irl_params"]["wrap_absorbing"]:
        exp_suffix = "--absorbing" + exp_suffix

    if exp_specs["scale_env_with_demo_stats"]:
        exp_suffix = "--scale" + exp_suffix

    if exp_specs["using_gpus"] > 0:
        print("\n\nUSING GPU\n\n")
        ptu.set_gpu_mode(True, args.gpu)

    exp_id = exp_specs["exp_id"]
    exp_prefix = exp_specs["exp_name"]
    seed = exp_specs["seed"]
    set_seed(seed)

    exp_prefix = exp_prefix + exp_suffix

    # log/exp_prefix/exp_name
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs, seed=seed)

    experiment(exp_specs)
