import os
import collections

import numpy as np
import torch
import torchrl
import tensordict
import tqdm
import wandb

from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)

import uuid

import custom_torchrl_env

config = {
    'num_cells': 1024,
    'lr': 1e-4,
    'max_grad_norm': 1.0,
    'device': 'cuda',
    'batch_size': 1024,
    'env_worker_threads': os.cpu_count()-4,
    'frames_per_batch': 8*1024,
    'total_frames': 2048*1024,
    'clip_epsilon': 0.2,
    'gamma': 0.99,
    'lambda': 0.95,
    'entropy_eps': 1e-4,
    'max_steps': 1000,
}

slurm_vars = ("SLURM_CLUSTER_NAME", "SLURM_JOB_PARTITION"
              "SLURMD_NODENAME", "SLURM_JOB_ACCOUNT",
              "SLURM_CPUS_ON_NODE", "SLURM_GPUS",
              "SLURM_JOB_START_TIME", "SLURM_MEM_PER_NODE",
              "SLURM_NNODES", "SLURM_RESTART_COUNT")
for var in slurm_vars:
    if var in os.environ:
        config[f"${var}"] = os.environ[var]

id = uuid.uuid4()

wandb.init(
    project="Test-Custom-TorchRL-Env",
    notes="Testing rodent running with PPO. This test is local.",
    name=f"ppotest_{id}",
    config=config
)

env_worker_threads = config["env_worker_threads"]
device = config["device"]
print(f"Starting environment with {env_worker_threads} threads.")
env = custom_torchrl_env.RodentRunEnv(batch_size=(config["batch_size"],),
                                      device=device,
                                      worker_thread_count=env_worker_threads)

env = TransformedEnv(
    env,
    Compose(
        # normalize observations
        ObservationNorm(in_keys=["observation"]),
        StepCounter(max_steps=config['max_steps']),
    ),
)

# This needs to be set to 1024, but is it bc of nn shape or batch_size?
env.transform[0].init_stats(num_iter=config["batch_size"], reduce_dim=0, cat_dim=0)

actor_net = torch.nn.Sequential(
    torch.nn.LazyLinear(config["num_cells"], device=device),
    torch.nn.Tanh(),
    torch.nn.LazyLinear(config["num_cells"], device=device),
    torch.nn.Tanh(),
    torch.nn.LazyLinear(config["num_cells"], device=device),
    torch.nn.Tanh(),
    torch.nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
    tensordict.nn.distributions.NormalParamExtractor(),
)
policy_module = tensordict.nn.TensorDictModule(
    actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
)
policy_module = torchrl.modules.ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["loc", "scale"],
    distribution_class=torchrl.modules.TanhNormal,
    distribution_kwargs={
        "min": env.action_spec.space.low,
        "max": env.action_spec.space.high,
    },
    return_log_prob=True,
    # we'll need the log-prob for the numerator of the importance weights
)
value_net = torch.nn.Sequential(
    torch.nn.LazyLinear(config["num_cells"], device=device),
    torch.nn.Tanh(),
    torch.nn.LazyLinear(config["num_cells"], device=device),
    torch.nn.Tanh(),
    torch.nn.LazyLinear(config["num_cells"], device=device),
    torch.nn.Tanh(),
    torch.nn.LazyLinear(1, device=device),
)
value_module = torchrl.modules.ValueOperator(
    module=value_net,
    in_keys=["observation"]
)
print("Testing policy module output shape:", policy_module(env.reset()).shape)
print("Testing value module output shape:", value_module(env.reset()).shape)
collector = torchrl.collectors.SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=config["frames_per_batch"],
    total_frames=config["total_frames"],
    split_trajs=False,
    device=device,
)
advantage_module = torchrl.objectives.value.GAE(
    gamma=config["gamma"],
    lmbda=config["lambda"],
    value_network=value_module,
    average_gae=True,
    device=device
)
loss_module = torchrl.objectives.ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=config["clip_epsilon"],
    entropy_bonus=bool(config["entropy_eps"]),
    entropy_coef=config["entropy_eps"],
    # these keys match by default but we set this for completeness
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)
optim = torch.optim.Adam(loss_module.parameters(), config["lr"])

for i, tensordict_data in tqdm.tqdm(enumerate(collector)):
    for j in range(tensordict_data.shape[1]):
        #Training
        advantage_module(tensordict_data[:,j])
        loss_vals = loss_module(tensordict_data[:,j])
        loss_value = (
            loss_vals["loss_objective"]
            + loss_vals["loss_critic"]
            + loss_vals["loss_entropy"]
        )
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), config["max_grad_norm"])
        optim.step()
        optim.zero_grad()

        #Logging
        mean_reward = tensordict_data["next", "reward"].mean().item()
        avg_step_count = tensordict_data["step_count"].to(dtype=torch.float64).mean().item()
        max_step_count = tensordict_data["step_count"].max().item()
        wandb.log({
            "mean_reward": mean_reward,
            "avg_step_count": avg_step_count,
            "max_step_count": max_step_count,
            "loss_objective": loss_vals["loss_objective"],
            "loss_critic": loss_vals["loss_critic"],
            "loss_entropy": loss_vals["loss_entropy"],
            "loss_total": loss_value
        })

wandb.finish()
        
