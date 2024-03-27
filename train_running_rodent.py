import os
import collections

import numpy as np
import torch
import torchrl
import tensordict
import tqdm

import custom_torchrl_env

num_cells = 1024
lr = 1e-4
max_grad_norm = 1.0
device = 'cuda'
env_batch_size = 2048
env_worker_threads = os.cpu_count()-4
frames_per_batch = 8*1024
total_frames = 2048*1024

clip_epsilon = (
    0.2  # clip value for PPO
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

print(f"Starting environment with {env_worker_threads} threads.")
env = custom_torchrl_env.RodentRunEnv(batch_size=(env_batch_size,),
                                      device=device,
                                      worker_thread_count=env_worker_threads)

actor_net = torch.nn.Sequential(
    torch.nn.LazyLinear(num_cells, device=device),
    torch.nn.Tanh(),
    torch.nn.LazyLinear(num_cells, device=device),
    torch.nn.Tanh(),
    torch.nn.LazyLinear(num_cells, device=device),
    torch.nn.Tanh(),
    torch.nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
    tensordict.nn.distributions.NormalParamExtractor(),
)
policy_module = tensordict.nn.TensorDictModule(
    actor_net, in_keys=["fullphysics"], out_keys=["loc", "scale"]
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
    torch.nn.LazyLinear(num_cells, device=device),
    torch.nn.Tanh(),
    torch.nn.LazyLinear(num_cells, device=device),
    torch.nn.Tanh(),
    torch.nn.LazyLinear(num_cells, device=device),
    torch.nn.Tanh(),
    torch.nn.LazyLinear(1, device=device),
)
value_module = torchrl.modules.ValueOperator(
    module=value_net,
    in_keys=["fullphysics"]
)
print("Testing policy module output shape:", policy_module(env.reset()).shape)
print("Testing value module output shape:", value_module(env.reset()).shape)
collector = torchrl.collectors.SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)
advantage_module = torchrl.objectives.value.GAE(
    gamma=gamma,
    lmbda=lmbda,
    value_network=value_module,
    average_gae=True,
    device=device
)
loss_module = torchrl.objectives.ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    # these keys match by default but we set this for completeness
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)
optim = torch.optim.Adam(loss_module.parameters(), lr)

for i, tensordict_data in tqdm.tqdm(enumerate(collector)):
    for j in range(tensordict_data.shape[1]):
        advantage_module(tensordict_data[:,j])
        loss_vals = loss_module(tensordict_data[:,j])
        loss_value = (
            loss_vals["loss_objective"]
            + loss_vals["loss_critic"]
            + loss_vals["loss_entropy"]
        )
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
        optim.step()
        optim.zero_grad()
