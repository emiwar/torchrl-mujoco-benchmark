import numpy as np
import torch
import torchrl

import custom_torchrl_env

test_args = [
    {'batch_size': (16,), 'device': 'cuda', 'worker_thread_count': 4},
    {'batch_size': (1024,), 'device': 'cpu'},
    {'batch_size': (1024,), 'device': 'cuda'},
    {'batch_size': (7, 13), 'device': 'cuda'},
]

for args in test_args:
    args["device"] = "cpu"
    env = custom_torchrl_env.RodentRunEnv(**args)
    torchrl.envs.utils.check_env_specs(env)
    transEnv = torchrl.envs.TransformedEnv(
        env,
        torchrl.envs.Compose(
            # normalize observations
            torchrl.envs.ObservationNorm(in_keys=["observation"]),
            torchrl.envs.StepCounter(max_steps=5),
            torchrl.envs.RewardSum()
        ),
    )
    transEnv.transform[0].init_stats(num_iter=48, cat_dim=0, reduce_dim=tuple(range(len(env.batch_size)+1)))
    torchrl.envs.utils.check_env_specs(transEnv)

    T = 20
    rollout = transEnv.rollout(T, break_when_any_done=False)
    for t in range(0, T, 5):
        if torch.any(rollout[..., t]["observation"] != rollout[..., 0]["observation"]):
            raise RuntimeError(f"Observation at t={t} did not produce same observation as t=0.")
    print("Reset after max_steps works.")

    td = transEnv.reset()
    for t in range(3):
        td["action"] = transEnv.rand_action()["action"]
        td_, td = transEnv.step_and_maybe_reset(td)
        
    #Flag every second environment for resetting
    td["done"][..., ::2, :] = True
    td = transEnv.maybe_reset(td)
    if (td["step_count"][..., ::2, :]==0).all() and (td["step_count"][..., 1::2, :]>0).all():
        print("Partial reset of step_count works.")
    else:
        raise RuntimeError("Partial reset of step_count does not work.")
    if (td["observation"][..., ::2, :] == td["observation"][..., 0:1, :]).all() and \
       not (td["observation"][..., 1::2, :] == td["observation"][..., 0:1, :]).all() :
        print("Partial reset of mujoco envs works.")
    else:
        raise RuntimeError("Partial reset of mujoco envs does not work.")