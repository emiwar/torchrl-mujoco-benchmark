import time
import os

import torch
import numpy as np
import pandas as pd
import mujoco

from custom_torchrl_env import CustomMujocoEnvDummy

steps = 20
batch_sizes = 2**np.arange(15)

results = []
for model in os.listdir("models"):
    if not model.endswith(".xml"):
        continue
    for batch_size in batch_sizes:
        for device in ('cpu', 'cuda'):
            print((model, batch_size, device))
            mjModel = mujoco.MjModel.from_xml_path(f"models/{model}")
            env = CustomMujocoEnvDummy(mjModel, batch_size=torch.Size((batch_size,)),
                                       device=device)
    
            s_time = time.time()
            env.rollout(steps)
            e_time = time.time()
            results.append((model, steps, batch_size, device, os.cpu_count(),
                            "env.rollout", e_time-s_time))
    
            env = CustomMujocoEnvDummy(mjModel, batch_size=torch.Size((batch_size,)),
                                       device=device)
            s_time = time.time()
            for t in range(steps):
                env.step(env.rand_action())
            e_time = time.time()
            results.append((model, steps, batch_size, device, os.cpu_count(),
                            "env.step(env.rand_action())", e_time-s_time))
    
            env = CustomMujocoEnvDummy(mjModel, batch_size=torch.Size((batch_size,)),
                                       device=device)
            s_time = time.time()
            for t in range(steps):
                env.simulation_pool.step()
            e_time = time.time()
            results.append((model, steps, batch_size, device, os.cpu_count(),
                            "env.simulation_pool.step()", e_time-s_time))

df = pd.DataFrame(results, columns=["model", "n_steps", "batch_size", "device",
                                    "n_threads", "stepping_method", "running_time"])
df["steps_per_second"] = df.eval("n_steps * batch_size / running_time")
df.to_csv("benchmark_results.csv")
