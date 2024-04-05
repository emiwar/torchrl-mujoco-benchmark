import os
import numpy as np

import torch
import torchrl.data
from tensordict import TensorDict

import mujoco
#This package is in a fork of MuJoCo: https://github.com/emiwar/mujoco/tree/feature/simulation_pool
#Build and install according to 
#https://mujoco.readthedocs.io/en/stable/programming/index.html#building-from-source
import mujoco._simulation_pool

class CustomMujocoEnvBase(torchrl.envs.EnvBase):
    def __init__(self, mj_model: mujoco.MjModel, seed=None, batch_size=[], device="cpu",
                 worker_thread_count:int = os.cpu_count()):
        super().__init__(device=device, batch_size=batch_size)
        self._mj_model = mj_model
        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)
        flat_batch_size = self.batch_size.numel()
        self.simulation_pool = mujoco._simulation_pool.SimulationPool(mj_model, flat_batch_size,
                                                                      worker_thread_count,
                                                                      mujoco.mjtState.mjSTATE_FULLPHYSICS)
        
    
    def _make_spec(self):
        state_size = mujoco.mj_stateSize(self._mj_model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        action_size = mujoco.mj_stateSize(self._mj_model, mujoco.mjtState.mjSTATE_CTRL)
        self.observation_spec = torchrl.data.CompositeSpec(
            observation = torchrl.data.UnboundedContinuousTensorSpec(
                shape=self.batch_size + (state_size,),
                dtype=torch.float32
            ),
            shape=self.batch_size
        )
        #Not sure about this one...
        self.state_spec = self.observation_spec.clone()

        self.action_spec = torchrl.data.BoundedTensorSpec(
            low=-torch.ones(self.batch_size + (action_size,), dtype=torch.float32, device=self.device),
            high=torch.ones(self.batch_size + (action_size,), dtype=torch.float32, device=self.device),
            device=self.device,
            dtype=torch.float32)

        #self.reward_spec = torchrl.data.UnboundedContinuousTensorSpec(shape=self.batch_size)
        #self.done_spec = torchrl.data.BinaryDiscreteTensorSpec(n=self.batch_size, shape=self.batch_size, dtype=torch.bool)

    def _reset(self, tensordict=None):
        if tensordict is not None:
            self.simulation_pool.reset(tensordict['_reset'].flatten().cpu())
        else:
            self.simulation_pool.reset(np.ones(self.batch_size.numel(), dtype=np.bool_))        
        tensordict = TensorDict({"observation": self._getPhysicsState()},
                                batch_size=self.batch_size)
        return tensordict

    def _step(self, tensordict):
        state_size = mujoco.mj_stateSize(self._mj_model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self.simulation_pool.step()
        out = TensorDict({
            "observation": self._getPhysicsState(),
            "reward": self._getReward(),
            "done": self._getDone()
        }, batch_size=self.batch_size)
        return out

    def _set_seed(self, seed):
        rng = torch.manual_seed(seed)
        self.rng = rng
        
    def _getPhysicsState(self):
        state_size = mujoco.mj_stateSize(self._mj_model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        observation = torch.from_numpy(np.array(self.simulation_pool.getState(), copy=False))
        if observation.isnan().any():
            raise RuntimeError("MuJoCo physics state contain NaNs.")
        return observation.to(device=self.device, dtype=torch.float32).reshape(self.batch_size + (state_size,))
    
    def _getReward(self):
        raise NotImplementedError("Reward function not implemented.")

    def _getDone(self):
        raise NotImplementedError("Termination criterion not implemented.")

class CustomMujocoEnvDummy(CustomMujocoEnvBase):
    '''Dummy implementation of the custom MuJoCo environment that never terminates nor gives any reward.'''
    def _getReward(self):
        return torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
    def _getDone(self):
        return torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

class RodentRunEnv(CustomMujocoEnvBase):

    def __init__(self, seed=None, batch_size=[1], device="cpu", worker_thread_count = os.cpu_count()):
        mj_model = mujoco.MjModel.from_xml_path("models/rodent.xml")
        super().__init__(mj_model=mj_model, seed=seed,
                         batch_size=batch_size, device=device,
                         worker_thread_count=worker_thread_count)
        self._forward_reward_weight = 10
        self._ctrl_cost_weight = 0.1
        self._healthy_reward = 1.0
        self._min_z = 0.02
        
    def _step(self, tensordict):
        action = tensordict["action"]
        if action.isnan().any():
            raise ValueError("Passed action contains NaNs.")
        state_size = mujoco.mj_stateSize(self._mj_model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        control_size = mujoco.mj_stateSize(self._mj_model, mujoco.mjtState.mjSTATE_CTRL)
        com_before = torch.from_numpy(np.array(self.simulation_pool.getSubtree_com())[:, 1, :]).to(self.device).reshape(self.batch_size + (3,))
        self.simulation_pool.setControl(np.clip(action.cpu().numpy().reshape(self.batch_size.numel(), control_size), -1, 1))
        self.simulation_pool.step()
        com_after = torch.from_numpy(np.array(self.simulation_pool.getSubtree_com())[:, 1, :]).to(self.device).reshape(self.batch_size + (3,))
        
        # Calculate reward
        velocity = (com_after - com_before) /  self._mj_model.opt.timestep
        forward_reward = self._forward_reward_weight * velocity[..., 0]
        ctrl_cost = self._ctrl_cost_weight * torch.square(action).sum(axis=-1)
        reward = (forward_reward + self._healthy_reward - ctrl_cost).to(dtype=torch.float32)
        done = com_after[..., 2] < self._min_z
        
        out = TensorDict({
            "observation": self._getPhysicsState(),
            "reward": reward.reshape(self.batch_size + (1,)),
            "done": done
        }, batch_size=self.batch_size)
        return out
