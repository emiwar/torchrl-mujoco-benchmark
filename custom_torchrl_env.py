import os
import numpy as np

import torch
import torchrl.data
import tensordict

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
        self.simulation_pool = mujoco._simulation_pool.SimulationPool(mj_model, flat_batch_size, worker_thread_count)

    def _make_spec(self):
        state_size = mujoco.mj_stateSize(self._mj_model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        action_size = mujoco.mj_stateSize(self._mj_model, mujoco.mjtState.mjSTATE_CTRL)
        self.observation_spec = torchrl.data.CompositeSpec(
            fullphysics = torchrl.data.UnboundedContinuousTensorSpec(
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

        self.reward_spec = torchrl.data.UnboundedContinuousTensorSpec(shape=self.batch_size+(1,))

    def _reset(self, _):
        flat_batch_size = self.batch_size.numel()
        self.simulation_pool.setReset(np.ones(flat_batch_size, dtype=np.bool_))
        self.simulation_pool.step()
        self.simulation_pool.setReset(np.zeros(flat_batch_size, dtype=np.bool_))
        out = tensordict.TensorDict({
            "fullphysics": self._getPhysicsState()
        }, batch_size=self.batch_size)
        return out

    def _step(self, action):
        state_size = mujoco.mj_stateSize(self._mj_model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self.simulation_pool.step()
        out = tensordict.TensorDict({
            "fullphysics": self._getPhysicsState(),
            "reward": self._getReward(),
            "done": self._getDone()
        }, batch_size=self.batch_size)
        return out

    def _set_seed(self, seed):
        rng = torch.manual_seed(seed)
        self.rng = rng
        
    def _getPhysicsState(self):
        state_size = mujoco.mj_stateSize(self._mj_model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        fullphysics = torch.from_numpy(np.array(self.simulation_pool.getState()))
        return fullphysics.to(device=self.device, dtype=torch.float32).reshape(self.batch_size + (state_size,))
    
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