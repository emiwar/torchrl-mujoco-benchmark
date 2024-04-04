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
        self.simulation_pool = mujoco._simulation_pool.SimulationPool(mj_model, flat_batch_size, worker_thread_count)
        
    
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

        # self.reward_spec = torchrl.data.UnboundedContinuousTensorSpec(shape=self.batch_size)
        # self.done_spec = torchrl.data.BinaryDiscreteTensorSpec(n=self.batch_size., shape=self.batch_size, dtype=torch.bool)

    # # this doesnt work
    # def _reset(self, tensordict):
    #     # flat_batch_size = self.batch_size.numel()
    #     # self.simulation_pool.setReset(np.ones(flat_batch_size, dtype=np.bool_))
        
    #     self.simulation_pool.step()
    #     # self.simulation_pool.setReset(np.zeros(flat_batch_size, dtype=np.bool_))
    #     out = TensorDict({
    #         "observation": self._getPhysicsState()
    #     }, batch_size=self.batch_size)
    #     return out

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
        observation = torch.from_numpy(np.array(self.simulation_pool.getState()))
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
        mj_model = mujoco.MjModel.from_xml_path("/home/charles/github/torchrl-mujoco-benchmark/models/rodent.xml")
        super().__init__(mj_model=mj_model, seed=seed,
                         batch_size=batch_size, device=device,
                         worker_thread_count=worker_thread_count)
        self._forward_reward_weight = 10
        self._ctrl_cost_weight = 0.1
        self._healthy_reward = 1.0
        self._min_z = 0.02
        self.step_count = torch.zeros(batch_size, dtype=torch.int64, device=device)
        state_size = mujoco.mj_stateSize(self._mj_model, mujoco.mjtState.mjSTATE_FULLPHYSICS)

        self.observation_spec = torchrl.data.CompositeSpec(
            observation = torchrl.data.UnboundedContinuousTensorSpec(
                shape=self.batch_size + (state_size,),
                device=device,
                dtype=torch.float32
            ),
            step_count = torchrl.data.UnboundedContinuousTensorSpec(
                shape=self.batch_size + (1,),
                device=device,
                dtype=torch.int64
            ),
            episode_reward = torchrl.data.UnboundedContinuousTensorSpec(
                shape=self.batch_size + (1,),
                device=device,
                dtype=torch.float32
            ),
            shape=self.batch_size
        )
        

    def _reset(self, tensordict=None):
        
        if tensordict is not None:
            self.simulation_pool.reset(tensordict['done'].flatten().cpu())
            # Resets step counts and done states, updates observation
            tensordict['step_count'] *= ~tensordict['done']
            tensordict['episode_reward'] *= ~tensordict['done']
            self.step_count = tensordict['step_count'] 
            tensordict['done'] = torch.zeros(size=tensordict['done'].size(), dtype=torch.bool)
            tensordict['observation'] = self._getPhysicsState()
            
            return tensordict
        else:
            self.simulation_pool.reset(np.ones(self.batch_size.numel(), dtype=np.bool_))
            self.step_count = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
            self.episode_reward = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
            # out = super()._reset(tensordict)
        
            out = TensorDict({
                "observation": self._getPhysicsState(),
                "step_count": self.step_count,
                "episode_reward": self.episode_reward
            }, batch_size=self.batch_size)
            return out
        
    def _step(self, tensordict):
        action = tensordict["action"]
        if action.isnan().any():
            raise ValueError("Passed action contains NaNs.")
        state_size = mujoco.mj_stateSize(self._mj_model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        com_before = torch.from_numpy(np.array(self.simulation_pool.getSubtree_com())[:, 1, :]).to(self.device)
        self.simulation_pool.setControl(np.clip(action.cpu().numpy(), -1, 1))
        self.simulation_pool.step()
        com_after = torch.from_numpy(np.array(self.simulation_pool.getSubtree_com())[:, 1, :]).to(self.device)
        
        # Calculate reward
        velocity = (com_after - com_before) /  self._mj_model.opt.timestep
        forward_reward = self._forward_reward_weight * velocity[:, 0]
        ctrl_cost = self._ctrl_cost_weight * torch.square(action).sum(axis=-1)
        reward = (forward_reward + self._healthy_reward - ctrl_cost).to(dtype=torch.float32)
        done = com_after[:, 2] < self._min_z
        
        # I'm confused if this should be done in _reset or not
        # the actual resetting logic should be done outside of the env,
        # such as in the torchrl DataCollector
        # self.simulation_pool.setReset(done.flatten().cpu())
        self.step_count += 1
        self.episode_reward += reward
        # self.step_count *= ~done
        out = TensorDict({
            "observation": self._getPhysicsState(),
            "reward": reward,
            "done": done,
            "step_count": self.step_count,
            "episode_reward": self.episode_reward,
        }, batch_size=self.batch_size)
        return out
