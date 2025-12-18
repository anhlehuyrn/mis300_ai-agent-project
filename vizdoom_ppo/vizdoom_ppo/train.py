
import logging
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution, scenarios_path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ViZDoomEnv(gym.Env):
    def __init__(self, scenario_path: Path, visible: bool = False):
        if not scenario_path.exists():
            raise FileNotFoundError(f"Scenario file not found at: {scenario_path}")

        self.game = DoomGame()
        self.game.load_config(str(scenario_path))
        
        # Point to the correct path for the scenario WAD file
        wad_path = os.path.join(scenarios_path, "basic.wad")
        self.game.set_doom_scenario_path(wad_path)
        
        self.game.set_window_visible(visible)
        self.game.set_mode(Mode.PLAYER)
        self.game.set_screen_format(ScreenFormat.GRAY8)
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        self.game.init()

        self.action_map = [
            [1, 0, 0, 0],  # Move Forward
            [0, 1, 0, 0],  # Move Backward
            [0, 0, 1, 0],  # Turn Left
            [0, 0, 0, 1],  # Turn Right
        ]
        self.observation_space = spaces.Box(low=0, high=255, shape=(480, 640, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(self.action_map))

        self.state = None
        self.last_health = 100
        self.last_ammo = 26

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.new_episode()
        self.state = self.game.get_state().screen_buffer
        observation = np.expand_dims(self.state, axis=-1).astype(np.float32) / 255.0
        info = {}
        return observation, info

    def step(self, action_index):
        action = self.action_map[action_index]
        info = {}
        
        reward = self.game.make_action(action)
        
        terminated = self.game.is_episode_finished()
        if terminated:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            return observation, 0.0, True, False, info

        state = self.game.get_state()
        if state is None:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            reward = 0.0
            terminated = True
            return observation, reward, terminated, False, info

        observation = state.screen_buffer
        observation = np.expand_dims(observation, axis=-1).astype(np.float32) / 255.0

        health = state.game_variables[0]
        ammo = state.game_variables[1]

        reward += (health - self.last_health) * 0.1
        reward += (ammo - self.last_ammo) * 0.1
        
        self.last_health = health
        self.last_ammo = ammo
        
        reward -= 0.01

        return observation, reward, terminated, False, info

    def close(self):
        self.game.close()

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log episodic reward
        if self.n_calls % 1000 == 0:
            ep_info_buffer = self.model.ep_info_buffer
            if len(ep_info_buffer) > 0:
                ep_rew_mean = np.mean([ep_info['r'] for ep_info in ep_info_buffer])
                self.logger.record('rollout/ep_rew_mean', ep_rew_mean)
        return True

def main():
    try:
        # Corrected path for the Docker container
        scenario_path = Path(__file__).parent / "scenarios" / "basic.cfg"
        log_dir = "logs/"
        os.makedirs(log_dir, exist_ok=True)
        vec_normalize_path = os.path.join(log_dir, "vec_normalize.pkl")

        env = ViZDoomEnv(scenario_path)
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.)

        model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir)
        
        logging.info("Starting model training.")
        model.learn(total_timesteps=10000, callback=TensorboardCallback())
        
        model.save("vizdoom_ppo_model")
        env.save(vec_normalize_path)
        logging.info(f"Model and VecNormalize stats saved. Stats at: {vec_normalize_path}")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
    finally:
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    main()
