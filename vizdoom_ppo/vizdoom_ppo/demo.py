
import logging
import os
import time
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution, scenarios_path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ViZDoomEnv(gym.Env):
    def __init__(self, scenario_path: Path, visible: bool = True):
        super().__init__()
        if not scenario_path.exists():
            raise FileNotFoundError(f"Scenario file not found at: {scenario_path}")

        self.game = DoomGame()
        self.game.load_config(str(scenario_path))
        
        # Point to the correct path for the scenario WAD file
        wad_path = os.path.join(scenarios_path, "basic.wad")
        self.game.set_doom_scenario_path(wad_path)

        self.game.set_window_visible(visible)
        self.game.set_mode(Mode.SPECTATOR) # Spectator mode for watching the agent
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.new_episode()
        state = self.game.get_state()
        if state is None:
            observation = np.zeros(self.observation_space.shape, dtype=np.uint8)
            info = {}
            return observation, info
        observation = np.expand_dims(state.screen_buffer, axis=-1)
        info = {}
        return observation, info

    def step(self, action_index):
        action = self.action_map[action_index]
        info = {}
        
        reward = self.game.make_action(action)
        
        terminated = self.game.is_episode_finished()
        if terminated:
            observation = np.zeros(self.observation_space.shape, dtype=np.uint8)
            return observation, 0.0, True, False, info

        state = self.game.get_state()
        if state is None:
            observation = np.zeros(self.observation_space.shape, dtype=np.uint8)
            reward = 0.0
            terminated = True
            return observation, reward, terminated, False, info

        observation = state.screen_buffer
        observation = np.expand_dims(observation, axis=-1)

        return observation, reward, terminated, False, info

    def close(self):
        self.game.close()

    def render(self):
        # This method is for compatibility with SB3, but rendering is handled by ViZDoom window
        pass

def main():
    try:
        # Corrected path for the Docker container
        scenario_path = Path(__file__).parent / "scenarios" / "basic.cfg"
        model_path = Path(__file__).parent.parent.parent / "vizdoom_ppo_model.zip"
        log_dir = Path(__file__).parent.parent.parent / "logs"
        vec_normalize_path = log_dir / "vec_normalize.pkl"

        if not Path(model_path).exists():
            logging.error(f"Model file not found at: {model_path}. Please run train.py first.")
            return
        
        if not Path(vec_normalize_path).exists():
            logging.error(f"VecNormalize stats not found at: {vec_normalize_path}. Please run train.py first.")
            return

        # Use a separate DummyVecEnv for the demo
        env_fns = [lambda: ViZDoomEnv(scenario_path)]
        env = DummyVecEnv(env_fns)
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False # Set to evaluation mode
        env.norm_reward = False # Do not normalize rewards
        
        model = PPO.load(model_path, env=env)

        logging.info("Starting demo. Press Ctrl+C to stop.")
        obs = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            # time.sleep(0.028) # ~35 FPS, to make it watchable
            if np.any(dones):
                logging.info("Episode finished. Resetting...")
                obs = env.reset()

    except KeyboardInterrupt:
        logging.info("Demo stopped by user.")
    except Exception as e:
        logging.error(f"An error occurred during demo: {e}", exc_info=True)
    finally:
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    main()
