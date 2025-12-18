
import logging
import os
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution, scenarios_path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ViZDoomEnv:
    def __init__(self, scenario_path: Path, visible: bool = True):
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

        self.observation_space = np.array([480, 640])
        self.action_space = [
            [1, 0, 0, 0],  # Move Forward
            [0, 1, 0, 0],  # Move Backward
            [0, 0, 1, 0],  # Turn Left
            [0, 0, 0, 1],  # Turn Right
        ]

    def reset(self):
        self.game.new_episode()
        state = self.game.get_state()
        if state is None:
            return np.zeros(self.observation_space, dtype=np.uint8)
        return state.screen_buffer

    def step(self, action_index):
        action = self.action_space[action_index]
        self.game.make_action(action)
        done = self.game.is_episode_finished()
        if done:
            observation = np.zeros(self.observation_space, dtype=np.uint8)
            return observation, 0.0, True, {}

        state = self.game.get_state()
        observation = state.screen_buffer if state else np.zeros(self.observation_space, dtype=np.uint8)
        reward = self.game.get_last_reward()

        return observation, reward, done, {}

    def close(self):
        self.game.close()

    def render(self):
        # This method is for compatibility with SB3, but rendering is handled by ViZDoom window
        pass

def main():
    try:
        scenario_path = Path("vizdoom_ppo/scenarios/basic.cfg")
        model_path = "vizdoom_ppo_model.zip"
        log_dir = "logs/"
        vec_normalize_path = os.path.join(log_dir, "vec_normalize.pkl")

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
