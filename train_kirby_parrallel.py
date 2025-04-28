import datetime
import os
import time
from pathlib import Path

from pyboy import PyBoy, WindowEvent
from gym.wrappers import FrameStack, NormalizeObservation

from AISettings.KirbyAISettings import KirbyAI
from MetricLogger import MetricLogger
from agent import AIPlayer
from wrappers import SkipFrame, ResizeObservation
from CustomPyBoyGym import CustomPyBoyGym
from gym.vector import SyncVectorEnv

import numpy as np

"""
Hardcoded Settings
"""
episodes = 400
observation_type = "tiles"
game_dimensions = (20, 16)
frame_stack = 4
skip_frames = 4
game_path = "games/Kirby_Dream_Land.gb"

"""
Logger Setup
"""
now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir = Path("checkpoints") / "Kirby_Dream_Land" / now
save_dir_boss = Path("checkpoints") / "Kirby_Dream_Land" / (now + "-boss")

"""
Load Emulator and Environment
"""
pyboy_list = []  # Create this global list to store all pyboys

def make_env():
    def _init():
        pyboy = PyBoy("games/Kirby_Dream_Land.gb", window_type="headless", window_scale=3, debug=False, game_wrapper=True)
        ai_settings = KirbyAI()

        env = CustomPyBoyGym(pyboy, observation_type="tiles")
        env.setAISettings(ai_settings)

        env = SkipFrame(env, skip=4)
        env = ResizeObservation(env, (20, 16))
        env = NormalizeObservation(env)
        env = FrameStack(env, num_stack=4)

        pyboy_list.append(pyboy)  # Save pyboy reference

        return env
    return _init

num_envs = 3
envs = SyncVectorEnv([make_env() for _ in range(num_envs)])

# pyboy = PyBoy(game_path, window_type="headless", window_scale=3, debug=False, game_wrapper=True)
ai_settings = KirbyAI()

# env = CustomPyBoyGym(pyboy, observation_type=observation_type)
# env.setAISettings(ai_settings)
filtered_actions = ai_settings.GetActions()

# env = SkipFrame(env, skip=skip_frames)
# env = ResizeObservation(env, game_dimensions)
# env = NormalizeObservation(env)
# env = FrameStack(env, num_stack=frame_stack)

"""
Load AI Players
"""
ai_player = AIPlayer(
    (frame_stack,) + game_dimensions,
    len(filtered_actions),
    save_dir,
    now,
    ai_settings.GetHyperParameters()
)

boss_ai_player = AIPlayer(
    (frame_stack,) + game_dimensions,
    len(filtered_actions),
    save_dir_boss,
    now,
    ai_settings.GetBossHyperParameters()
)

"""
Training Loop (Parallelized Version)
"""
save_dir.mkdir(parents=True, exist_ok=True)
save_dir_boss.mkdir(parents=True, exist_ok=True)

logger = MetricLogger(save_dir_boss)

ai_player.saveHyperParameters()
boss_ai_player.saveHyperParameters()

print("Starting Kirby training (headless)")
print(f"Total Episodes: {episodes}")

ai_player.net.train()
boss_ai_player.net.train()

# Track per-environment episode counters
episode_counters = [0 for _ in range(num_envs)]
max_episodes_per_env = episodes // num_envs + 1

# Reset all environments
observations = envs.reset()
start_times = [time.time() for _ in range(num_envs)]

done_training = False

while not done_training:
    actions = []
    players = []

    # Select action for each environment
    for i in range(num_envs):
        pyboy = pyboy_list[i]
        if ai_settings.IsBossActive(pyboy):
            player = boss_ai_player
        else:
            player = ai_player
        players.append(player)

        action_id = player.act(observations[i])
        actions.append(action_id)

    actions = np.array(actions)

    # Step all environments together
    next_observations, rewards, dones, infos = envs.step(actions)

    # For each environment individually
    for i in range(num_envs):
        players[i].cache(observations[i], next_observations[i], actions[i], rewards[i], dones[i])
        q, loss = players[i].learn()

        logger.log_step(rewards[i], loss, q, players[i].scheduler.get_last_lr())

        # If episode done (Kirby died or cleared level)
        if dones[i] or (time.time() - start_times[i] > 500):
            logger.log_episode()
            logger.record(
                episode=episode_counters[i],
                epsilon=players[i].exploration_rate,
                stepsThisEpisode=players[i].curr_step,
                maxLength=ai_settings.GetLength(pyboy_list[i])
            )

            episode_counters[i] += 1
            start_times[i] = time.time()

            # Reset only that specific environment
            obs_reset = envs.reset_done(indices=[i])
            next_observations[i] = obs_reset[0]  # Insert the fresh reset obs

    observations = next_observations

    # Check if all environments have finished enough episodes
    if all(ep >= max_episodes_per_env for ep in episode_counters):
        done_training = True

# Save models
ai_player.save()
boss_ai_player.save()

envs.close()