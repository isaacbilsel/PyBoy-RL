import datetime
import os
import time
from pathlib import Path

from pyboy import PyBoy
from gym.wrappers import FrameStack, NormalizeObservation

from AISettings.MarioAISettings import MarioAI
from MetricLogger import MetricLogger
from agent import AIPlayer
from wrappers import SkipFrame, ResizeObservation
from PIL import Image
from CustomPyBoyGym import CustomPyBoyGym
import torch

# --- Hardcoded Settings ---
episodes = 1000
observation_type = "tiles"
game_dimensions = (20, 18)  # Mario's screen tile dimensions
frame_stack = 4
skip_frames = 4
game_path = "games/Super_Mario_Land.gb"  # Adjust as needed

mode = "platformer"  # Could later support different Mario levels/modes

# Checkpoint options (disabled by default)
checkpoint_path = ''  # e.g. 'checkpoints/Super_Mario_Land/2025-05-07T12-00-00/mario_net.chkpt'

# --- Logger Setup ---
now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir = Path("checkpoints") / "Super_Mario_Land" / f"{now}-{mode}"

# --- Load Emulator and Environment ---
pyboy = PyBoy(game_path, window_type="headless", window_scale=3, debug=False, game_wrapper=True)
ai_settings = MarioAI()

env = CustomPyBoyGym(pyboy=pyboy, observation_type=observation_type)
env.setAISettings(ai_settings)
filtered_actions = ai_settings.GetActions()

env = SkipFrame(env, skip=skip_frames)
env = ResizeObservation(env, game_dimensions)
env = NormalizeObservation(env)
env = FrameStack(env, num_stack=frame_stack)

# --- Load AIPlayer ---
config = ai_settings.GetHyperParameters()

ai_player = AIPlayer(
    (frame_stack,) + game_dimensions,
    len(filtered_actions),
    save_dir,
    now,
    config,
    duel_dqn=True,
    use_per=True,
    use_noisy=True,
    n_step=3
)

# Optional: Resume training
# ai_player.loadModel(checkpoint_path)
# ai_player.memory.clear()
# ai_player.optimizer = torch.optim.Adam(ai_player.net.parameters(), lr=config.learning_rate)
# ai_player.scheduler = torch.optim.lr_scheduler.ExponentialLR(ai_player.optimizer, gamma=config.learning_rate_decay)

# --- Training Loop Setup ---
pyboy.set_emulation_speed(0)
save_dir.mkdir(parents=True)
gif_dir = Path("episodes") / "Super_Mario_Land" / f"{now}-{mode}"
gif_dir.mkdir(parents=True, exist_ok=True)

logger = MetricLogger(save_dir)
ai_player.saveHyperParameters()

print("Starting Mario training (headless)")
print(f"Total Episodes: {episodes}")

ai_player.net.train()
time_limit = 50  # seconds per episode

for e in range(episodes):
    observation = env.reset()
    start = time.time()
    frames = []

    while True:
        action_id = ai_player.act(observation)
        actions = filtered_actions[action_id]
        next_observation, reward, done, info = env.step(actions)

        img = pyboy.screen_image().copy()
        frames.append(img)

        ai_player.cache(observation, next_observation, action_id, reward, done)
        q, loss = ai_player.learn()

        logger.log_step(reward, loss, q, ai_player.scheduler.get_last_lr())
        observation = next_observation

        if done or time.time() - start > time_limit:
            break

    logger.log_episode()
    logger.record(
        episode=e,
        epsilon=ai_player.exploration_rate,
        stepsThisEpisode=ai_player.curr_step,
        maxLength=ai_settings.GetLength(pyboy)
    )

    gif_path = gif_dir / f"episode_{e:04d}.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=40, loop=0)

ai_player.save()
env.close()
