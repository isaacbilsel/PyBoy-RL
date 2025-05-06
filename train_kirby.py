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
from PIL import Image
from CustomPyBoyGym import KirbyGymEnv  # your new custom environment
import sys



"""
Hardcoded Settings
"""
episodes = 1500
observation_type = "tiles"
game_dimensions = (20, 16)
frame_stack = 4
skip_frames = 4
game_path = "games/Kirby_Dream_Land.gb"

mode = "boss" if "boss" in sys.argv else "platformer"
print(f"Training Kirby mode: {mode.upper()}")

"""
Logger Setup
"""
now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir = Path("checkpoints") / "Kirby_Dream_Land" / f"{now}-{mode}"

"""
Load Emulator and Environment
"""
pyboy = PyBoy(game_path, window_type="headless", window_scale=3, debug=False, game_wrapper=True)
ai_settings = KirbyAI()

env = KirbyGymEnv(mode=mode, pyboy=pyboy, observation_type=observation_type)
env.setAISettings(ai_settings)
filtered_actions = ai_settings.GetActions()

env = SkipFrame(env, skip=skip_frames)
env = ResizeObservation(env, game_dimensions)
env = NormalizeObservation(env)
env = FrameStack(env, num_stack=frame_stack)

"""
Load AI Players
"""
config = ai_settings.GetBossHyperParameters() if mode == "boss" else ai_settings.GetHyperParameters()

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

"""
Training Loop
"""
pyboy.set_emulation_speed(0)
save_dir.mkdir(parents=True)
gif_dir = Path("episodes") / "Kirby_Dream_Land" / f"{now}-{mode}"
gif_dir.mkdir(parents=True, exist_ok=True)


logger = MetricLogger(save_dir)

ai_player.saveHyperParameters()

print("Starting Kirby training (headless)")
print(f"Total Episodes: {episodes}")

ai_player.net.train()

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

        if done or time.time() - start > 60:
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
    # print(f"Saved episode GIF to {gif_path}")



ai_player.save()
env.close()
