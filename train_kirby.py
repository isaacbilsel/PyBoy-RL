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

"""
Hardcoded Settings
"""
episodes = 1500
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
pyboy = PyBoy(game_path, window_type="headless", window_scale=3, debug=False, game_wrapper=True)
ai_settings = KirbyAI()

env = CustomPyBoyGym(pyboy, observation_type=observation_type)
env.setAISettings(ai_settings)
filtered_actions = ai_settings.GetActions()

env = SkipFrame(env, skip=skip_frames)
env = ResizeObservation(env, game_dimensions)
env = NormalizeObservation(env)
env = FrameStack(env, num_stack=frame_stack)

"""
Load AI Players
"""
ai_player = AIPlayer(
    (frame_stack,) + game_dimensions,
    len(filtered_actions),
    save_dir,
    now,
    ai_settings.GetHyperParameters(),
    duel_dqn=True,
    use_per=True,
    use_noisy=True,
    n_step=3
)

boss_ai_player = AIPlayer(
    (frame_stack,) + game_dimensions,
    len(filtered_actions),
    save_dir_boss,
    now,
    ai_settings.GetBossHyperParameters(),
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
save_dir_boss.mkdir(parents=True)

logger = MetricLogger(save_dir_boss)

ai_player.saveHyperParameters()
boss_ai_player.saveHyperParameters()

print("Starting Kirby training (headless)")
print(f"Total Episodes: {episodes}")

ai_player.net.train()
boss_ai_player.net.train()

for e in range(episodes):
    observation = env.reset()
    start = time.time()

    while True:
        player = boss_ai_player if ai_settings.IsBossActive(pyboy) else ai_player

        action_id = player.act(observation)
        actions = filtered_actions[action_id]
        next_observation, reward, done, info = env.step(actions)

        player.cache(observation, next_observation, action_id, reward, done)
        q, loss = player.learn()

        logger.log_step(reward, loss, q, player.scheduler.get_last_lr())

        observation = next_observation

        if done or time.time() - start > 60:
            break

    logger.log_episode()
    logger.record(
        episode=e,
        epsilon=player.exploration_rate,
        stepsThisEpisode=player.curr_step,
        maxLength=ai_settings.GetLength(pyboy)
    )

ai_player.save()
boss_ai_player.save()
env.close()
