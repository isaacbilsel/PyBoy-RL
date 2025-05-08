from asyncio import sleep
import itertools
from pyboy import WindowEvent
from AISettings.AISettingsInterface import AISettingsInterface, Config

class GameState():
    def __init__(self, pyboy):
        game_wrapper = pyboy.game_wrapper()
        "Find the real level progress x"
        level_block = pyboy.get_memory_value(0xC0AB)
        # C202 Mario's X position relative to the screen
        mario_x = pyboy.get_memory_value(0xC202)
        scx = pyboy.botsupport_manager().screen(
        ).tilemap_position_list()[16][0]
        real = (scx - 7) % 16 if (scx - 7) % 16 != 0 else 16

        self.real_x_pos = level_block * 16 + real + mario_x
        self.time_left = game_wrapper.time_left
        self.lives_left = game_wrapper.lives_left
        self.score = game_wrapper.score
        self._level_progress_max = max(game_wrapper._level_progress_max, self.real_x_pos)
        self.world = game_wrapper.world


class MarioAI(AISettingsInterface):
    TARGET_X_POS = 3380          # real_x_pos that corresponds to the flag in 1‑1
    WIN_BONUS    = 10_000
    DEATH_PENALTY= -5_000
    FRAME_PENALTY= -0.2          # subtract every frame → prefer shorter runs
    POTENTIAL_SCALE = 8          # weight for potential‑based shaping
    STUCK_WINDOW = 40            # frames with no forward Δx → extra penalty
	

    def __init__(self):
        # super().__init__()
        self.realMax = [] #[[1,1, 2500], [1,1, 200]]
        self.best_x_this_episode = 0
        self.frames_since_move   = 0

    def _potential(self, x_now: int) -> float:
        return min(x_now / self.TARGET_X_POS, 1.0)

    def GetReward(self, prev: GameState, pyboy):
        # 1. Hard terminal checks ------------------------------------------------
        timeRespawn = pyboy.get_memory_value(0xFFA6)
        if timeRespawn:                       # Mario already dead → nothing to learn
            return 0

        cur = self.GetGameState(pyboy)

        # 2. Per‑step primitives -------------------------------------------------
        # 2.1 Death and win
        death   = self.DEATH_PENALTY if cur.lives_left < prev.lives_left else 0
        win     = 0
        if (cur.world != prev.world) or (cur.real_x_pos >= self.TARGET_X_POS):
            win = (self.WIN_BONUS                       # huge terminal reward
                   + cur.time_left)                     # small early‑finish bonus

        # 2.2 Dense forward progress
        dx = cur.real_x_pos - prev.real_x_pos
        if dx > 0:
            self.frames_since_move = 0
        else:
            self.frames_since_move += 1
        movement = dx                                   # 1 tile → +1 reward

        # 2.3 Potential‑based shaping (Ng & Russell 1999)
        phi_prev = self._potential(prev._level_progress_max)
        phi_cur  = self._potential(cur._level_progress_max)
        shaping  = self.POTENTIAL_SCALE * (phi_cur - phi_prev)

        # 2.4 Small per‑frame cost to push for speed
        frame_cost = self.FRAME_PENALTY

        # 2.5 Stuck penalty: if no forward progress for STUCK_WINDOW frames
        stuck_pen = -2 if self.frames_since_move >= self.STUCK_WINDOW else 0

        # 3. Aggregate -----------------------------------------------------------
        reward = (movement + shaping + frame_cost +
                  death + win + stuck_pen)

        # 4. Book‑keeping for next step ------------------------------------------
        if win or death:                  # new episode starts next reset()
            self.best_x_this_episode = 0
            self.frames_since_move   = 0
        else:
            self.best_x_this_episode = max(self.best_x_this_episode,
                                           cur._level_progress_max)

        return reward

    def GetActions(self):
        baseActions = [WindowEvent.PRESS_ARROW_RIGHT,
                        WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_ARROW_LEFT]

        totalActionsWithRepeats = list(itertools.permutations(baseActions, 2))
        withoutRepeats = []

        for combination in totalActionsWithRepeats:
            reversedCombination = combination[::-1]
            if(reversedCombination not in withoutRepeats):
                withoutRepeats.append(combination)

        filteredActions = [[action] for action in baseActions] + withoutRepeats

        # remove  ['PRESS_ARROW_RIGHT', 'PRESS_ARROW_LEFT']
        del filteredActions[4]

        return filteredActions

    def PrintGameState(self, pyboy):
        gameState = GameState(pyboy)
        game_wrapper = pyboy.game_wrapper()

        print("'Fake', level_progress: ", game_wrapper.level_progress)
        print("'Real', level_progress: ", gameState.real_x_pos)
        print("_level_progress_max: ", gameState._level_progress_max)
        print("World: ", gameState.world)
        print("Time respawn", pyboy.get_memory_value(0xFFA6))

    def GetGameState(self, pyboy):
        return GameState(pyboy)

    def GetHyperParameters(self) -> Config:
        config = Config()
        config.exploration_rate_decay = 0.999
        return config

    def GetLength(self, pyboy):
        result = sum([x[2] for x in self.realMax])

        pyboy.game_wrapper()._level_progress_max = 0 # reset max level progress because game hasnt implemented it
        self.realMax = []

        return result
