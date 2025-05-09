from collections import deque
import itertools
from pyboy import WindowEvent
from AISettings.AISettingsInterface import AISettingsInterface, Config


class GameState():
    """
    Snapshot of the important RAM‑based game variables for reward shaping.
    """
    def __init__(self, pyboy):
        gw = pyboy.game_wrapper()

        # --- Horizontal progress ------------------------------------------------
        level_block = pyboy.get_memory_value(0xC0AB)         # coarse tile block
        mario_x     = pyboy.get_memory_value(0xC202)         # fine X within screen
        scx         = pyboy.botsupport_manager().screen().tilemap_position_list()[16][0]
        real        = (scx - 7) % 16 or 16                   # screen offset
        self.real_x_pos = level_block * 16 + real + mario_x

        # --- Vertical position (used for fall / trap detection) -----------------
        # 0xC20C = Mario's Y position relative to screen in SML
        self.y_pos = pyboy.get_memory_value(0xC20C)

        # --- Other game stats ---------------------------------------------------
        self.time_left   = gw.time_left
        self.lives_left  = gw.lives_left
        self.score       = gw.score
        self.world       = gw.world  # (world, level)
        self._level_progress_max = max(gw._level_progress_max, self.real_x_pos)


class MarioAI(AISettingsInterface):
    TARGET_X_POS   = 3380          # ≈ flag in 1‑1; only for potential shaping
    WIN_BONUS      = 10_000
    DEATH_PENALTY  = -5_000
    FRAME_PENALTY  = -0.2
    POTENTIAL_SCALE = 8
    STUCK_WINDOW    = 40           # frames

    # --------------------------------------------------------------------- #
    def __init__(self):
        super().__init__()
        self.best_x_this_episode = 0
        self.frames_since_move   = 0
        self.realMax = []

        # Trap‑avoidance state
        self.position_history = deque(maxlen=40)
        self.visited_tiles    = set()
        self.prev_tile_count  = 0
        self.frames_still     = 0

    # --------------------------------------------------------------------- #
    def _potential(self, x_now: int) -> float:
        return min(x_now / self.TARGET_X_POS, 1.0)

    # --------------------------------------------------------------------- #
    def GetReward(self, prev: GameState, pyboy):
        # -- Respawn wait: no learning signal -----------------------------------
        if pyboy.get_memory_value(0xFFA6):   # countdown when dead
            return 0

        cur = self.GetGameState(pyboy)

        level_key = (cur.world[0], cur.world[1])
        found = False
        for rec in self.realMax:
            if (rec[0], rec[1]) == level_key:
                rec[2] = max(rec[2], cur._level_progress_max)
                found = True
                break
        if not found:
            self.realMax.append(
                [level_key[0], level_key[1], cur._level_progress_max]
            )

        # -- Terminal rewards / penalties ---------------------------------------
        death = self.DEATH_PENALTY if cur.lives_left < prev.lives_left else 0
        win   = 0
        if (cur.world != prev.world) or (cur.real_x_pos >= self.TARGET_X_POS):
            win = self.WIN_BONUS + cur.time_left

        # -- Dense horizontal progress ------------------------------------------
        dx = cur.real_x_pos - prev.real_x_pos
        self.frames_since_move = 0 if dx > 0 else self.frames_since_move + 1
        movement = dx

        # -- Potential‑based shaping --------------------------------------------
        shaping = self.POTENTIAL_SCALE * (
            self._potential(cur._level_progress_max)
            - self._potential(prev._level_progress_max)
        )

        # -- Frame cost ----------------------------------------------------------
        frame_cost = self.FRAME_PENALTY

        # -- Stillness penalty (velocity ≈ 0 for many frames) -------------------
        vel_mag = abs(dx)
        self.frames_still = 0 if vel_mag >= 0.5 else self.frames_still + 1
        still_pen = -0.1 * self.frames_still if self.frames_still > 30 else 0

        # -- Confinement penalty: tiny X‑range for a while -----------------------
        self.position_history.append(cur.real_x_pos)
        motion_range = max(self.position_history) - min(self.position_history)
        box_penalty = -0.05 * self.frames_still if len(self.position_history) == 40 and motion_range < 10 else 0

        # -- Exploration bonus ---------------------------------------------------
        tile_col = cur.real_x_pos // 8
        if tile_col not in self.visited_tiles:
            explore_bonus = 1
            self.visited_tiles.add(tile_col)
            self.prev_tile_count += 1
        else:
            explore_bonus = 0

        # -- Falling penalty (into wells / pits) ---------------------------------
        fall_pen = -2 if cur.y_pos > prev.y_pos else 0

        # -- Stuck penalty (no forward dx for long window) -----------------------
        stuck_pen = -2 if self.frames_since_move >= self.STUCK_WINDOW else 0

        # -- Aggregate -----------------------------------------------------------
        reward = (
            movement + shaping + frame_cost
            + death + win
            + still_pen + box_penalty + stuck_pen + fall_pen
            + explore_bonus
        )

        # -- Reset episodic state if needed --------------------------------------
        if win or death:
            self._reset_episode_state()
        else:
            self.best_x_this_episode = max(self.best_x_this_episode,
                                           cur._level_progress_max)
        return reward

    # --------------------------------------------------------------------- #
    def _reset_episode_state(self):
        self.best_x_this_episode = 0
        self.frames_since_move   = 0
        self.frames_still        = 0
        self.position_history.clear()
        self.visited_tiles.clear()
        self.prev_tile_count     = 0

    # --------------------------------------------------------------------- #
    # Unchanged helper methods below
    def GetActions(self):
        base = [WindowEvent.PRESS_ARROW_RIGHT,
                WindowEvent.PRESS_BUTTON_A,
                WindowEvent.PRESS_ARROW_LEFT]

        combos = list(itertools.permutations(base, 2))
        combos = [c for c in combos if c[::-1] not in combos]  # dedup reverse
        actions = [[a] for a in base] + combos
        actions = [a for a in actions if a !=
                   [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_ARROW_LEFT]]
        return actions

    def GetGameState(self, pyboy): return GameState(pyboy)

    def PrintGameState(self, pyboy):
        gs = GameState(pyboy)
        gw = pyboy.game_wrapper()
        print(f"'Fake' level_progress: {gw.level_progress}")
        print(f"'Real' level_progress: {gs.real_x_pos}")
        print(f"_level_progress_max  : {gs._level_progress_max}")
        print(f"World               : {gs.world}")
        print(f"Time respawn        : {pyboy.get_memory_value(0xFFA6)}")

    def GetHyperParameters(self) -> Config:
        cfg = Config()
        cfg.exploration_rate_decay = 0.999
        return cfg

    def GetLength(self, pyboy):
        """
        Sum of per‑level max progress for the whole run.
        Works even when the list is empty or missing.
        """
        if not hasattr(self, "realMax") or not self.realMax:
            return 0

        total = sum(rec[2] for rec in self.realMax)

        # reset shared state for the next run
        pyboy.game_wrapper()._level_progress_max = 0
        self.realMax.clear()
        return total