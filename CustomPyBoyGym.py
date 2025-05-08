from pyboy.pyboy import *
from AISettings.AISettingsInterface import AISettingsInterface
import os

class CustomPyBoyGym(PyBoyGymEnv):
    def step(self, list_actions):
        """
            Simultanious action implemention
        """
        info = {}

        previousGameState = self.aiSettings.GetGameState(self.pyboy)

        if list_actions[0] == self._DO_NOTHING:
            pyboy_done = self.pyboy.tick()
        else:
            # release buttons if not pressed now but were pressed in the past
            for pressedFromBefore in [pressed for pressed in self._button_is_pressed if self._button_is_pressed[pressed] == True]: # get all buttons currently pressed
                if pressedFromBefore not in list_actions:
                    release = self._release_button[pressedFromBefore]
                    self.pyboy.send_input(release)
                    self._button_is_pressed[release] = False

            # press buttons we want to press
            for buttonToPress in list_actions:
                self.pyboy.send_input(buttonToPress)
                self._button_is_pressed[buttonToPress] = True # update status of the button

            pyboy_done = self.pyboy.tick()

        # reward 
        reward = self.aiSettings.GetReward(previousGameState, self.pyboy)

        observation = self._get_observation()

        done = pyboy_done or self.pyboy.game_wrapper().game_over()
        return observation, reward, done, info

    def setAISettings(self, aisettings: AISettingsInterface):
        self.aiSettings = aisettings

    def reset(self, **kwargs):
        """ Reset (or start) the gym environment throught the game_wrapper """
        if not self._started:
            self.game_wrapper.start_game(**self._kwargs)
            self._started = True
        else:
            self.game_wrapper.reset_game()

        # release buttons if not pressed now but were pressed in the past
        for pressedFromBefore in [pressed for pressed in self._button_is_pressed if self._button_is_pressed[pressed] == True]: # get all buttons currently pressed
            self.pyboy.send_input(self._release_button[pressedFromBefore])
        self._button_is_pressed = {button: False for button in self._buttons} # reset all buttons

        return self._get_observation()


class KirbyGymEnv(CustomPyBoyGym):
    def __init__(self, mode="platformer", *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mode in ["platformer", "boss", "tree"], "Invalid mode for KirbyGymEnv"
        self.mode = mode
        self.state_path = {
            "platformer": "states/kirby_platform.gb.state",
            "boss": "states/kirby_boss.gb.state",
            "tree": "states/kirby_tree.gb.state"
        }

    def step(self, list_actions):
        observation, reward, done, info = super().step(list_actions)
        done = done or self.aiSettings.IsDone()
        return observation, reward, done, info

    def reset(self, **kwargs):
        if not self._started:
            self.game_wrapper.start_game(**self._kwargs)
            self._started = True
            for _ in range(50):  # let game initialize for 200 frames
                self.pyboy.tick()
            # After first launch, load checkpoint
            self.load_state()
        else:
            self.load_state()

        # Reset buttons
        for pressedFromBefore in [
            pressed for pressed in self._button_is_pressed
            if self._button_is_pressed[pressed]
        ]:
            self.pyboy.send_input(self._release_button[pressedFromBefore])
        self._button_is_pressed = {button: False for button in self._buttons}

        self.aiSettings.Reset()
        
        return self._get_observation()

    def load_state(self):
        path = self.state_path[self.mode]
        if not os.path.exists(path):
            raise FileNotFoundError(f"State file not found: {path}")
        with open(path, "rb") as f:
            self.pyboy.load_state(f)

    def register_state(self, mode, path):
        self.state_path[mode] = path

