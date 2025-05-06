from collections import deque
import random
import numpy as np
import torch
from AISettings.AISettingsInterface import Config
from model import DDQN, DuelDDQN, NoisyDDQN

class AIPlayer:
    def __init__(self, state_dim, action_space_dim, save_dir, date, config: Config, duel_dqn=False, use_per=False, use_noisy=False, n_step=1):
        self.state_dim = state_dim
        self.action_space_dim = action_space_dim
        self.save_dir = save_dir
        self.date = date
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if use_noisy:
            self.net = NoisyDDQN(self.state_dim, self.action_space_dim).to(device=self.device)
        elif duel_dqn:
            self.net = DuelDDQN(self.state_dim, self.action_space_dim).to(device=self.device)
        else:
            self.net = DDQN(self.state_dim, self.action_space_dim).to(device=self.device)

        self.use_noisy = use_noisy

        self.config = config

        self.exploration_rate = self.config.exploration_rate
        self.exploration_rate_decay = self.config.exploration_rate_decay
        self.exploration_rate_min = self.config.exploration_rate_min
        self.curr_step = 0

        self.use_per = use_per

        if self.use_per:
            self.memory = []
            self.priorities = []
        else:
            self.memory = deque(maxlen=self.config.deque_size)

        self.batch_size = self.config.batch_size
        self.save_every = self.config.save_every

        self.gamma = self.config.gamma
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.learning_rate_decay)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.burnin = self.config.burnin
        self.learn_every = self.config.learn_every
        self.sync_every = self.config.sync_every

        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=self.n_step)

        self.V_min = -10000
        self.V_max = 10000
        self.num_atoms = 51
        self.delta_z = (self.V_max - self.V_min) / (self.num_atoms - 1)
        self.support = torch.linspace(self.V_min, self.V_max, self.num_atoms).to(self.device)

    def act(self, state):
        if self.use_noisy:
            state = np.array(state)
            state = torch.tensor(state).float().to(device=self.device).unsqueeze(0)
            with torch.no_grad():
                neuralNetOutput = self.net(state, model="online")
                probs = torch.softmax(neuralNetOutput, dim=2)
                q_values = torch.sum(probs * self.support, dim=2)
                actionIdx = torch.argmax(q_values, axis=1).item()
        else:
            if random.random() < self.exploration_rate:
                actionIdx = random.randint(0, self.action_space_dim - 1)
            else:
                state = np.array(state)
                state = torch.tensor(state).float().to(device=self.device).unsqueeze(0)
                with torch.no_grad():
                    neuralNetOutput = self.net(state, model="online")
                    probs = torch.softmax(neuralNetOutput, dim=2)
                    q_values = torch.sum(probs * self.support, dim=2)
                    actionIdx = torch.argmax(q_values, axis=1).item()
            self.exploration_rate *= self.exploration_rate_decay
            self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1
        return actionIdx

    def cache(self, state, next_state, action, reward, done):
        state = torch.tensor(np.array(state)).float().to(self.device)
        next_state = torch.tensor(np.array(next_state)).float().to(self.device)
        action = torch.tensor([action]).to(self.device)
        reward = torch.tensor([reward]).to(self.device)
        done = torch.tensor([done]).to(self.device)

        self.n_step_buffer.append((state, next_state, action, reward, done))

        if len(self.n_step_buffer) < self.n_step:
            return

        state_n, next_state_n, action_n, reward_n, done_n = self._get_n_step_info()

        if self.use_per:
            self.memory.append((state_n, next_state_n, action_n, reward_n, done_n))
            max_priority = max(self.priorities, default=1.0)
            self.priorities.append(max_priority)
            if len(self.memory) > self.config.deque_size:
                self.memory.pop(0)
                self.priorities.pop(0)
        else:
            self.memory.append((state_n, next_state_n, action_n, reward_n, done_n))

        if done:
            self.n_step_buffer.clear()

    def _get_n_step_info(self):
        reward, next_state, done = self.n_step_buffer[-1][3], self.n_step_buffer[-1][1], self.n_step_buffer[-1][4]
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[3], transition[1], transition[4]
            reward = r + self.gamma * reward * (1 - d.float())
            next_state, done = (n_s, d) if d else (next_state, done)
        state, _, action, _, _ = self.n_step_buffer[0]
        return state, next_state, action, reward, done

    def recall(self):
        if self.use_per:
            priorities = np.array(self.priorities)
            probs = priorities ** 0.6
            probs /= probs.sum()
            indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
            batch = [self.memory[idx] for idx in indices]
            weights = (len(self.memory) * probs[indices]) ** (-0.4)
            weights /= weights.max()
            weights = torch.tensor(weights, device=self.device, dtype=torch.float)
            state, next_state, action, reward, done = map(torch.stack, zip(*batch))
            return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze(), indices, weights
        else:
            batch = random.sample(self.memory, self.batch_size)
            state, next_state, action, reward, done = map(torch.stack, zip(*batch))
            return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        if self.curr_step % self.save_every == 0:
            self.save()
        if self.curr_step < self.burnin:
            return None, None
        if self.curr_step % self.learn_every != 0:
            return None, None

        if self.use_per:
            state, next_state, action, reward, done, indices, weights = self.recall()
        else:
            state, next_state, action, reward, done = self.recall()

        self.actions_taken = action
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)

        td_estimate_mean = td_est.mean().item()

        if self.use_per:
            loss = self.update_Q_online(td_est, td_tgt, weights)
        else:
            loss = self.update_Q_online(td_est, td_tgt)

        if self.use_noisy:
            self.net.reset_noise()

        return td_estimate_mean, loss

    def update_Q_online(self, predicted_dist, target_dist, weights=None):
        EPS = 1e-5
        probs = torch.softmax(predicted_dist, dim=2)
        actions = self.actions_taken
        action_probs = probs[range(self.batch_size), actions]
        action_probs = torch.clamp(action_probs, EPS, 1.0)
        target_dist = torch.clamp(target_dist, EPS, 1.0)
        if weights is None:
            loss = - (target_dist * action_probs.log()).sum(dim=1).mean()
        else:
            loss = - (weights * (target_dist * action_probs.log()).sum(dim=1)).mean()
        self.optimizer.zero_grad()
        if not torch.isfinite(loss):
            print("NaN/Inf loss detected – skipping this update")
            return loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def td_estimate(self, state, action):
        return self.net(state, model="online")

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        batch_size = reward.size(0)
        next_dist = self.net(next_state, model="online")
        next_probs = torch.softmax(next_dist, dim=2)
        next_q = torch.sum(next_probs * self.support, dim=2)
        next_action = torch.argmax(next_q, dim=1)
        next_dist_target = self.net(next_state, model="target")
        next_dist_target = torch.softmax(next_dist_target, dim=2)
        next_dist_target = next_dist_target[range(batch_size), next_action]
        projected_dist = self._projection(next_dist_target, reward, done)
        return projected_dist

    def _projection(self, next_dist, rewards, dones):
        batch_size = rewards.size(0)
        projected_dist = torch.zeros((batch_size, self.num_atoms), device=self.device)
        for j in range(self.num_atoms):
            tz_j = torch.clamp(rewards + (1 - dones.float()) * (self.gamma ** self.n_step) * (self.V_min + j * self.delta_z), self.V_min, self.V_max)
            b_j = (tz_j - self.V_min) / self.delta_z
            l = b_j.floor().long()
            u = b_j.ceil().long()
            eq_mask = (u == l)
            projected_dist.view(-1).index_add_(0, (l + self.num_atoms * torch.arange(batch_size, device=self.device)).view(-1), next_dist[:, j] * (u.float() - b_j + eq_mask.float()))
            projected_dist.view(-1).index_add_(0, (u + self.num_atoms * torch.arange(batch_size, device=self.device)).view(-1), next_dist[:, j] * (b_j - l.float() + eq_mask.float()))
        projected_dist = projected_dist / projected_dist.sum(dim=1, keepdim=True)
        projected_dist = torch.clamp(projected_dist, 1e-5, 1.0)
        return projected_dist

    def loadModel(self, path):
        dt = torch.load(path, map_location=torch.device(self.device))
        self.net.load_state_dict(dt["model"])
        self.exploration_rate = dt["exploration_rate"]
        print(f"Loading model at {path} with exploration rate {self.exploration_rate}")

    def saveHyperParameters(self):
        save_HyperParameters = self.save_dir / "hyperparameters"
        with open(save_HyperParameters, "w") as f:
            f.write(f"exploration_rate = {self.config.exploration_rate}\\n")
            f.write(f"exploration_rate_decay = {self.config.exploration_rate_decay}\\n")
            f.write(f"exploration_rate_min = {self.config.exploration_rate_min}\\n")
            f.write(f"deque_size = {self.config.deque_size}\\n")
            f.write(f"batch_size = {self.config.batch_size}\\n")
            f.write(f"gamma = {self.config.gamma}\\n")
            f.write(f"learning_rate = {self.config.learning_rate}\\n")
            f.write(f"learning_rate_decay = {self.config.learning_rate_decay}\\n")
            f.write(f"burnin = {self.config.burnin}\\n")
            f.write(f"learn_every = {self.config.learn_every}\\n")
            f.write(f"sync_every = {self.config.sync_every}\\n")
            f.write(f"n_step = {self.n_step}\\n")

    def save(self):
        save_path = self.save_dir / f"mario_net_0{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate), save_path)
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")
