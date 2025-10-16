import os
import random
from Direction import Direction
from Display import Display
import time
import tkinter as tk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Mlp(nn.Module):
    # Input layer recoit stats: 12 inputs
    # n HiddenLayers
    # Output: 4 directions

    def __init__(self, in_features=12, h1=64, h2=64, out_features=4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


class Model:
    def __init__(self, game, args):
        self.mlp = Mlp()
        self.optimizer = optim.Adam(self.mlp.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.game = game
        self.visual = args.visual == "on"
        self.learning = args.dontlearn
        self.printing = args.noprint
        self.load_path = args.load
        self.save_path = args.save
        self.step_by_step = args.step_by_step
        self.sleep = 0.3 / args.speed if self.visual else 0

        self.num_episodes = 200
        self.max_step = 2000
        self.gamma = 0.4
        self.epsilon = 1 if self.learning else 0
        self.min_epsilon = 0.01 if self.learning else 0
        self.epsilon_decay = 0.95
        self.memory = []

        self.init_error = False
        if self.load_path and os.path.exists(self.load_path):
            try:
                self.mlp.load_state_dict(torch.load(self.load_path))
            except Exception as e:
                self.init_error = True
                print(f"Error: Can't load the file: {e}")
        else:
            if self.load_path:
                self.init_error = True
                print(f"Error: Path does not exist: {self.load_path}")

    def _update_state(self, cell, direction, distance) -> tuple[int, int]:
        direction *= 3
        match cell:
            case 'G':
                direction += 1
                distance = 1
            case 'R':
                direction += 2
                distance = 1
            case 'W':
                if distance != 1:
                    distance = 0
                else:
                    distance = 1
            case 'S':
                if distance != 1:
                    distance = 0
                else:
                    distance = 1
        return (distance, direction)

    def convert_state(self) -> list[int]:
        board = self.game.get_snake_view()
        head: tuple[int, int] = self.game.board.get_snake_head()
        head = (head[0] + 1, head[1] + 1)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        state = [0 for _ in range(0, 12)]
        for i in range(1, 11):
            for d in range(0, 4):
                y = head[0] + i * directions[d][0]
                x = head[1] + i * directions[d][1]
                if y < 0 or y > 11 or x < 0 or x > 11:
                    continue
                cell = board[y][x]
                if cell != '0':
                    d: tuple[int, int] = self._update_state(cell, d, i)
                    if state[d[1]] == 0:
                        state[d[1]] = d[0]
        return state

    def get_opposite(self, d: Direction):
        match d:
            case Direction.UP:
                return Direction.DOWN
            case Direction.DOWN:
                return Direction.UP
            case Direction.LEFT:
                return Direction.RIGHT
            case Direction.RIGHT:
                return Direction.LEFT

    def choose_action(self, state, current_direction):
        directions = list(Direction)
        if random.uniform(0, 1) < self.epsilon:
            direction_ind = random.randint(0, 3)
            direction = directions[direction_ind]
            while (direction == self.get_opposite(current_direction)):
                direction_ind = random.randint(0, 3)
                direction = directions[direction_ind]
            return direction
        else:
            with torch.no_grad():
                q_values = self.mlp(torch.tensor(state, dtype=torch.float32))
                valid_indices = [
                    i for i in range(4)
                    if directions[i] != self.get_opposite(current_direction)
                    ]
                best_in_subset = torch.argmax(q_values[valid_indices]).item()
                direction_ind = valid_indices[best_in_subset]
            direction = directions[direction_ind]
            return direction

    def display_Q(self):
        print(self.Q_table)

    def _encode_state(self, state: list[int]) -> int:
        value = 0
        for i in range(len(state)):
            value = value * 4 + state[i]
        return value

    def training(self, sessions: int):
        display = None
        if self.visual:
            root = tk.Tk()
            display = Display(root, self.step_by_step)
        else:
            root = None
        for _ in range(sessions):
            if self.session(display):
                break
        if self.visual:
            display.close()
        if self.visual:
            root.mainloop()

    def start_episode(self, display, episode):
        if self.printing:
            print(f'Begin episode {episode}!\n')
        if display:
            display.next_step(self.sleep)
        else:
            time.sleep(self.sleep)
        self.game.reset()
        current_direction = self.game.board.get_starting_direction()
        if self.visual:
            display.update(self.game.board.board)
        state = self.convert_state()
        return current_direction, state

    def update_game(self, display):
        if self.visual and display.closed:
            return 1
        if display:
            display.next_step(self.sleep)
        else:
            time.sleep(self.sleep)

    def update_values(self, best_length, best_survival):
        if self.game.get_best_length() > best_length:
            best_length = self.game.get_best_length()
        if self.game.get_best_survival() > best_survival:
            best_survival = self.game.get_best_survival()
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay
            )
        return best_length, best_survival

    def save_model(self):
        try:
            os.makedirs("model", exist_ok=True)
            torch.save(self.mlp.state_dict(), self.save_path)
        except Exception as e:
            print(f"Error: Can't save the file at {self.save_path}: {e}")
            try:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                name = os.path.splitext(self.save_path)[0]
                backup = f"{name}_backup_{timestamp}.pth"
                print(f"Trying to make a backup at {backup}")
                os.makedirs("model", exist_ok=True)
                torch.save(self.mlp.state_dict(), self.save_path)
                print(f"Sucess! File saved at {backup}")
            except Exception as e:
                print(f"Error: Can't make a backup: {e}")
                exit(1)

    def session(self, display):
        self.epsilon = 1 if self.learning else 0
        best_survival, best_length = 0, 3
        for episode in range(self.num_episodes):
            current_direction, state = self.start_episode(display, episode)
            for step in range(self.max_step):
                if self.update_game(display):
                    return 1
                if self.printing:
                    print(f'Step {step}:\n')
                action = self.choose_action(state, current_direction)
                current_direction = action
                if self.printing:
                    print(f'action: {action}')
                reward, end = self.game.move_snake(action)
                next_state = self.convert_state()
                if self.visual:
                    display.update(self.game.board.board)
                if self.printing:
                    print(self.game.print_snake_view())
                if self.learning:
                    self.memory.append((
                        state,
                        action,
                        reward,
                        next_state,
                        end
                        ))
                if (end):
                    break
                state = next_state
                if len(self.memory) > 64:
                    batch = random.sample(self.memory, 64)
                    self.train_batch(batch)
            best_length, best_survival = self.update_values(best_length, best_survival)

        print(f"Best length: {best_length}\nBest survival: {best_survival}")
        if self.update_game(display):
            return 1
        if self.save_path:
            self.save_model()
        return 0

    def train_batch(self, batch):
        states, actions, rewards, next_states, ends = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        ends = torch.tensor(ends, dtype=torch.bool)

        q_values = self.mlp(states)

        with torch.no_grad():
            next_q_values = self.mlp(next_states).max(1)[0]
            target = rewards + self.gamma * next_q_values * (~ends)

        directions = list(Direction)
        action_indices = torch.tensor([directions.index(a) for a in actions])

        q_value = q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)

        loss = self.criterion(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
