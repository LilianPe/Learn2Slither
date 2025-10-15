import os
import numpy as np
import random
from Direction import Direction
from Display import Display
import time
import tkinter as tk
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim

class Mlp(nn.Module):
    # States -> array avec un etat entre 0 et 3 pour
    # chaque case visible serait trop grande
    # On fait donc ce genre de table:
    # state = (
    #     block_up,    green_apple_up,    red_apple_up,
    #     block_down,  green_apple_down,  red_apple_down,
    #     block_left,  green_apple_left,  red_apple_left,
    #     block_right, green_apple_right, red_apple_right
    # )
    # Avec un etat entre 0 et 3
    # 0 = rien
    # 1 = proche (1–3 cases)
    # 2 = moyen (4–6 cases)
    # 3 = loin (7+ cases)


    # Input layer recoit stats: 12 inputs
    # n HiddenLayers
    # Output: 4 directions

    def __init__(self, in_features=12, h1=8, h2=8, out_features=4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return(x)

class Model:
    def __init__(self, game, name, args):
        mlp = Mlp()
        optimizer = optim.Adam(mlp.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        gamma = 0.9
        self.epsilon = 1 if self.learning else 0
        self.min_epsilon = 0.01 if self.learning else 0
        self.epsilon_decay = self.min_epsilon ** (1 / self.num_episodes)
        memory = []

        self.game = game
        self.learning = args.dontlearn
        self.visual = args.visual == "on"
        self.printing = args.noprint
        self.step_by_step = args.step_by_step
        self.sleep = 0.3 / args.speed if self.visual else 0
        self.name = name
        self.n_state = 16777216
        self.n_action = 4
        self.init_error = False
        if os.path.exists(name):
            try:
                self.Q_table = np.load(name)
                if len(self.Q_table) != self.n_state:
                    print("Invalid Q_Table.")
                    self.init_error = True
            except Exception as e:
                self.init_error = True
                print(f"Can't load the file: {e}")
        else:
            if name:
                print("Invalid path.")
                self.init_error = True
            self.Q_table: np.array = np.zeros((self.n_state, self.n_action))
        self.alpha = 0.2
        self.gamma = 0.9
        self.num_episodes = 1000
        self.max_step = 2000

    def _update_state(self, cell, direction, distance) -> tuple[int, int]:
        if distance <= 3:
            distance = 1
        elif distance <= 6:
            distance = 2
        else:
            distance = 3
        direction *= 3
        match cell:
            case 'G':
                direction += 1
            case 'R':
                direction += 2
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
            while(direction == self.get_opposite(current_direction)):
                direction_ind = random.randint(0, 3)
                direction = directions[direction_ind]
            return direction, direction_ind
        else:
            with torch.no_grad():
                q_values = self.mlp(torch.tensor(state, dtype=torch.float32))
                valid_indices = [i for i in range(4) if directions[i] != self.get_opposite(current_direction)]
                best_in_subset = torch.argmax(q_values[valid_indices]).item()
                direction_ind = valid_indices[best_in_subset]
            direction = directions[direction_ind]
            # while(direction == self.get_opposite(current_direction)):
            #     direction_ind = random.randint(0, 3)
            #     direction = directions[direction_ind]
            return direction, direction_ind

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

    def session(self, display):
        self.epsilon = 1 if self.learning else 0
        best_survival, best_length = 0, 3
        for episode in range(self.num_episodes):
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
            state = self._encode_state(self.convert_state()) # State simplifie pour Q-Learning 
            for step in range(self.max_step):
                if self.visual and display.closed:
                    return 1
                if display:
                    display.next_step(self.sleep)
                else:
                    time.sleep(self.sleep)
                if self.printing:
                    print(f'Step {step}:\n')
                action, action_ind = self.choose_action(state, current_direction) # Choisis action par rapport a la Q-table
                current_direction = action
                if self.printing:
                    print(f'action: {action}')
                reward, end = self.game.move_snake(action) # Renvoie la reward de l'action et l'effectue
                if self.visual:
                    display.update(self.game.board.board)
                if self.printing:
                    print(self.game.print_snake_view())
                old_value = self.Q_table[state, action_ind] #------------------------------------------------
                if end:
                    target = reward
                    new = (1 - self.alpha) * old_value + self.alpha * target # ---------------Mise a jour de la Qtable si fin
                    self.Q_table[state, action_ind] = new
                    break
                next_state = self._encode_state(self.convert_state())
                next_max = np.max(self.Q_table[next_state, :]) # --------------------------Mise a jour sinon
                target = reward + self.gamma * next_max
                new_value = (1 - self.alpha) * old_value + self.alpha * target
                self.Q_table[state, action_ind] = new_value # ------------------------------------------------------
                state = next_state
            if self.game.get_best_length() > best_length:
                best_length = self.game.get_best_length()
            if self.game.get_best_survival() > best_survival:
                best_survival = self.game.get_best_survival()
            self.epsilon = max(
                self.min_epsilon,
                self.epsilon * self.epsilon_decay
                )

        print(f"Best length: {best_length}\nBest survival: {best_survival}")
        if display:
            display.next_step(self.sleep)
        else:
            time.sleep(self.sleep)
        if self.learning:
            os.makedirs("model", exist_ok=True)
            np.save(self.name, self.Q_table)
        return 0
