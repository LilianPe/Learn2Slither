import os
import numpy as np
import random
from Direction import Direction
from Display import Display
import time
import tkinter as tk


class Model:
    def __init__(self, game, name, args):
        # States -> Q-table avec un etat entre 0 et 3 pour
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
        # On a donc 4**12 = 16 777 216 etats

        self.game = game
        self.learning = args.dontlearn
        self.visual = args.visual == "on"
        self.printing = args.noprint
        self.sleep = 0.3 if self.visual else 0
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
            self.Q_table: np.array = np.zeros((self.n_state, self.n_action))
        self.alpha = 0.2
        self.gamma = 0.9
        self.epsilon = 1 if self.learning else 0
        self.min_epsilon = 0.01 if self.learning else 0
        self.epsilon_decay = 0.9995
        self.num_episodes = 4
        self.max_step = 200

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

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)
        else:
            return np.argmax(self.Q_table[state, :])

    def display_Q(self):
        print(self.Q_table)

    def _encode_state(self, state: list[int]) -> int:
        value = 0
        for i in range(len(state)):
            value = value * 4 + state[i]
        return value

    def training(self, sessions: int):
        if self.visual:
            root = tk.Tk()
            display = Display(root)
        else:
            root = None
        for _ in range(sessions):
            if self.session(display):
                break
        display.close()
        if self.visual:
            root.mainloop()

    def session(self, display):
        best_survival, best_length = 0, 0
        directions: list[Direction] = [
            Direction.UP,
            Direction.DOWN,
            Direction.LEFT,
            Direction.RIGHT
            ]
        self.epsilon = 1 if self.learning else 0
        for episode in range(self.num_episodes):
            if self.printing:
                print(f'Begin episode {episode}!\n')
            time.sleep(self.sleep)
            if self.game.get_best_length() > best_length:
                best_length = self.game.get_best_length()
            if self.game.get_best_survival() > best_survival:
                best_survival = self.game.get_best_survival()
            self.game.reset()
            if self.visual:
                display.update(self.game.board.board)
            state = self._encode_state(self.convert_state())
            for step in range(self.max_step):
                if display.closed:
                    return 1
                time.sleep(self.sleep)
                if self.printing:
                    print(f'Step {step}:\n')
                action = self.choose_action(state)
                if self.printing:
                    print(f'action: {action}')
                reward, end = self.game.move_snake(directions[action])
                if self.visual:
                    display.update(self.game.board.board)
                if self.printing:
                    print(self.game.print_snake_view())
                old_value = self.Q_table[state, action]
                if end:
                    target = reward + self.gamma
                    new = (1 - self.alpha) * old_value + self.alpha * target
                    self.Q_table[state, action] = new
                    break
                next_state = self._encode_state(self.convert_state())
                next_max = np.max(self.Q_table[next_state, :])
                target = reward + self.gamma * next_max
                new_value = (1 - self.alpha) * old_value + self.alpha * target
                self.Q_table[state, action] = new_value
                state = next_state

            self.epsilon = max(
                self.min_epsilon,
                self.epsilon * self.epsilon_decay
                )

        print(f"Best length: {best_length}\nBest survival: {best_survival}")
        time.sleep(0.2)
        os.makedirs("model", exist_ok=True)
        np.save(self.name, self.Q_table)
        return 0

    # def update_stats(self, best_survival, best_length, best_ep_length, best_ep_survival) :
    #     s = best_survival if best_survival > best_ep_survival else best_ep_survival
    #     l = best_length if best_length > best_ep_length else best_ep_length
    #     return s, l 

    # def session(self, display):
    #     directions: list[Direction] = [
    #         Direction.UP,
    #         Direction.DOWN,
    #         Direction.LEFT,
    #         Direction.RIGHT
    #         ]
    #     self.epsilon = 1 if self.learning else 0
    #     best_survival, best_length = 0, 0
    #     for e in range(self.num_episodes):
    #         best_ep_survival, best_ep_length = self.episode(e, display, directions)
    #         best_survival, best_length = self.update_stats(best_survival, best_length, best_ep_length, best_ep_survival)
    #     print(f"Best length: {best_length}\nBest survival: {best_survival}")
    #     time.sleep(0.2)
    #     os.makedirs("model", exist_ok=True)
    #     np.save(self.name, self.Q_table)
    #     return 0

    # def episode(self, episode, display, directions):
    #     best_survival, best_length = 0, 0
    #     if self.printing:
    #         print(f'Begin episode {episode}!\n')
    #     time.sleep(self.sleep)
    #     self.game.reset()
    #     if self.visual:
    #         display.update(self.game.board.board)
    #     state = self._encode_state(self.convert_state())
    #     for step in range(self.max_step):
    #         if display.closed:
    #             return 1
    #         time.sleep(self.sleep)
    #         if self.printing:
    #             print(f'Step {step}:\n')
    #         action = self.choose_action(state)
    #         if self.printing:
    #             print(f'action: {action}')
    #         reward, end = self.game.move_snake(directions[action])
    #         if self.visual:
    #             display.update(self.game.board.board)
    #         if self.printing:
    #             print(self.game.print_snake_view())
    #         old_value = self.Q_table[state, action]
    #         if end:
    #             target = reward + self.gamma
    #             new = (1 - self.alpha) * old_value + self.alpha * target
    #             self.Q_table[state, action] = new
    #             break
    #         next_state = self._encode_state(self.convert_state())
    #         next_max = np.max(self.Q_table[next_state, :])
    #         target = reward + self.gamma * next_max
    #         new_value = (1 - self.alpha) * old_value + self.alpha * target
    #         self.Q_table[state, action] = new_value
    #         state = next_state

    #     if self.game.get_best_length() > best_length:
    #         best_length = self.game.get_best_length()
    #     if self.game.get_best_survival() > best_survival:
    #         best_survival = self.game.get_best_survival()
    #     self.epsilon = max(
    #         self.min_epsilon,
    #         self.epsilon * self.epsilon_decay
    #         )
        
    #     return best_survival, best_length
