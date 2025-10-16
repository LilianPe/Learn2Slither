import tkinter as tk
import time


class Display:
    def __init__(self, root, step_by_step):
        self.root = root
        self.root.title("Learn2Slither")
        self.closed = False
        self.canvas = tk.Canvas(
            self.root,
            width=400,
            height=400,
            bg="black"
        )
        if step_by_step:
            root.bind("<Right>", self._on_next_step)
            root.bind("<space>", self._on_next_step)
        self.pressed = False
        self.step_by_step = step_by_step
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.cells = []
        self.board = [[0 for _ in range(0, 10)] for _ in range(0, 10)]
        self.init_display()

    def init_display(self):
        if self.closed:
            return
        self.canvas.pack()
        for i in range(10):
            row = []
            for j in range(10):
                x1, y1 = j*40, i*40
                x2, y2 = x1+40, y1+40
                rect = self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill="white", outline="gray"
                )
                row.append(rect)
            self.cells.append(row)
        self.root.update_idletasks()
        self.root.update()

    def _on_next_step(self, event):
        self.pressed = True

    def display_board(self):
        if self.closed:
            return
        COLORS = {
            0: "white",   # vide
            1: "green",   # tête
            2: "lightgreen",  # corps
            3: "blue",    # pomme verte
            4: "red",     # pomme rouge
        }
        for i in range(10):
            for j in range(10):
                val = self.board[i][j]
                self.canvas.itemconfig(self.cells[i][j], fill=COLORS[val])
        self.root.update_idletasks()
        self.root.update()

    def update(self, board):
        self.check_window()
        if self.closed:
            return
        self.board = board
        self.display_board()

    def close(self):
        if not self.closed:
            self.pressed = True
            self.root.destroy()

    def check_window(self):
        try:
            self.root.winfo_exists()
        except tk.TclError:
            self.closed = True
            return False

    def next_step(self, time_to_sleep):
        self.check_window()
        if self.closed:
            return
        if not self.step_by_step:
            time.sleep(time_to_sleep)
            return
        while not self.pressed:
            self.root.update_idletasks()
            self.root.update()
            time.sleep(0.05)
        self.pressed = False
