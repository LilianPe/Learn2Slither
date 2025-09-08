import tkinter as tk


class Display:
    def __init__(self, root):
        self.root = root
        self.root.title("Learn2Slither")
        self.closed = False
        self.canvas = tk.Canvas(
            self.root,
            width=400,
            height=400,
            bg="black"
        )
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
            self.root.destroy()

    def check_window(self):
        try:
            self.root.winfo_exists()
        except tk.TclError:
            self.closed = True
            return False
