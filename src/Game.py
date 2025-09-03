from Board import Board
from Direction import Direction


class Game:
    def __init__(self):
        """Init the snake game"""
        self.actual_direction = None
        self.board: Board = Board()
        # ajouter best length et best survival et des getters30

    def move_snake(self, direction: Direction) -> int:
        if not self.board.get_snake_is_alive():
            return
        reward = self.board.move_snake(direction)
        return reward, not self.board.get_snake_is_alive()
        # Ajouter plus tard des trigger, Green apple -> recompense, Red apple -> malus, mort -> Gros malus

    def print_board(self):
        """Display the actual board"""
        if self.board.get_snake_is_alive():
            self.board.display()
        else:
            print("SNAKE IS DEAD!")
        print("\n\n")

    def _generate_view(self) -> list[list[str]]:
        """Return the view of the snake"""
        b = self.board.board
        items = ['0', 'H', 'S', 'G', 'R']
        view = []
        head: tuple[int, int] = self.board.get_snake_head()
        for i in range(-1, 11):
            line = []
            # print(view)
            for j in range(-1, 11):
                # if (i < 0 or i > 9) and (j < 0 or j > 9):
                #     continue
                if i == head[0] or j == head[1]:
                    if i < 0 or i > 9 or j < 0 or j > 9:
                        line.append("W")
                    else:
                        line.append(items[b[i][j]])
                else:
                    line.append(" ")
            view.append(line)
        return view

    def print_snake_view(self):
        if not self.board.get_snake_is_alive():
            print("SNAKE IS DEAD!")
            return
        view = self._generate_view()
        for lines in view:
            line = ""
            for cell in lines:
                line += cell
            print(line)
    
    def get_snake_view(self):
        return self._generate_view()

    def get_snake_body(self) -> list[tuple[int, int]]:
        """Return snake body"""
        return self.board.get_snake_body()

    def reset(self):
        self.board = Board()
        self.actual_direction = None
    
    def get_best_length(self):
        return self.board.get_best_length()

    def get_best_survival(self):
        return self.board.get_best_survival()