import random
from Direction import Direction


class Board:
    def __init__(self):
        """
        Init the snake board.
        1: HEAD
        2: BODY
        3: GREEN APPLE
        4: RED APPLE
        """
        self.board = [[0 for _ in range(0, 10)] for _ in range(0, 10)]
        self._snake_head: tuple[int, int] = None
        self._snake_tail: tuple[int, int] = None
        self._snake_body: list[tuple[int, int]] = []
        self._snake_is_alive = True
        self.reward = 0
        self._best_length = 3
        self.best_survival = 0
        self._starting_direction = None
        self._generate_snake()
        self._generate_green_apple()
        self._generate_green_apple()
        self._generate_red_apple()

    def _generate_snake_head(self) -> None:
        """Generate the snake head"""
        positions = [(y, x) for x in range(10) for y in range(10)]
        busy = [(y, x) for (y, x) in positions if self.board[y][x] != 0]
        free: list[tuple[int, int]] = list(set(positions) - set(busy))
        n: int = random.randint(0, len(free) - 1)
        self._snake_head = free[n]
        self._snake_body.append(self._snake_head)
        self.board[self._snake_head[0]][self._snake_head[1]] = 1

    def _is_valid(self, cell) -> bool:
        """Check if a cell is valid (in the board and empty)"""
        for n in cell:
            if n < 0 or n > 9:
                return False
        if self.board[cell[0]][cell[1]] != 0:
            return False
        return True

    def _add_tuple(self, a, b) -> tuple[int, int]:
        """Add two tuples content"""
        return (a[0] + b[0], a[1] + b[1])

    def _generate_snake_body(self) -> None:
        """Generate the snake body"""
        cell = self._snake_tail if self._snake_tail else self._snake_head
        next_cells: list[tuple[int, int]] = []
        directions: list[tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for d in directions:
            tmp = self._add_tuple(cell, d)
            if self._is_valid(tmp):
                next_cells.append(tmp)
        if (len(next_cells) > 1):
            n: int = random.randint(0, len(next_cells) - 1)
        else:
            n: int = 0
        self._snake_tail = next_cells[n]
        self._snake_body.append(self._snake_tail)
        self.board[self._snake_tail[0]][self._snake_tail[1]] = 2

    def _update_starting_direction(self):
        if self._snake_head[0] - 1 >= 0 and self._snake_head[0] - 1 <= 9 and self.board[self._snake_head[0] - 1][self._snake_head[1]] == 2:
            self._starting_direction = Direction.DOWN
        elif self._snake_head[0] + 1 >= 0 and self._snake_head[0] + 1 <= 9 and self.board[self._snake_head[0] + 1][self._snake_head[1]] == 2:
            self._starting_direction = Direction.UP
        elif self._snake_head[1] - 1 >= 0 and self._snake_head[1] - 1 <= 9 and self.board[self._snake_head[0]][self._snake_head[1] - 1] == 2:
            self._starting_direction = Direction.RIGHT
        elif self._snake_head[1] + 1 >= 0 and self._snake_head[1] + 1 <= 9 and self.board[self._snake_head[0]][self._snake_head[1] + 1] == 2:
            self._starting_direction = Direction.LEFT

    def _generate_snake(self) -> None:
        """Generate the snake on the board"""
        self._generate_snake_head()
        self._generate_snake_body()
        self._update_starting_direction()
        self._generate_snake_body()

    def _generate_green_apple(self) -> None:
        """Generate a green apple on an empty cell of the board"""
        positions = [(y, x) for x in range(10) for y in range(10)]
        busy = [(y, x) for (y, x) in positions if self.board[y][x] != 0]
        free: list[tuple[int, int]] = list(set(positions) - set(busy))
        n: int = random.randint(0, len(free) - 1)
        pos: tuple[int, int] = free[n]
        self.board[pos[0]][pos[1]] = 3

    def _generate_red_apple(self) -> None:
        """Generate a red apple on an empty cell of the board"""
        positions = [(y, x) for x in range(10) for y in range(10)]
        busy = [(y, x) for (y, x) in positions if self.board[y][x] != 0]
        free: list[tuple[int, int]] = list(set(positions) - set(busy))
        n: int = random.randint(0, len(free) - 1)
        pos: tuple[int, int] = free[n]
        self.board[pos[0]][pos[1]] = 4

    def _update_body(self, apple):
        match(apple):
            case 0:
                cell = self._snake_body.pop()
                self.board[cell[0]][cell[1]] = 0
                self._snake_tail = self._snake_body[-1]
                self.reward = -2.5
            case 3:
                self._generate_green_apple()
                self.reward = 40
                self._best_length += 1
            case 4:
                cell = self._snake_body.pop()
                self.board[cell[0]][cell[1]] = 0
                cell = self._snake_body.pop()
                self.board[cell[0]][cell[1]] = 0
                if len(self._snake_body):
                    body_len = len(self._snake_body)
                    self._snake_tail = self._snake_body[body_len - 1]
                self._generate_red_apple()
                self.reward = -40
                self._best_length -= 1

    def _check_new_pos(self, pos: tuple[int, int]) -> int:
        for i in pos:
            if i < 0 or i > 9:
                self._snake_is_alive = False
                self.reward = -80
                return 0
        if self.board[pos[0]][pos[1]] == 2:
            self._snake_is_alive = False
            self.reward = -80
            return 0
        self._update_body(self.board[pos[0]][pos[1]])
        if len(self._snake_body) == 0:
            self.reward = -80
            self._snake_is_alive = False
            return 0
        self.best_survival += 1
        return 1

    def _update_head_position(self, d) -> int:
        """Update head position for d move"""
        head = self._snake_head
        ds = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        prev_head = self._snake_head
        match d:
            case Direction.UP:
                self._snake_head = self._add_tuple(self._snake_head, ds[0])
            case Direction.DOWN:
                self._snake_head = self._add_tuple(self._snake_head, ds[1])
            case Direction.LEFT:
                self._snake_head = self._add_tuple(self._snake_head, ds[2])
            case Direction.RIGHT:
                self._snake_head = self._add_tuple(self._snake_head, ds[3])
        self._snake_body.insert(0, self._snake_head)
        head = self._snake_head
        if self._check_new_pos(head):
            if len(self._snake_body) > 1:
                self.board[prev_head[0]][prev_head[1]] = 2
            self.board[head[0]][head[1]] = 1

    def move_snake(self, d: Direction) -> int:
        """Handle snake movement"""
        # self._snake_body.clear()
        self._update_head_position(d)
        return self.reward

    def display(self):
        """Display the actual board"""
        for ligne in self.board:
            print(ligne)

    def get_snake_body(self) -> list[tuple[int, int]]:
        return (self._snake_body)

    def get_snake_is_alive(self) -> bool:
        return (self._snake_is_alive)

    def get_snake_head(self) -> tuple[int, int]:
        return self._snake_head

    def get_best_survival(self):
        return self.best_survival

    def get_best_length(self):
        return self._best_length

    def get_starting_direction(self):
        return self._starting_direction
