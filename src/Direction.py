from enum import Enum


class Direction(Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"

direction_map = {
    Direction.UP:    [Direction.UP, Direction.LEFT, Direction.RIGHT],
    Direction.DOWN:  [Direction.DOWN, Direction.RIGHT, Direction.LEFT],
    Direction.LEFT:  [Direction.LEFT, Direction.DOWN, Direction.UP],
    Direction.RIGHT: [Direction.RIGHT, Direction.UP, Direction.DOWN],
}