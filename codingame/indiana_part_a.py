import sys
import math
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

# W: number of columns.
# H: number of rows.
W, H = [int(i) for i in input().split()]
types = {
    "1": ["DOWN"],
    "2": ["LEFT", "RIGHT"],
    "3": ["DOWN"],
    "4": ["LEFT", "DOWN"],
    "5": ["RIGHT", "DOWN"],
    "6": ["LEFT", "RIGHT"],
    "7": ["DOWN"],
    "8": ["DOWN"],
    "9": ["DOWN"],
    "10": ["LEFT"],
    "11": ["RIGHT"],
    "12": ["DOWN"],
    "13": ["DOWN"],
}

opposite = {
    "LEFT": "RIGHT",
    "TOP": "DOWN"
}
opposite.update({v: k for k, v in opposite.items()})

left = lambda x, y: [x - 1, y]
right = lambda x, y: [x + 1, y]
up = lambda x, y: [x, y - 1]
down = lambda x, y: [x, y + 1]

directions = {
    "LEFT": left,
    "RIGHT": right,
    "UP": up,
    "DOWN": down,
}

grid = {}
for i in range(H):
    LINE = input().split(" ")
    # represents a line in the grid and contains W integers. Each integer
    # represents one room of a given type.
    for j, val in enumerate(LINE):
        grid[Point(j, i)] = val

print(grid, file=sys.stderr)
# the coordinate along the X axis of the exit (not useful for this first
# mission, but must be read).
EX = int(input())


def find_direction(cell_type, previous_pos):
    possible_d = types[cell_type]

    if len(possible_d) == 1:
        direction = possible_d[0]
    else:
        if POS in possible_d:
            direction = opposite[POS]
        else:
            direction = [i for i in possible_d if i != opposite[POS]][0]
    return direction

# game loop
previous_pos = None
while 1:
    XI, YI, POS = input().split(" ")
    XI = int(XI)
    YI = int(YI)

    cell_type = grid[Point(XI, YI)]
    print(cell_type, POS, XI, YI, file=sys.stderr)

    if cell_type == "0":
        print("{} {}".format(XI, YI + 1))
        continue
    else:
        possible_d = types[cell_type]
        direction = find_direction(cell_type, previous_pos)
        XI, YI = directions[direction](XI, YI)

    # One line containing the X Y coordinates of the room in which you believe
    # Indy will be on the next turn.
    print("{} {}".format(XI, YI))
    previous_pos = POS
