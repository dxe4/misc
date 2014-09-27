import sys
import math
from itertools import product
# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

# W: width of the building.
# H: height of the building.
W, H = [int(i) for i in input().split()]
N = int(input())  # maximum number of turns before game over.
X0, Y0 = [int(i) for i in input().split()]

direction = {
    'U': (0, 1),
    'UR': (1, 1),
    'R': (1, 0),
    'DR': (1, -1),
    'D': (0, -1),
    'DL': (-1, -1),
    'L': (-1, 0),
    'UL': (-1, 1),
}

opposite = {
    'UD': (False, True),
    'RL': (True, False),
    'DU': (False, True),
    'LR': (True, False),
}


def is_opposite(BOMB_DIR, previous):
    if previous is None:
        return (False, False)
    else:
        result = []
        for pair in product(BOMB_DIR, previous):
            try:
                result.append(opposite[''.join(pair)])
            except KeyError:
                pass  # Not opposite
        if not result:
            return (False, False)
        # This is probably an itertools job
        x_opposite = bool([i[0] for i in result if i[0] is True])
        y_opposite = bool([i[1] for i in result if i[1] is True])
        return x_opposite, y_opposite


def should_scale(BOMB_DIR, previous):
    x_opposite, y_opposite = is_opposite(BOMB_DIR, previous)
    return not x_opposite, not y_opposite

# game loop
previous = None
x_reverted = False
y_reverted = False

# The factor has to change according to previous pos in case of opposite
factor = 1

while 1:
    # the direction of the bombs from batman's current location (U, UR, R, DR,
    # D, DL, L or UL)
    BOMB_DIR = input()
    X1, Y1 = direction[BOMB_DIR]

    if previous == BOMB_DIR:
        x_reverted, x_reverted = False, False
        factor = 2

    scale_x, scale_y = int((W - X0) / factor), int((H - Y0) / factor)
    factor = factor + 2
    scale_x, scale_y = max(scale_x, 1), max(scale_y, 1)

    # print("scale", scale_x, scale_y, file=sys.stderr)
    _x, _y = should_scale(BOMB_DIR, previous)

    if (not _x or x_reverted) and not previous == BOMB_DIR:
        scale_x = 1
        x_reverted = True
    if (not _y or y_reverted) and not previous == BOMB_DIR:
        scale_y = 1
        y_reverted = True

    print("scale", scale_x, scale_y, file=sys.stderr)
    X0, Y0 = X0 + X1 * scale_x, Y0 - Y1 * scale_y

    X0, Y0 = max(X0, 0), max(Y0, 0)
    X0, Y0 = min(X0, W - 1), min(Y0, H - 1)
    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr)
    previous = BOMB_DIR

    print(X0, Y0)  # the location of the next window Batman should jump to.
