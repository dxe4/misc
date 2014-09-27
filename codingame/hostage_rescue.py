import sys
import math
from itertools import product, starmap
from collections import defaultdict
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


def _default_tuple():
    return (False, False)

opposite = defaultdict(_default_tuple)
opposite.update({
    'UD': (False, True),
    'RL': (True, False),
    'DU': (False, True),
    'LR': (True, False),
})


def is_opposite(BOMB_DIR, previous_bomb):
    if previous_bomb is None:
        return (False, False)
    else:
        result = []
        for pair in product(BOMB_DIR, previous_bomb):
            # convert char tuple to string
            key = ''.join(pair)
            # If key does not exist returns (False, False)
            result.append(opposite[key])

        if not result:
            return (False, False)

        # This is probably an itertools job
        x_opposite = bool([i[0] for i in result if i[0] is True])
        y_opposite = bool([i[1] for i in result if i[1] is True])

        return x_opposite, y_opposite


def should_scale(BOMB_DIR, previous_bomb):
    x_opposite, y_opposite = is_opposite(BOMB_DIR, previous_bomb)
    return not x_opposite, not y_opposite


# game loop
previous_bomb = None
prevopis_pos_x = X0
prevopis_pos_y = Y0
x_reverted = False
y_reverted = False
direction_swap = 0
same_bomb_times = 0

# The factor has to change according to previous pos in case of opposite
factor = 2

while 1:
    # the direction of the bombs from batman's current location (U, UR, R, DR,
    # D, DL, L or UL)
    BOMB_DIR = input()
    X1, Y1 = direction[BOMB_DIR]

    distance_x, distance_y = prevopis_pos_x - X0, prevopis_pos_y - Y0

    opposite_x, opposite_y = is_opposite(BOMB_DIR, previous_bomb)

    scale_x = (W - X0) / factor
    scale_y = (H - Y0) / factor
    scale_x, scale_y = starmap(max, [(scale_x, 1), (scale_y, 1)])

    factor = factor + 1
    if any([opposite_x, opposite_y]):
        direction_swap = direction_swap + 1
        scale_x = scale_x / 4
        scale_y = scale_y / 4
    elif previous_bomb == BOMB_DIR:
        factor = 1
        same_bomb_times = same_bomb_times + 1
        scale_x = ((W - X0) / factor) * same_bomb_times
        scale_y = ((H - Y0) / factor) * same_bomb_times
    if direction_swap >= 2:
        scale_x = 1
        scale_y = 1

    print("scale", scale_x, scale_y, file=sys.stderr)
    X0 = X0 + X1 * scale_x
    Y0 = Y0 - Y1 * scale_y

    X0, Y0 = starmap(max, [(X0, 0), (Y0, 0)])
    X0, Y0 = starmap(min, [(X0, W - 1), (Y0, H - 1)])
    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr)
    previous_bomb = BOMB_DIR

    X0, Y0 = int(X0), int(Y0)
    print(X0, Y0)  # the location of the next window Batman should jump to.
    prevopis_pos_x, prevopis_pos_y = X0, Y0
