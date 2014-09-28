import sys
import math
from itertools import product, starmap
from collections import defaultdict, namedtuple
# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

# W: width of the building.
# H: height of the building.
W, H = [int(i) for i in input().split()]
N = int(input())  # maximum number of turns before game over.
X0, Y0 = [int(i) for i in input().split()]


class XY(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def scale_pos(self, scale, direction):
        self.x = self.x + (scale.x * direction.x)
        self.y = self.y + (scale.y * direction.y)

    def validate_move(self):
        '''
        Ensure points are acceptable by codingame
        '''
        x, y = self.x, self.y
        # Could be float because of scaling
        x, y = map(int, [x, y])

        x = min(x, W - 1)
        y = min(y, H - 1)

        x = max(x, 0)
        y = max(y, 0)

        self.x, self.y = x, y


directions = {
    'U': XY(0, -1),
    'UR': XY(1, -1),
    'R': XY(1, 0),
    'DR': XY(1, 1),
    'D': XY(0, 1),
    'DL': XY(-1, 1),
    'L': XY(-1, 0),
    'UL': XY(-1, -1),
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

Y = {'U', 'D'}
X = {'L', 'R'}


def similarity(bomb, previous_bomb, axis):
    if previous_bomb is None:
        return False
    else:
        # All new letters in bomb
        changed = {i for i in bomb if i not in previous_bomb}
        # Check if the new letter match the given axis
        return bool(changed.intersection(axis))


def initial_scale(axis_size, current_pos):
    '''
    Chances are if we start at the center we are closer to the bomb so
    If we start close to the edge move size / 2.7
    If we start closer to the center move size / 1.7
    '''
    difff = float(axis_size - current_pos) / float(axis_size)
    if 0.2 > difff < 0.8:
        return axis_size / 2.7
    else:
        return axis_size / 1.7


current_pos = XY(X0, Y0)
previous_pos = current_pos

scale = XY(initial_scale(W, X0),
           initial_scale(H, Y0))
swapped = XY(0, 0)
previous_bomb = None
# The factor has to change according to previous pos in case of opposite

while 1:
    # the direction of the bombs from batman's current location (U, UR, R, DR,
    # D, DL, L or UL)
    bomb_direction = input()
    move_direction = directions[bomb_direction]

    current_pos.scale_pos(scale, move_direction)
    current_pos.validate_move()

    # Save previous values
    previous_pos = XY(current_pos.x, current_pos.y)
    previous_bomb = bomb_direction

    # the location of the next window Batman should jump to.
    print(current_pos.x, current_pos.y)
