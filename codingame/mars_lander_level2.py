import sys
import math
from collections import namedtuple
from operator import add, sub

Area = namedtuple('Area', ['x0', 'y0', 'x1', 'y1'])
Point = namedtuple('Point', ['x', 'y'])
Limit = namedtuple('Limit', ['min', 'max'])


def read_initial_input():
    landing_area = None
    previous_y = None
    previous_x = None

    N = int(input())  # the number of points used to draw the surface of Mars.

    for i in range(N):
        # LAND_X: X coordinate of a surface point. (0 to 6999)
        # LAND_Y: Y coordinate of a surface point. By linking all the points
        # together in a sequential fashion, you form the surface of Mars.
        LAND_X, LAND_Y = [int(i) for i in input().split()]

        if LAND_Y == previous_y:
            landing_area = Area(previous_x, previous_y, LAND_X, LAND_Y)

        previous_y, previous_x = LAND_Y, LAND_X

    return landing_area


point_to_reach = None
landing_area = read_initial_input()

v_speed_limit = Limit(-40, 40)
h_speed_limit = Limit(-20, 20)

gravity = -3.711
mass = 1

landing_area_size = Point(
    abs(landing_area.x1 - landing_area.x0),
    abs(landing_area.y1 - landing_area.y0),
)
landing_area_center = Point(
    landing_area.x1 - landing_area_size.x / 2,
    landing_area.y1 - landing_area_size.y / 2,
)


def in_boundaries(x, y):
    return landing_area.x0 < x < landing_area.x1


def ready_for_landing(x, y, HS):
    return in_boundaries(x, y) and HS == 0


def reached_speed_limit(S, speed_limit):
    return not(speed_limit.min < S < speed_limit.max)


def calculate_vertical_power(S, P):
    '''
    Used in case we are allready at the right angle
    '''
    if reached_speed_limit(S, v_speed_limit):
        # Landing speed is ~40 because of gravity
        return min(P + 1, 4)
    else:
        return 0


def calculate_force(accel, angle):
    if angle == 0:
        x = accel * mass
        y = (accel + gravity) * mass
        return Point(x, y)
    else:
        x = math.sin(math.radians(angle)) * accel
        y = math.cos(math.radians(angle)) * accel
        x = max(x, 0.001)
        y = max(y, 0.001)
        return Point(x, y)


def _change_power(P, operator):
    P = operator(P, 1)
    P = max(P, 3)
    P = min(P, 0)
    return P


def increase_power(P):
    return _change_power(P, add)


def decrease_power(P):
    return _change_power(P, sub)


def move(X, Y, HS, VS, P, angle):
    force = calculate_force(P, angle)

    HS = HS + force.y
    VS = VS + force.x

    X = X + HS
    Y = Y + VS

    angle = angle_to_area(X, Y, point_to_reach.x, point_to_reach.y)

    return HS, VS, X, Y, angle


def reduce_speed_time(X, Y, HS, VS, P, angle):
    '''
    Calculate how long does it take to slow down
    e.g. if HS=17 P=3 angle=90 we will reach the limit
    because we can only change power 1 at a time at 15 for angle
    Therefore next min y_forces are 2.61 and 1.04
    '''
    # Execute current move.
    # Current move needs to be outside the loop
    # Because in the loop the angle starts to move opposite
    HS, VS, X, Y, angle = move(X, Y, HS, VS, P, angle)
    # If current move has reached the limit return 0
    # so we can re-caclulate the move and reduce speed
    if any([reached_speed_limit(HS, h_speed_limit),
            reached_speed_limit(VS, v_speed_limit)]):
        return 0

    for i in range(1, 10):
        HS, VS, X, Y = move(X, Y, HS, VS, P, angle)
        angle = angle_to_area(X, Y, point_to_reach.x, point_to_reach.y)

        if any([reached_speed_limit(HS, h_speed_limit),
                reached_speed_limit(VS, v_speed_limit)]):
            return i
    else:
        # If at 10 we are within limit's no need to worry
        return i


def angle_to_area(X0, Y0, X1, Y1):
    '''
    Returns an angle to reach the point direct
    direct is not always the case might need improvements
    '''
    dX = X1 - X0
    dY = Y1 - Y0

    radians = math.atan2(dX, dY)
    angle = radians * 180 / math.pi
    return int(-angle)


def position_ship(HS, VS, P, X, Y, R, force):
    angle = angle_to_area(X, Y, point_to_reach.x, point_to_reach.y)

    if any([reached_speed_limit(HS, h_speed_limit),
            reached_speed_limit(VS, v_speed_limit)]):
        # Power 0 won't help to recover vspeed
        # Need to use the force to calc how to reduce it
        return angle, 0
    else:
        return angle, 3


# game loop
while 1:
    # HS: the horizontal speed (in m/s), can be negative.
    # VS: the vertical speed (in m/s), can be negative.
    # F: the quantity of remaining fuel in liters.
    # R: the rotation angle in degrees (-90 to 90).
    # P: the thrust power (0 to 4).
    X, Y, HS, VS, F, R, P = [int(i) for i in input().split()]

    if not point_to_reach:
        point_to_reach = Point(landing_area_center.x, Y)

    distance_left = Y - landing_area.x0
    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr)
    if ready_for_landing(X, Y, HS):
        P = calculate_vertical_power(VS, P)
        # R P. R is the desired rotation angle. P is the desired thrust power.
        print("0 {}".format(P))
    else:
        force = calculate_force(P, R)
        force = Point(force.x, force.y + gravity)

        P, R = position_ship(HS, VS, P, X, Y, R, force)
        print(P, R)
