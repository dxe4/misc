import sys
import math
from collections import namedtuple


width = 7000
height = 3000

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

        previous_y = LAND_Y
        previous_x = LAND_X

    return landing_area


def pick_point_to_reach():
    '''
    This picks a point above the landing area with no intelligence
    It may need to be improved later on
    (eg if theres obstacles, or not enough fuel)
    '''
    return Point(landing_area_center.x,
                 landing_area_center.y + height / 3)

landing_area = read_initial_input()

speed_limit = Limit(-38, 38)
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

point_to_reach = pick_point_to_reach()


def in_boundaries(x, y):
    return landing_area.x0 < x < landing_area.x1


def ready_for_landing(x, y, HS):
    return in_boundaries(x, y) and HS == 0


def reached_speed_limit(*args):
    '''
    args: HS, VS
    '''
    return bool([i for i in args
                 if i <= speed_limit.min or i >= speed_limit.max])


def calculate_vertical_power(S, P):
    '''
    Used in case we are allready at the right angle
    '''
    if reached_speed_limit(S):
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


def angle_to_area(X0, Y0, X1, Y1):
    '''
    Returns an angle to reach the point direct
    direct is not always the case might need improvements
    '''
    dX = X1 - X0
    dY = Y1 - Y0
    angle = math.atan2(dY, dX) * 180 / math.pi
    return int(angle)


def position_ship(HS, VS, P, X, Y, R, force):
    angle = angle_to_area(X, Y, point_to_reach.x, point_to_reach.y)

    if reached_speed_limit(HS, VS):
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

        print(force.x, force.y, file=sys.stderr)

        P, R = position_ship(HS, VS, P, X, Y, R, force)
        print(P, R)
