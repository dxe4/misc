import sys
import math
import math

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

N = int(input())  # the number of points used to draw the surface of Mars.

landing_area = None
previous_y = None
previous_x = None
for i in range(N):
    # LAND_X: X coordinate of a surface point. (0 to 6999)
    # LAND_Y: Y coordinate of a surface point. By linking all the points
    # together in a sequential fashion, you form the surface of Mars.
    LAND_X, LAND_Y = [int(i) for i in input().split()]

    if LAND_Y == previous_y:
        landing_area = [(previous_x, previous_y), (LAND_X, LAND_Y)]

    previous_y = LAND_Y
    previous_x = LAND_X

max_v_speed = -38
gravity = 3.711
mass = 1


def in_boundaries(x, y):
    return landing_area[0][0] < x < landing_area[1][0]


def calculate_vertical_power(VS, P):
    '''
    Used in case we are allready at the right angle
    '''
    if VS < max_v_speed:
        P = min(P + 1, 4)
    else:
        P = 0
    return P


def calculate_force(accel, angle):
    if angle == 0:
        x = accel * mass
        y = (accel + gravity) * mass
        return x, y
    else:
        x = math.sin(math.radians(angle)) * accel
        y = math.cos(math.radians(angle)) * accel
        x = max(x, 0.001)
        y = max(y, 0.001)
        return x, y


def angle_to_area(X0, Y0, X1, Y1):
    '''
    Returns an angle to reach the point direct
    direct is not always the case might need improvements
    '''
    dX = X0 - X1
    dY = Y0 - Y1
    angle = math.atan2(dY / dX) * 180 / pi
    return int(angle)


def position_ship(HS, VS, P, X, Y, R, force_x, force_y):
    angle = angle_to_area(X, Y, landing_area[0][0], landing_area[0][1])
    return 15, 2

# game loop
while 1:
    # HS: the horizontal speed (in m/s), can be negative.
    # VS: the vertical speed (in m/s), can be negative.
    # F: the quantity of remaining fuel in liters.
    # R: the rotation angle in degrees (-90 to 90).
    # P: the thrust power (0 to 4).
    X, Y, HS, VS, F, R, P = [int(i) for i in input().split()]

    distance_left = Y - landing_area[0][0]
    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr)
    if in_boundaries(X, Y):
        P = calculate_vertical_power(VS, P)
        # R P. R is the desired rotation angle. P is the desired thrust power.
        print("0 {}".format(P))
    else:
        force_x, force_y = calculate_force(P, R)
        force_y = force_y - gravity
        print(force_x, force_y, file=sys.stderr)

        P, R = position_ship(HS, VS, P, X, Y, R, force_x, force_y)
        print(P, R)
