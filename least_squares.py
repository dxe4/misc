"""
/*============================================================================*/
/*                  Fit Analysis by Least Squares (FABLS)                     */
/*----------------------------------------------------------------------------*/
/*                                                                            */
/*    Program: fabls (Fit Analysis By Least Squares)                          */
/*                                                                            */
/*    Remarks: This program was inspired by a FORTRAN IV program of the       */
/*             same name for the IBM1130 written by Gordon West which was,    */
/*             in turn, inspired by a FORTRAN II program developed at         */
/*             Wright-Patterson Air Force Base for their IBM 7094.            */
/*                                                                            */
/*             The program fits linear, quadratic, exponential, logarithmic,  */
/*             and power functions to the supplied set of (x,y) values and    */
/*             prints the resulting functions and standard deviations.        */
/*                                                                            */
/*    History:                                                                */
/*                                                                            */
/*    Date     V.M   Description                                         Who  */
/*  --------  -----  -------------------------------------------------   ---  */
/*  98-12-28   1.00  Initial implementation                              MRD  */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/*                    Copyright (C) 1998 Morris R Dovey                       */
/*============================================================================*/
"""
import math


def alog(x):
    if x < 0:
        return -math.log(-x)
    elif x == 0:
        return 0
    elif x > 0:
        return math.log(x)


def pick_sign(number):
    if number > 0:
        return '-'
    else:
        return '+'


def linear(n, px, py, r):
    s1 = s2 = s3 = s4 = s = 0

    for i in range(0, n):
        x = px[i]
        y = py[i]
        s1 += x
        s2 += x * x
        s3 += y
        s4 += x * y

    denom = n * s2 - s1 * s1
    if denom == 0:
        raise ValueError("Denominator 0")

    a1 = (s3 * s2 - s1 * s1) / denom
    a2 = (n * s4 - s3 * s1) / denom

    for i in range(0, n):
        dy = py[i] - (a2 * px[i] + a1)
        s += dy * dy

    s = math.sqrt(s / r)
    sign = pick_sign(a1)

    print('Linear: y={} x {} {}; s = {}]'.format(a2, sign, abs(a1), s))

    return s


def quadratic(n, px, py, r):
    s1 = s2 = s3 = s4 = s5 = s6 = s7 = s = 0
    for i in range(0, n):
        x = px[i]
        y = py[i]
        s1 += x
        s2 += x * x
        s3 += x * x * x
        s4 += x * x * x * x
        s5 += y
        s6 += x * y
        s7 += x * x * y

    denom = n * (s2 * s4 - s3 * s3) - \
        s1 * (s1 * s4 - s3 * s3) + \
        s2 * (s1 * s3 - s3 * s2)

    if denom == 0:
        raise ValueError("Denominator 0")

    a1 = (s5 * (s2 * s4 - s3 * s3) -
          s6 * (s1 * s4 - s2 * s3) +
          s7 * (s1 * s3 - s2 * s2)) / denom
    a2 = (n * (s6 * s4 - s3 * s7) -
          s1 * (s5 * s4 - s7 * s2) +
          s2 * (s5 * s3 - s6 * s2)) / denom
    a3 = (n * (s2 * s7 - s6 * s3) -
          s1 * (s1 * s7 - s5 * s3) +
          s2 * (s1 * s6 - s5 * s2)) / denom

    for i in range(0, n):
        x = px[i]
        dy = py[i] - (a3 * x * x + a2 * x + a1)
        s += dy * dy

    s = math.sqrt(s / r)
    if a1 < 0:
        sign = '-'
    else:
        sign = '+'

    sign = pick_sign(a1)
    sign2 = pick_sign(a2)
    print("quadratic: y = {} x^2 {} {} x {} {}; s={}".format(
        a3, sign2, abs(a2), sign, abs(a1), s))

    return s


def exponential(n, px, py, r):
    s1 = s2 = s3 = s4 = s = 0
    for i in range(0, n):
        x = px[i]
        y = alog(py[i])
        s1 += x
        s2 += x * x
        s3 += y
        s4 += x * y

    denom = n * s2 - s1 * s1
    if denom == 0:
        raise ValueError("Denominator 0")

    a1 = (s3 * s2 - s1 * s4) / denom
    a2 = (n * s4 - s3 * s1) / denom
    for i in range(0, n):
        dy = alog(py[i]) - (a2 * px[i] + a1)
        s += dy * dy

    s = math.sqrt(s / r)
    sign = pick_sign(a1)
    print("Exponential: y = exp({} x {} {}); s = {}".format(
        a2, sign, abs(a1), s))

    return s


def logarithmic(n, px, py, r):
    s1 = s2 = s3 = s4 = s = 0
    for i in range(0, n):
        x = alog(px[i])
        y = py[i]
        s1 += x
        s2 += x * x
        s3 += y
        s4 += x * y

    denom = n * s2 - s1 * s1
    if denom == 0:
        raise ValueError("Demoninator 0")

    a1 = (s3 * s2 - s1 * s4) / denom
    a2 = (n * s4 - s3 * s1) / denom

    for i in range(0, n):
        x = alog(px[i])
        dy = py[i] - (x * a2 + a1)
        s += dy * dy

    s = math.sqrt(s / r)
    sign = pick_sign(a1)
    print("Logarithmic: y = ({}) ln(x) {} {}; s = {}".format(
        a2, sign, abs(a1), s))

    return s


def power(n, px, py, r):
    s1 = s2 = s3 = s4 = s = 0
    for i in range(0, n):
        if px[i] == 0:
            raise ValueError("Not power")

        x = alog(px[i])
        y = alog(py[i])

        s1 += x
        s2 += x * x
        s3 += y
        s4 += x * y

    denom = n * s2 - s1 * s1
    if denom == 0:
        raise ValueError("Demoninator 0")

    a1 = math.exp((s3 * s2 - s1 * s4) / denom)
    a2 = (n * s4 - s3 * s1) / denom
    for i in range(0, n):
        dy = py[i] - a1 * math.pow(px[i], a2)
        s += dy * dy

    s = math.sqrt(s / r)

    print("Power: y = ({}) x ^ ({}); s = {}".format(a1, a2, s))
    return s


def calulate(n, px, py):
    s_list = []
    r = n - 1
    functions = [linear, quadratic, exponential, logarithmic, power]
    for func in functions:
        try:
            s = func(n, px, py, r)
            s_list.append(s)
        except ValueError:
            # Demoninator 0, ignore
            s_list.append(None)

    print(s_list)
    least = s_list.index(min((i for i in s_list if i is not None)))
    best_func = functions[least]
    print("Best function {}".format(best_func.__name__))


n = 10
xp = []
yp = []
c1 = math.exp(1)
c2 = math.pi
c3 = math.sqrt(2)


for i in range(0, n):
    z = i + 1
    xp.append(z)
    # yp.append(c1 * z + c2)  # Linear
    # yp.append(c1 * z * z + c2 * z + c3)  # Quadratic
    # yp.append(math.exp(c1 * z + c2))  # Exponential
    # yp.append(z ? c1 * math.log(z) + c2 : c2)  # Logarithmic
    yp.append(c1 * math.pow(z, c2))  # Power

calulate(n, xp, yp)
