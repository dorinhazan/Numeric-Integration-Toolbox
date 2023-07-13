"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random
from sampleFunctions import *
import torch
import copy

class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        starting to interpolate arbitrary functions.
        """

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolates a given function f using cubic Bézier curves.
        Parameters
        ----------
        f: callable
           The function to interpolate.
        a: float
           The left endpoint of the interval to interpolate over.
        b: float
           The right endpoint of the interval to interpolate over.
        n: int
           The number of points to use in the interpolation.
        return: callable, A callable object that takes a float x in the range [a, b]and returns the interpolated value of f at x.
        """

        def ThomasAlgorithm(a, b, c, d):
            """
            Thomas algorithm implementation to solve the tridiagonal matrix equation.
            Parameters
            ----------
            a: numpy array
               The subdiagonal of the tridiagonal matrix.
            b: numpy array
               The diagonal of the tridiagonal matrix.
            c: numpy array
               The superdiagonal of the tridiagonal matrix.
            d: numpy array
               The right-hand side of the equation.
            return: (numpy array) The solution to the tridiagonal matrix equation.
            """
            n = len(d)
            c_ = np.zeros(n - 1)
            d_ = np.zeros(n)

            c_[0] = c[0] / b[0]
            d_[0] = d[0] / b[0]

            for i in range(1, n - 1):
                c_[i] = c[i] / (b[i] - a[i] * c_[i - 1])
                d_[i] = (d[i] - a[i - 1] * d_[i - 1]) / (b[i] - a[i - 1] * c_[i - 1])

            x = np.zeros(n)
            x[-1] = d_[-1]
            for i in range(n - 2, -1, -1):
                x[i] = d_[i] - c_[i] * x[i + 1]

            return x

        def bezier_cubic_coefficients(ps):
            """
            Computes the coefficients for a cubic Bézier according to given point ps
            Parameters
            ----------
            ps: numpy array
                The set of points to interpolate over.
            return: (numpy array) The coefficients for a cubic Bézier curve that passes through
                the given set of points.
            """
            n = len(ps) - 1

            # Buidling coefficence matrix.
            Coef_mat = 4 * np.identity(n)
            np.fill_diagonal(Coef_mat[1:], 1)
            np.fill_diagonal(Coef_mat[:, 1:], 1)
            Coef_mat[0, 0] = 2
            Coef_mat[n - 1, n - 1] = 7
            Coef_mat[n - 1, n - 2] = 2
            # build points vector
            P = [2 * (2 * ps[i] + ps[i + 1]) for i in range(n)]
            P[0] = ps[0] + 2 * ps[1]
            P[n - 1] = 8 * ps[n - 1] + ps[n]

            c = [Coef_mat[i][i + 1] for i in range(n - 1)]
            b = [Coef_mat[i][i] for i in range(n)]
            a = [Coef_mat[i + 1][i] for i in range(n - 1)]
            b.append(Coef_mat[n - 1][n - 1])

            A = ThomasAlgorithm(a, b, c, P)
            B = [0] * n
            for i in range(n - 1):
                B[i] = 2 * ps[i + 1] - A[i + 1]
            B[n - 1] = (A[n - 1] + ps[n]) / 2

            return A, B

        def bezier_cubic(points):
            """
            Calculates the cubic Bezier curve coefficients given a list of control points.
            Parameters
            ----------
            points: list
                    A list of control points in the form [(x0, y0), (x1, y1), ..., (xn, yn)].

            Returns: (tuple) A tuple of two lists containing the coefficients for the cubic Bezier curve.
            """
            A, B = bezier_cubic_coefficients(points)
            return [[points[i], A[i], B[i], points[i + 1]] for i in range(len(points) - 1)]

        if (n == 1):
            return lambda x: f((a + b) / 2)
        else:
            x_points = np.linspace(a, b, n)
            y_points = np.array([f(x) for x in x_points])
            x_values = bezier_cubic(x_points)
            y_values = bezier_cubic(y_points)
            bezier_curves = []
            for i in range(len(x_values)):
                p0 = [x_values[i][0], y_values[i][0]]
                p1 = [x_values[i][1], y_values[i][1]]
                p2 = [x_values[i][2], y_values[i][2]]
                p3 = [x_values[i][3], y_values[i][3]]
                bezier_curves.append([p0, p1, p2, p3])


        def bisection(f, a, b, tol):
            """
            Finds the root of a function using the bisection method.
            Parameters
            ----------
            f: function
               The function to find the root of.
            a: float
               The lower bound of the search interval.
            b: float
               The upper bound of the search interval.
            tol: float
               The tolerance for the root.

            Returns: (float) The root of the function.
            """
            fa = f(a)
            fb = f(b)
            while (b - a) / 2 > tol:
                c = (a + b) / 2
                fc = f(c)
                if fc == 0:
                    return c
                elif fa * fc < 0:
                    b = c
                    fb = fc
                else:
                    a = c
                    fa = fc
            return (a + b) / 2

        def g_new(P0, P1, P2, P3, x):
            """
            Calculates the y value of a point on a cubic Bezier curve given the x value.
            Parameters
            ----------
            P0: list
                The first control point of the curve.
            P1: list
                The second control point of the curve.
            P2: list
                The third control point of the curve.
            P3: list
                The fourth control point of the curve.
            x: float
               The x value of the point to calculate the y value for.

            Returns: (float) The y value of the point on the curve.
            """
            coefficients = [-P0[0] + 3 * P1[0] - 3 * P2[0] + P3[0], 3 * P0[0] - 6 * P1[0] + 3 * P2[0],-3 * P0[0] + 3 * P1[0], P0[0] - x]
            f2 = lambda x: np.power(coefficients[0] * x , 3)  + np.power(coefficients[1] * x , 2) + coefficients[2] * x + coefficients[3]
            # root = bisection(f2, 0, 1, 0.0001)
            i = 0
            up = 1
            down = 0
            root = bisection(f2, down, up, 0.0001)
            while root > 1 or root < 0:
                if i == 20:
                    break
                if root > 1:  # If the root is greater than 1, update the upper bound
                    up = root
                elif root < 0:  # If the root is less than 0, update the lower bound
                    down = root
                root = bisection(f2, down, up, 0.0001)
                i += 1
            ## the Bezier curve formula
            return (1 - root) ** 3 * P0[1] + 3 * (1 - root) ** 2 * root * P1[1] + 3 * (1 - root) * root ** 2 * P2[1] + root ** 3 * P3[1]

        def g(x):
            # Set the initial values of the first and last index of the x_points list

            first = 0
            last = len(x_points) - 1

            # Loop until the first index is less than or equal to the last index
            while first <= last:
                # Calculate the median index between the first and last index
                median = (first + last) // 2
                # Check if the input x is between the median and the next point on x_points
                if x_points[median] <= x and x <= x_points[median + 1]:
                    # If x is between median and next point, use g_new function to get y value
                    return g_new(bezier_curves[median][0], bezier_curves[median][1], bezier_curves[median][2],bezier_curves[median][3], x)
                else:
                    # If x is not between median and next point, update the first or last index value accordingly
                    if x < x_points[median + 1]:
                        last = median - 1
                    else:
                        first = median + 1

        # Return g function
        return g



##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)
            ff = ass1.interpolate(f, -10, 10, 1)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)


    def test_with_poly_restrict(self):

        ass1 = Assignment1()
        a = np.random.randn(5)
        mean_err = 0
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)

if __name__ == "__main__":
    unittest.main()
