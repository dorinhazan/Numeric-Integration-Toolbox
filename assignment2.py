"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.

        This function may not work correctly if there is infinite number of
        intersection points.


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.
        """



        # Define a function for the Newton-Raphson method to find the root of a function
        def newton_raphson_method(f, df, x0, maxerr, max_iter):
            try:
                # Try for the maximum number of iterations allowed to find the root of the function
                for i in range(max_iter):
                    fx = f(x0)
                    if fx == 0:
                        return x0
                    dfx = df(x0)
                    if dfx == 0:
                        return None
                    x1 = x0 - fx / dfx
                    # If the difference between the current and previous estimate of the root is within the maximum error, return the estimate
                    if abs(x1 - x0) <= maxerr:
                        return x1
                    x0 = x1
                    # If the maximum number of iterations is reached, return the most recent estimate of the root
                    if i == max_iter - 1:
                        return x1
                # return x1

            except:
                return
        # Define a function to numerically differentiate a function f at a point x
        def differentiate(f, x, h=1e-8):
            return (f(x + h) - f(x)) / h
        # Define the function that is the difference between f1 and f2, and its derivative
        f = lambda x: f1(x) - f2(x)
        df = lambda x: differentiate(f, x)  # differentiate is a function that calculates the numerical derivative of f

        result = []

        points = np.linspace(a, b, 1000)
        for i in range(len(points) - 1):
            # Check if there is a sign change in f between adjacent points
            if f(points[i]) * f(points[i + 1]) <= 0:
                # If there is, use the Newton-Raphson method to estimate the root of f between the adjacent points
                root = newton_raphson_method(f, df, points[i], maxerr, 200)
                if root:
                    result.append(root)

        return result

##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))


if __name__ == "__main__":
    unittest.main()
