"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and 
the leftmost intersection points of the two functions. 

The functions for the numeric answers are specified in MOODLE. 


This assignment is more complicated than Assignment1 and Assignment2 because: 
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors. 
    2. You have the freedom to choose how to calculate the area between the two functions. 
    3. The functions may intersect multiple times. Here is an example: 
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately. 
       You should explain why in one of the theoretical questions in MOODLE. 

"""

import numpy as np
import time
import random
from assignment2 import Assignment2


class Assignment3:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the integration error. 
        Your secondary objective is minimizing the running time. The assignment
        will be tested on variety of different functions. 
        
        Integration error will be measured compared to the actual value of the 
        definite integral. 
        
        Note: It is forbidden to call f more than n times. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
        """

        def trapezoid(f, a, b, n):
            """
            Helper function that calculates the definite integral of a function
            using the trapezoidal rule.

            """
            # create an array of n equally spaced points between a and b
            X = np.linspace(a, b, n)
            n -= 1
            # calculate the width of each trapezoid
            h = (b - a) / n
            # evaluate f at each point in X
            y = [f(x)for x in X]
            # add up the areas of each trapezoid and return the result
            y_point =y[1:] + y[:-1]
            result = np.float32((h/2)*np.sum(y_point))
            return result

        # Using trapezoid rule for small n
        if n < 3: return trapezoid(f, a, b, n)

        # Using simpson's rule for n >= 3
        if n % 2 == 0: n -= 1

        X = np.linspace(a, b, n)
        # calculate the value of the integral at the endpoints
        integration = f(a) + f(b)
        # calculate the width of each interval
        h = (b - a) / (n-1)
        #Simpson's rule
        for i in range(1,n-1): #check
            if i % 2 == 0:
                integration += 2 * f(X[i])
            else:
                integration += 4 * f(X[i])
        result = integration * (h/3)
        return np.float32(result)


    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Finds the area enclosed between two functions. This method finds
        all intersection points between the two functions to work correctly.

        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        Note, there is no such thing as negative area.

        In order to find the enclosed area the given functions must intersect
        in at least two points. If the functions do not intersect or intersect
        in less than two points this function returns NaN.
        This function may not work correctly if there is infinite number of
        intersection points.


        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis
        """
        # Define the range of x-values to search for intersection points
        start = 1
        end = 150
        find_integral = Assignment2()

        # Find all intersection points between f1 and f2 within the given range
        intersections = find_integral.intersections(f1, f2, start, end)

        # If there are less than 2 intersection points, return NaN
        if len(intersections) < 2:
            return np.float32('nan')

        # If there are exactly 2 intersection points, find the area between the curves
        elif len(intersections) == 2:
            a, b = intersections[0], intersections[1]
            if a > b:
                a, b = b, a
            integrand = lambda x: abs(f1(x) - f2(x))
            return self.integrate(integrand, a, b, 1000)
        # If there are more than 2 intersection points, find the areas between the curves
        else:
            # Define the upper and lower functions for each interval between adjacent intersection points
            functions = []
            for i in range(len(intersections)):
                x = intersections[i]
                if f1(x) > f2(x):
                    functions.append((f1, f2))
                else:
                    functions.append((f2, f1))

            # Calculate the area between the upper and lower functions
            areas = []
            for i in range(len(intersections) - 1):
                a, b = intersections[i], intersections[i + 1]
                if a > b:
                    a, b = b, a

                # Define the integrand as the difference between the upper and lower functions
                integrand = lambda x: functions[i][0](x) - functions[i][1](x)
                areas.append(abs(self.integrate(integrand, a, b, 1000)))
            return np.float32(sum(areas))

##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment3(unittest.TestCase):

    def test_integrate_float32(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])

        r = ass3.integrate(f1, -1, 1, 10)


        self.assertEqual(r.dtype, np.float32)

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = strong_oscilations()
        r = ass3.integrate(f1, 0.09, 10, 20)
        print(r)
        true_result = -7.78662 * 10 ** 33
        print(abs((r - true_result) / true_result))
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))


if __name__ == "__main__":
    unittest.main()
