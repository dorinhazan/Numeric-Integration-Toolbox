"""
In this assignment you should fit a model function of your choice to data
that you sample from a contour of given shape. Then you should calculate
the area of that shape.

The sampled data is very noisy so you should minimize the mean least squares
between the model you fit and the data points you sample.

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You
must make sure that the fitting function returns at most 5 seconds after the
allowed running time elapses. If you know that your iterations may take more
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment.
Note: !!!Despite previous note, using reflection to check for the parameters
of the sampled function is considered cheating!!! You are only allowed to
get (x,y) points from the given shape by calling sample().
"""
import math
from turtle import Shape

import numpy as np
import time
import random
from functionUtils import AbstractShape
from scipy.interpolate import splprep, splev
from sklearn.cluster import KMeans




class MyShape(AbstractShape):
    def __init__(self, area):
        """
        Constructor for the MyShape class.

        Parameters
        ----------
        area : np.float32
            The area of the shape.
        """
        self._area = area

    def area(self) -> np.float32:
        """
                Get the area of the shape.

        Returns
        -------
        np.float32
            The area of the shape.
        """
        return self._area




class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """
        pass


    def area(self, contour: callable, maxerr=0.001)->np.float32:
        """
        Compute the area of the shape with the given contour.

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """
        # Generate a set of contour points
        n = 4000
        contour = contour(n)
        x = [p[0] for p in contour]
        y = [p[1] for p in contour]

        # Use the shoelace formula to calculate the area of the polygon
        return np.float32(0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(n - 1))))

    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape.

        Parameters
        ----------
        sample : callable.
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        An object extending AbstractShape.
        """

        # initialize the list of data points and the stop time
        data_points = [] #the point's (x,y) from the samples

        # Insert points to the list until the maxtime/2 is reached
        for i in range(int(maxtime/2)):
            x, y = sample()
            x = float(x)
            y = float(y)
            data_points.append([x, y])

        # use k-means clustering to group the data points into clusters
        num_data_point = int(len(data_points))
        if num_data_point < 32: #if there isn't a lot of points
            kmeans = KMeans(n_clusters = num_data_point,n_init=1) #create a clusters
        else: kmeans = KMeans(n_clusters = 32,n_init=1) #create a clusters
        kmeans.fit(data_points)
        clean_points = []
        for i in kmeans.cluster_centers_: #append the point to a new list after the fit
            clean_points.append([i[0], i[1]])
        clean_points.sort() #sort the new list acoording to x and than y

        # Set the centroid to the leftmost point in the sorted list
        centroid = clean_points[0]
        sort_list_cen = [centroid] #list with only one point is sorted


        # Function to calculate the Euclidean distance between two points
        def euclidean_distance(p1 , p2) -> float:
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        while len(clean_points) != 2:
            # calculate the distances of each point to the current centroid
            centers = []
            for i in range (len(clean_points)):
                distance = euclidean_distance(clean_points[i],centroid)
                centers.append([distance,[clean_points[i][0],clean_points[i][1]]])
            centers.sort()
            list_points =[]
            for auc_res in centers :
                list_points.append(auc_res[1])

            # add the closest point to the list of sorted centers
            new_center = list_points[1]
            sort_list_cen.append(new_center)

            # remove the previous centroid from the list of clean points and update the centroid
            clean_points.remove(centroid)
            centroid = new_center

        # add the final two points to the list of sorted centers and close the shape
        sort_list_cen.append(clean_points[1])
        sort_list_cen.append(sort_list_cen[0])
        center = sort_list_cen

        # calculate the area of the shape using the Shoelace Formula
        a = 0
        for i in range(len(center) - 1):
            a += 0.5 * (center[i][0] * center[i + 1][1] - center[i + 1][0] * center[i][1])

        # return a MyShape object with the absolute value of the calculated area
        return MyShape(abs(np.float32(a)))




##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=10)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    # def test_delay(self):
    #     circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        def sample():
            time.sleep(7)
            return circ()

        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)


if __name__ == "__main__":
    unittest.main()
