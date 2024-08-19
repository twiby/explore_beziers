import sys
import numpy as np

sys.path.append("..")
from point import Point
from constants import *

from .printable_arc import PrintableArc

class Arc(PrintableArc):
	def __init__(self, p0, p1, p2, *args, **kwargs):
		if not type(p0) is Point:
			raise ValueError
		if not type(p1) is Point:
			raise ValueError
		if not type(p2) is Point:
			raise ValueError

		self.points = [p0, p1, p2]
		self.degenerate = False
		self.init_center()
		PrintableArc.__init__(self, *args, **kwargs)

	def set_points(self, p0, p1, p2):
		self.points = [p0, p1, p2]
		self.init_center()

	def chord(self):
		return (self.points[2] - self.points[0]).norm()

	def init_center(self):
		'''https://web.archive.org/web/20161011113446/http://www.abecedarical.com/zenosamples/zs_circle3pts.html'''
		if colinears(self.points[0], self.points[1], self.points[2]):
			self.degenerate = True
			self.center = None
			self.radius = None
			return

		m_11 = determinant_3x3_last_column_ones(np.array([
			[self.points[0].x, self.points[0].y],
			[self.points[1].x, self.points[1].y],
			[self.points[2].x, self.points[2].y],
		]))

		m_12 = determinant_3x3_last_column_ones(np.array([
			[self.points[0].x**2 + self.points[0].y**2, self.points[0].y],
			[self.points[1].x**2 + self.points[1].y**2, self.points[1].y],
			[self.points[2].x**2 + self.points[2].y**2, self.points[2].y],
		]))

		m_13 = determinant_3x3_last_column_ones(np.array([
			[self.points[0].x**2 + self.points[0].y**2, self.points[0].x],
			[self.points[1].x**2 + self.points[1].y**2, self.points[1].x],
			[self.points[2].x**2 + self.points[2].y**2, self.points[2].x],
		]))

		m_14 = determinant_3x3(np.array([
			[self.points[0].x**2 + self.points[0].y**2, self.points[0].x, self.points[0].y],
			[self.points[1].x**2 + self.points[1].y**2, self.points[1].x, self.points[1].y],
			[self.points[2].x**2 + self.points[2].y**2, self.points[2].x, self.points[2].y],
		]))
		
		try:
			x = 0.5 * m_12 / m_11
			y = -0.5 * m_13 / m_11
			self.center = Point(x, y)
			self.radius = np.sqrt(x**2 + y**2 + m_14/m_11)
		except:
			self.degenerate = True
			self.center = None
			self.radius = None

	def angle_parameters(self):
		vector_start = self.points[0] - self.center
		vector_mid = self.points[1] - self.center
		vector_end = self.points[2] - self.center
		theta_start = vector_start.angle()
		theta_mid = vector_mid.angle()
		theta_end = vector_end.angle()
		while theta_end < theta_start:
			theta_end += 2*np.pi
		if not (theta_mid >= theta_start and theta_mid <= theta_end):
			theta_start, theta_end = theta_end, theta_start + 2*np.pi
		assert(theta_end > theta_start)
		assert(theta_end < theta_start + 2*np.pi)
		return theta_start, theta_end

	def point_on_arc(self, p, eps):
		if self.degenerate:
			return point_on_line(self.points[0], p, self.points[2], eps)

		vec = p - self.center
		radius_diff = abs(vec.norm() - self.radius)
		if radius_diff > eps:
			return False

		theta_start, theta_end = self.angle_parameters()
		angle = vec.angle()
		while angle > theta_end:
			angle -= 2*np.pi
		while angle < theta_start:
			angle += 2*np.pi
		return angle >= theta_start and angle <= theta_end

	def __repr__(self):
		return "Arc(" \
		+ str(self.points[0]) + ", "\
		+ str(self.points[1]) + ", "\
		+ str(self.points[2]) + ", "\
		+ ")"

def area(p1, p2, p3):
	'''shoelace formula'''
	return abs(0.5 * (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)))

def colinears(p1, p2, p3):
	return point_on_line(p1, p2, p3, MIN_EPS)

def point_on_line(p1, p, p2, eps):
	height = 2 * area(p1, p2, p) / (p2 - p1).norm()
	return height < eps

def determinant_3x3(x):
	if x.shape != (3, 3):
		raise ValueError

	return x[0, 0] * (x[1, 1] * x[2, 2] - x[1, 2] * x[2, 1]) -\
		x[0, 1] * (x[1, 0] * x[2, 2] - x[1, 2] * x[2, 0]) +\
		x[0, 2] * (x[1, 0] * x[2, 1] - x[1, 1] * x[2, 0])

def determinant_3x3_last_column_ones(x):
	if x.shape != (3, 2):
		raise ValueError

	return x[0, 0] * (x[1, 1] - x[2, 1]) -\
		x[0, 1] * (x[1, 0] -  x[2, 0]) +\
		(x[1, 0] * x[2, 1] - x[1, 1] * x[2, 0])