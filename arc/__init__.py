import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.append("..")
from point import Point
from utils import set_current_subplot
from constants import *

class Arc:
	def __init__(self, p0, p1, p2, f = None, ax = None):
		if not type(p0) is Point:
			raise ValueError
		if not type(p1) is Point:
			raise ValueError
		if not type(p2) is Point:
			raise ValueError

		self.points = [p0, p1, p2]

		self.f = f
		self.ax = ax
		self.printed_points = None
		self.printed_line = None
		self.degenerate = False

		self.init_center()

	def set_points(self, p0, p1, p2):
		self.points = [p0, p1, p2]
		self.init_center()

	def init_center(self):
		'''https://web.archive.org/web/20161011113446/http://www.abecedarical.com/zenosamples/zs_circle3pts.html'''
		from numpy.linalg import det, norm

		if colinears(self.points[0], self.points[1], self.points[2], EPS_ARC):
			self.degenerate = True
			self.center = None
			self.radius = None
			return

		m_11 = np.array([
			[self.points[0].x, self.points[0].y, 1],
			[self.points[1].x, self.points[1].y, 1],
			[self.points[2].x, self.points[2].y, 1],
		])

		m_12 = np.array([
			[self.points[0].x**2 + self.points[0].y**2, self.points[0].y, 1],
			[self.points[1].x**2 + self.points[1].y**2, self.points[1].y, 1],
			[self.points[2].x**2 + self.points[2].y**2, self.points[2].y, 1],
		])

		m_13 = np.array([
			[self.points[0].x**2 + self.points[0].y**2, self.points[0].x, 1],
			[self.points[1].x**2 + self.points[1].y**2, self.points[1].x, 1],
			[self.points[2].x**2 + self.points[2].y**2, self.points[2].x, 1],
		])

		m_14 = np.array([
			[self.points[0].x**2 + self.points[0].y**2, self.points[0].x, self.points[0].y],
			[self.points[1].x**2 + self.points[1].y**2, self.points[1].x, self.points[1].y],
			[self.points[2].x**2 + self.points[2].y**2, self.points[2].x, self.points[2].y],
		])
		
		try:
			x = 0.5 * det(m_12) / det(m_11)
			y = -0.5 * det(m_13) / det(m_11)
			self.center = Point(x, y)
			self.radius = np.sqrt(x**2 + y**2 + det(m_14)/det(m_11))
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
			return colinears(self.points[0], p, self.points[2], eps)

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


	def set_fig(self, f, ax):
		self.f = f
		self.ax = ax

	def print(
		self, 
		linestyle = "-", linecolor = "blue", linewidth=1, 
		marker = "o", pointcolor = "red"
	):
		self.print_curve(
			color = linecolor, 
			linestyle = linestyle, 
			linewidth=linewidth
		)
		self.print_points(
			marker = marker, 
			color=pointcolor
		)

	def print_points(self, marker="o", color="red"):
		set_current_subplot(self.f, self.ax)
		x = [self.points[0].x, self.points[2].x]
		y = [self.points[0].y, self.points[2].y]
		if self.printed_points is None:
			self.printed_points, = plt.plot(
				x, y, 
				marker = marker, 
				linestyle = "", 
				color = color
			)
		else:
			self.printed_points.set_xdata(x)
			self.printed_points.set_ydata(y)

	def print_curve(self, color = "blue", linestyle = "-", linewidth = 1):
		set_current_subplot(self.f, self.ax)

		if self.degenerate:
			x = [self.points[0].x, self.points[2].x]
			y = [self.points[0].y, self.points[2].y]

		else:
			theta_start, theta_end = self.angle_parameters()
			theta_tot = theta_end - theta_start

			x = self.center.x + np.array([self.radius * np.cos(theta_start + t * theta_tot / (SAMPLING-1)) for t in range(SAMPLING)])
			y = self.center.y + np.array([self.radius * np.sin(theta_start + t * theta_tot / (SAMPLING-1)) for t in range(SAMPLING)])

		if self.printed_line is None:
			self.printed_line, = plt.plot(
				x, y,
				linestyle = linestyle,
				color = color,
				linewidth = linewidth
			)
		else:
			self.printed_line.set_x_data(x)
			self.printed_line.set_y_data(y)


def colinears(p1, p2, p3, eps):
	return abs(0.5 * (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))) < eps * eps