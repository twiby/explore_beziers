import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.append("..")
from constants import *
from utils import set_current_subplot

class PrintableArc:
	def __init__(self, f = None, ax = None):
		self.f = f
		self.ax = ax
		self.printed_points = None
		self.printed_line = None
		self.printed_mid_point = None

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

		# if self.printed_mid_point is None:
		# 	self.printed_mid_point, = plt.plot(
		# 		[self.points[1].x], [self.points[1].y], 
		# 		marker = "+", 
		# 		linestyle = "", 
		# 		color = color,
		# 		alpha = 0.6
		# 	)
		# else:
		# 	self.printed_mid_point.set_xdata([self.points[1].x])
		# 	self.printed_mid_point.set_ydata([self.points[1].y])

	def print_curve(self, color = "blue", linestyle = "-", linewidth = 1):
		set_current_subplot(self.f, self.ax)

		if self.degenerate:
			x = [self.points[0].x, self.points[2].x]
			y = [self.points[0].y, self.points[2].y]

		else:
			theta_start, theta_end = self.get_angles()
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

