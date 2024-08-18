import sys
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.append("..")
from utils import set_current_subplot
from constants import *

class PrintableBezier:
	def __init__(self, f = None, ax = None):
		self.f = f
		self.ax = ax
		self.printed_points = None
		self.printed_line = None

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
		x = [p.x for p in self.points]
		y = [p.y for p in self.points]
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
		sample_points = [self.sample(t/(SAMPLING-1)) for t in range(SAMPLING)]
		x = [p.x for p in sample_points]
		y = [p.y for p in sample_points]
		if self.printed_line is None:
			self.printed_line, = plt.plot(
				x, y,
				linestyle = linestyle,
				color = color,
				linewidth = linewidth
			)
		else:
			self.printed_line.set_xdata(x)
			self.printed_line.set_ydata(y)