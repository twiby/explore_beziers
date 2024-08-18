import sys
import copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

SAMPLING = 1000

INIT_CUTOFF = 0.5
INIT_BOT = 0.25
INIT_TOP = 0.75

MIN_EPS = 0.000005
MAX_EPS = 0.02
NB_EPS = 100

NB_POINTS_TO_TEST_ARC = 5
EPS_ARC = 0.001
MAX_ITERATION = 100
MAX_NB_ARCS = 1000

class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y
	def __add__(self, other):
		return Point(self.x + other.x, self.y + other.y)
	def __sub__(self, other):
		return Point(self.x - other.x, self.y - other.y)
	def __mul__(self, other):
		'''Only for scalars operands'''
		if type(other) is Point:
			raise ValueError
		return Point(self.x * other, self.y * other)
	def __rmul__(self, other):
		return self.__mul__(other)
	def __neg__(self):
		return Point(-self.x, -self.y)
	def __str__(self):
		return "Point(" + str(self.x) + ", " + str(self.y) + ")"
	def __repr__(self):
		return str(self)
	def angle(self):
		return np.arctan2(self.y, self.x)
	def norm(self):
		return np.linalg.norm([self.x, self.y])


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

class BezierCurveApproximation:
	def __init__(self, bez):
		self.original_bezier = bez.copy()
		self.current_bezier = bez.copy()
		self.t = 0
		self.arcs = []

	def compute(self, eps):
		'''shouldn't be reused after this call, this should be consumed'''
		print("COMPUTING", eps)
		counter = 0 
		while self.t < 1:
			counter += 1
			if counter > MAX_NB_ARCS:
				print("MAX NB OF ARCS REACHED")
				return self.arcs
				# raise ValueError("MAX ITERATION REACHED")

			arc, t = self.current_bezier.approximate_beginning_with_arc(eps)
			self.arcs.append(arc)
			self.t = t + self.t * (1-t)
			self.current_bezier = Bezier(*self.original_bezier.reverse().get_sub_bez_points(1-self.t)).reverse()

		return self.arcs

class Bezier:
	def __init__(self, p0, p1, p2, p3, f = None, ax = None):
		if not type(p0) is Point:
			raise ValueError
		if not type(p1) is Point:
			raise ValueError
		if not type(p2) is Point:
			raise ValueError
		if not type(p3) is Point:
			raise ValueError

		self.points = [p0, p1, p2, p3]
		self.init_coefs()

		self.f = f
		self.ax = ax
		self.printed_points = None
		self.printed_line = None

	def set_points(self, p0, p1, p2, p3):
		self.points = [p0, p1, p2, p3]
		self.init_coefs()

	def init_coefs(self):
		[p0, p1, p2, p3] = self.points
		self.coefs = [
			p0,
			-3*p0 + 3*p1, 
			3*p0 - 6*p1 + 3*p2,
			-p0 + 3*p1 - 3*p2 + p3
		]

	def reverse(self):
		points = self.points.copy()
		points.reverse()
		return Bezier(*points, self.f, self.ax)

	def sample(self, t):
		return \
			self.coefs[0] + \
			t * self.coefs[1] + \
			t*t * self.coefs[2] + \
			t*t*t * self.coefs[3]

	def sample_speed(self, t):
		return \
			self.coefs[1] + \
			2*t * self.coefs[2] + \
			3*t*t * self.coefs[3]

	def get_sub_bez_points(self, t):
		p0 = self.points[0]
		p1 = self.points[0] * (1-t) + self.points[1] * t
		p2 = self.points[1] * (1-t) + self.points[2] * t
		p2 = p1 * (1-t) + p2 * t
		p3 = self.sample(t)
		return (p0, p1, p2, p3)

	def split_at(self, t):
		pre_points = self.get_sub_bez_points(t)
		post_points = self.reverse().get_sub_bez_points(1-t)
		return (pre_points, post_points)

	def split_at_both(self, t1, t2):
		if t2 < t1:
			return split_at_both(t2, t1)
		pre_points = self.get_sub_bez_points(t2)
		b = Bezier(*pre_points)
		post_points = b.reverse().get_sub_bez_points(1 - (t1 / t2))
		return post_points.reverse()

	def test_arc(self, arc, max_t, eps):
		eps_t = max_t / 1000
		p = self.sample(eps_t)
		if not arc.point_on_arc(p, eps):
			return False
		p = self.sample(max_t-eps_t)
		if not arc.point_on_arc(p, eps):
			return False

		for n in range(NB_POINTS_TO_TEST_ARC):
			t = n * max_t / NB_POINTS_TO_TEST_ARC
			point = self.sample(t)
			if not arc.point_on_arc(point, eps):
				return False
		return True

	def approximate_beginning_with_arc(self, eps):
		t = 1
		step = 1

		arc = Arc(self.points[0], self.sample(t/2), self.sample(t))
		arc_is_good = self.test_arc(arc, t, eps)
		prev_arc_is_good = arc_is_good

		# early exit
		if arc_is_good:
			arc.set_fig(self.f, self.ax)
			return arc, t

		counter = 0
		while not (prev_arc_is_good and not arc_is_good):
			counter += 1
			if counter > MAX_ITERATION:
				raise ValueError("MAX_ITERATION reached")

			step /= 2
			prev_arc_is_good = arc_is_good
			prev_arc = copy.deepcopy(arc)

			if prev_arc_is_good:
				t += step
			else:
				t -= step

			arc = Arc(self.points[0], self.sample(t/2), self.sample(t))
			arc_is_good = self.test_arc(arc, t, eps)
			
		prev_arc.set_fig(self.f, self.ax)
		return prev_arc, t-step

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

	def copy(self):
		return Bezier(*self.points)

def colinears(p1, p2, p3, eps):
	return abs(0.5 * (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))) < eps * eps

def set_current_subplot(f, ax):
	plt.figure(f)
	plt.sca(ax)

def splitting_interface(beziers):
	fig, axs = plt.subplots(2, 2)
	axs = axs.flatten()
	fig.subplots_adjust(bottom=0.25)
	ax_slider = fig.add_axes([0.15, 0.1, 0.75, 0.03])
	slider = Slider(
	    ax = ax_slider,
	    label = 'Cutting point',
	    valmin = 0.0,
	    valmax = 1.0,
	    valinit = INIT_CUTOFF,
	)

	for n in range(len(beziers)):
		beziers[n].set_fig(fig, axs[n])

	pre_beziers = []
	post_beziers = []

	for n in range(len(beziers)):
		b = beziers[n]
		(pre_points, post_points) = b.split_at(INIT_CUTOFF)
		pre_beziers.append(Bezier(*pre_points, fig, axs[n]))
		post_beziers.append(Bezier(*post_points, fig, axs[n]))

		b.print(linewidth = 1.3)
		pre_beziers[-1].print(marker="x", pointcolor="orange", linecolor="orange")
		post_beziers[-1].print(marker="x", pointcolor="green", linecolor="green")

	def update(val):
		for n in range(len(beziers)):
			b = beziers[n]
			(pre_points, post_points) = b.split_at(val)
			pre_beziers[n].set_points(*pre_points)
			pre_beziers[n].print()
			post_beziers[n].set_points(*post_points)
			post_beziers[n].print()
		fig.canvas.draw_idle()
	slider.on_changed(update)
	plt.show()

def double_split_interface(beziers):
	fig, axs = plt.subplots(2, 2)
	axs = axs.flatten()
	fig.subplots_adjust(bottom=0.25)
	ax_slider = fig.add_axes([0.15, 0.1, 0.75, 0.03])
	slider_bot = Slider(
	    ax = ax_slider,
	    label = 'Bottom cut',
	    valmin = 0.0,
	    valmax = 0.5,
	    valinit = INIT_BOT,
	)
	ax_slider = fig.add_axes([0.15, 0.13, 0.75, 0.03])
	slider_top = Slider(
	    ax = ax_slider,
	    label = 'Top cut',
	    valmin = 0.5,
	    valmax = 1.0,
	    valinit = INIT_TOP,
	)

	cut = [INIT_BOT, INIT_TOP]
	cut_beziers = []
	for n in range(len(beziers)):
		b = beziers[n]
		b.set_fig(fig, axs[n])
		b.print(linewidth = 1.3)
		cut_points = b.split_at_both(*cut)
		cut_beziers.append(Bezier(*cut_points, fig, axs[n]))
		cut_beziers[-1].print(marker = "x", pointcolor = "orange", linecolor = "orange")

	def update():
		for n in range(len(beziers)):
			b = beziers[n]
			cut_points = b.split_at_both(*cut)
			cut_beziers[n].set_points(*cut_points)
			cut_beziers[n].print()
		fig.canvas.draw_idle()
	def update_bot(val):
		cut[0] = val
		update()
	def update_top(val):
		cut[1] = val
		update()

	slider_top.on_changed(update_top)
	slider_bot.on_changed(update_bot)

	plt.show()

def approximate_with_curves(beziers):
	fig, axs = plt.subplots(2, 2)
	fig.suptitle("Beziers curves approximation with circular arcs")
	axs = axs.flatten()
	fig.subplots_adjust(bottom=0.25)
	ax_slider = fig.add_axes([0.15, 0.1, 0.75, 0.03])
	log_init_eps = (np.log(MIN_EPS) + np.log(MAX_EPS)) / 2
	init_eps = np.exp(log_init_eps)
	slider = Slider(
	    ax = ax_slider,
	    label = 'log(precision)',
	    valmin = np.log(MIN_EPS),
	    valmax = np.log(MAX_EPS),
	    valinit = log_init_eps,
	)

	all_arcs = []
	for n in range(len(beziers)):
		axs[n].axis('equal')
		beziers[n].set_fig(fig, axs[n])
		beziers[n].print(linewidth = 1.3)

		approximator = BezierCurveApproximation(beziers[n])
		arcs = approximator.compute(init_eps)
		for arc in arcs:
			arc.set_fig(fig, axs[n])
			arc.print(linecolor="orange", marker="x", pointcolor="orange")
		all_arcs.append(arcs)

	def update(val):
		eps = np.exp(val)
		if eps <= 0:
			return
		for n in range(len(beziers)):
			set_current_subplot(beziers[n].f, beziers[n].ax)
			for arc in all_arcs[n]:
				arc.printed_line.remove()
				arc.printed_points.remove()
			approximator = BezierCurveApproximation(beziers[n])
			arcs = approximator.compute(eps)
			for arc in arcs:
				arc.set_fig(fig, axs[n])
				arc.print(linecolor="orange", marker="x", pointcolor="orange")
			all_arcs[n] = arcs
		fig.canvas.draw_idle()
	slider.on_changed(update)

	plt.show(block = True)

def nb_approximation(beziers):
	fig, axs = plt.subplots(2, 2)
	fig.suptitle("Number of arcs needed to approximate the Beziers curves")
	axs = axs.flatten()

	log_min_eps = np.log(MIN_EPS)
	log_max_eps = np.log(MAX_EPS)
	log_x = np.array([log_min_eps + n*(log_max_eps - log_min_eps)/(NB_EPS-1) for n in range(NB_EPS)])
	x = np.exp(log_x)

	for n in range(len(beziers)):
		print("BEZIER", n)
		set_current_subplot(fig, axs[n])
		y = np.array([len(BezierCurveApproximation(beziers[n]).compute(eps)) for eps in x])

		log_y = np.log(y)
		poly = np.polynomial.polynomial.Polynomial.fit(log_x, log_y, 1)
		log_y2 = poly(log_x)

		plt.plot(x, y, label = "nb of arcs")
		plt.plot(x, np.exp(log_y2), color = "red", alpha = 0.4, label = "fit degree: " + str(abs(poly.coef[1])))
		plt.xscale("log")
		plt.yscale("log")
		plt.ylabel("Number of arcs generated")
		plt.xlabel("Precision asked")
		plt.legend()

	plt.show(block = False)


if __name__ == "__main__":
	beziers = [
		Bezier(
			Point(0, 0), 
			Point(0, 1), 
			Point(2, 1), 
			Point(2, 0), 
		),
		Bezier(
			Point(0, 0), 
			Point(1, 0), 
			Point(2, 0), 
			Point(3, 0), 
		),
		Bezier(
			Point(0, 0), 
			Point(2, 1), 
			Point(0, 1), 
			Point(2, 0), 
		),
		Bezier(
			Point(0, 0), 
			Point(2.5, 1), 
			Point(-0.5, 1), 
			Point(2, 0), 
		),
	]

	# splitting_interface(beziers)
	# double_split_interface(beziers)

	nb_approximation(beziers)
	approximate_with_curves(beziers)
