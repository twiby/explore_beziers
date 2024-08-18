import sys
import copy

sys.path.append("..")
from arc import Arc
from point import Point
from constants import *

from .printable_bezier import PrintableBezier

class Bezier(PrintableBezier):
	def __init__(self, p0, p1, p2, p3, *args, **kwargs):
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
		PrintableBezier.__init__(self, *args, **kwargs)

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

	def copy(self):
		return Bezier(*self.points)