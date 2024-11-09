import numpy as np

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
	def __truediv__(self, other):
		if type(other) is Point:
			raise ValueError
		return Point(self.x / other, self.y / other)		
	def __neg__(self):
		return Point(-self.x, -self.y)
	def __str__(self):
		return "Point(" + str(self.x) + ", " + str(self.y) + ")"
	def __repr__(self):
		return str(self)
	def angle(self):
		return np.arctan2(self.y, self.x)
	def norm_squared(self):
		return self.dot(self)
	def norm(self):
		return np.linalg.norm([self.x, self.y])
	def cross(self, other):
		return self.x * other.y - self.y * other.x
	def dot(self, other):
		return self.x * other.x + self.y * other.y
