import matplotlib as mpl
import matplotlib.pyplot as plt

def set_current_subplot(f, ax):
	plt.figure(f)
	plt.sca(ax)