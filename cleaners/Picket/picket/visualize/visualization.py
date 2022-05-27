import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# set the colormap and centre the colorbar
class MidpointNormalize(mcolors.Normalize):
    """Normalise the colorbar."""
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def colorMap(matrix, labels, title='Attention Matrix'):
	norm = MidpointNormalize(0, matrix.max(), 0.)

	plt.imshow(matrix, cmap='RdBu', norm=norm)
	plt.colorbar()
	plt.xlabel('Attribute')
	plt.ylabel('Attribute')
	plt.xticks(range(len(labels)), labels)
	plt.xticks(rotation=90, fontsize=7)
	plt.yticks(range(len(labels)), labels)
	plt.yticks(fontsize=7)
	plt.tight_layout()
	plt.title(title)
	plt.show()