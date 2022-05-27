from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import colorsys
import matplotlib.lines as mlines
import numpy as np
import matplotlib.pyplot as plt
from ml.visualization.PointBrowser import PointBrowser

class TSNEPlot(object):
    def __init__(self, data, target, values, N = 10000):
        self.data = data
        self.target = target
        self.N = N
        self.values = values

        svd = TruncatedSVD(n_components=(data.shape[1]-1), n_iter=7, random_state=42)
        svd.fit(data)
        self.data_dense = svd.transform(data)

        model = TSNE(n_components=2, random_state=0, method="barnes_hut")
        self.Y = model.fit_transform(self.data_dense[0:N])


    def plot(self):
        category_list = {0: "correct", 1: "error"}
        colors = {0: 0.5, 1: 0.0}

        fig, (ax) = plt.subplots(1, 1)

        plts = []
        labels = []

        color_list = np.zeros((self.Y[:, 0].size, 3))
        for i in range(0, len(category_list)):
            mask = np.where(self.target[0:self.N] == i)
            #color_float = float(i) / 2.0
            rgb = colorsys.hsv_to_rgb(colors[i], 1.0, 1.0)
            color_list[mask] = rgb

            plts.append(mlines.Line2D([], [], color=rgb, markersize=15, label=category_list[i]))
            labels.append(category_list[i])

        ax.scatter(self.Y[:, 0], self.Y[:, 1], c=color_list, picker=5)

        ax.legend(plts, labels,
                  scatterpoints=1,
                  loc='lower left',
                  ncol=3,
                  fontsize=8)

        browser = PointBrowser(fig, ax, self.Y[:, 0], self.Y[:, 1], self.values[0:self.N])

        fig.canvas.mpl_connect('pick_event', browser.onpick)
        fig.canvas.mpl_connect('key_press_event', browser.onpress)
        fig.canvas.set_window_title('t-SNE Browser')

        plt.show()