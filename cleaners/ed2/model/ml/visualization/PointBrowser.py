import numpy as np
import webbrowser

class PointBrowser(object):
    """
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower axes.  Use the 'n'
    and 'p' keys to browse through the next and previous points
    """

    def __init__(self, fig, ax, xs, ys, labels):
        self.fig = fig
        self.ax = ax
        self.xs = xs
        self.ys = ys
        self.labels = labels

        self.lastind = 0

        self.text = self.ax.text(0.05, 0.95, 'selected: none',
                            transform=ax.transAxes, va='top')
        self.selected, = self.ax.plot([self.xs[0]], [self.ys[0]], 'o', ms=12, alpha=0.4,
                                 color='yellow', visible=False)

        self.current_repo = ""

    def onpress(self, event):
        if self.lastind is None:
            return
        if event.key not in ('n', 'p', 'enter'):
            return

        if event.key == 'enter':
            webbrowser.open(self.current_repo, new=0, autoraise=True)
            return

        if event.key == 'n':
            inc = 1
        else:
            inc = -1

        self.lastind += inc
        self.lastind = np.clip(self.lastind, 0, len(self.xs) - 1)
        self.update()

    def onpick(self, event):

        N = len(event.ind)
        if not N:
            return True

        # the click locations
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata

        distances = np.hypot(x - self.xs[event.ind], y - self.ys[event.ind])
        indmin = distances.argmin()
        dataind = event.ind[indmin]

        self.lastind = dataind
        self.update()

    def update(self):
        if self.lastind is None:
            return

        dataind = self.lastind

        self.selected.set_visible(True)
        self.selected.set_data(self.xs[dataind], self.ys[dataind])

        self.text.set_text('selected: ' + self.labels[dataind])
        print ('selected: ' + self.labels[dataind])

        self.current_repo = self.labels[dataind]


        self.fig.canvas.draw()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    X = np.random.rand(100, 200)
    xs = np.mean(X, axis=1)
    ys = np.std(X, axis=1)

    labels = ["test + "+ str(x) for x in range(len(xs))]

    fig, (ax) = plt.subplots(1, 1)
    ax.set_title('click on point to plot time series')
    line, = ax.plot(xs, ys, 'o', picker=5)  # 5 points tolerance

    browser = PointBrowser(fig, ax, xs, ys, labels)

    fig.canvas.mpl_connect('pick_event', browser.onpick)
    fig.canvas.mpl_connect('key_press_event', browser.onpress)

    plt.show()