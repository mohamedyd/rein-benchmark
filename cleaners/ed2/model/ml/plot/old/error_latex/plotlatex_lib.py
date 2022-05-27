import matplotlib.pyplot as plt
import numpy as np


def plot_list_latex(ranges, list_series, list_names, title, x_max = None):
    latex = "\\begin{tikzpicture}[scale=0.6]\n" + \
            "\\begin{axis}[\n" + \
            "title={" + title + "},\n" + \
            "xlabel={# Labelled Cells},\n" + \
            "ylabel={F1-score},\n" + \
            "ymin=0.0, ymax=1.0,\n"

    if x_max != None:
        latex += "xmin=0.0, xmax="+ str(x_max) +",\n"

    latex += "ytick={0.0,0.5,1.0},\n" + \
             "legend style={at={(0.5,-0.2)},anchor=north},\n" + \
             "xmajorgrids=true,\n" + \
             "ymajorgrids=true,\n" + \
             "cycle list name=color,\n" + \
             "grid style=dashed]\n\n"

    for i in range(len(list_series)):
        time_val = ranges[i]

        latex += "\\addplot+[mark=none] coordinates{"

        for c in range(len(list_series[i])):
            latex += "(" + str(time_val[c]) + "," + str(list_series[i][c]) + ")"
        latex += "};\n"
        latex += "\\addlegendentry{" + list_names[i] + "}\n"

    latex += "\end{axis}\n" + \
             "\end{tikzpicture}\n"


    print latex

def plot_list(ranges, list_series, list_names, title, x_max = None):
    fig = plt.figure()
    ax = plt.subplot(111)

    for i in range(len(list_series)):
        ax.plot(ranges[i], list_series[i], label=list_names[i])

    ax.set_ylabel('F1-score')

    ax.legend(loc=4)

    plt.show()