import matplotlib.pyplot as plt
import numpy as np

def plot_list(ranges, list_series, list_names, title, x_max = None):
    fig = plt.figure()
    ax = plt.subplot(111)

    for i in range(len(list_series)):
        ax.plot(ranges[i], list_series[i], label=list_names[i])

    ax.set_ylabel('F1-score')

    ax.legend(loc=4)

    plt.show()


def plot_list_latex(ranges, list_series, list_names, title, x_max = None):
    latex = "\\begin{tikzpicture}[scale=0.6]\n" + \
            "\\begin{axis}[\n" + \
            "title={" + title + "},\n" + \
            "xlabel={\\# Labelled Cells},\n" + \
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


def calc_integral(ranges, list_series, list_names, x_max=None, sorted=True, normalize=True):
    integral = np.zeros(len(list_names))
    sum_labels = np.zeros(len(list_names))

    for i in range(len(list_names)):
        for iteration in range(len(list_series[i])):

            if ranges[i][iteration] < x_max:
                if iteration == 0:
                    integral[i] += (ranges[i][iteration] - 0.0) * list_series[i][iteration]
                    sum_labels[i] += (ranges[i][iteration] - 0.0)
                else:
                    integral[i] += (ranges[i][iteration] - ranges[i][iteration - 1]) * list_series[i][iteration]
                    sum_labels[i] += (ranges[i][iteration] - ranges[i][iteration - 1])

    if normalize:
        integral /= sum_labels

    if sorted:
        sorted_indeces = (integral * -1).argsort()
        integral = integral[sorted_indeces]
        list_names = np.array(list_names)[sorted_indeces]

    return list_names, integral

def plot_integral(ranges, list_series, list_names, title, x_max = None, sorted=True):

    list_names, integral = calc_integral(ranges, list_series, list_names, x_max, sorted=True)

    print list_names
    print integral

    N = len(list_names)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()

    ax.bar(ind, integral, width, color='r')
    plt.xticks(ind, list_names)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Integral')
    ax.set_title(title)

    plt.show()


def plot_integral_latex(ranges, list_series, list_names, title, x_max = None, sorted=True):

    list_names, integral = calc_integral(ranges, list_series, list_names, x_max, sorted=True)

    barchart = "\\begin{tikzpicture}[scale=0.6]\n" + \
			"\\begin{axis}[\n" + \
			"    title={" + title + "},\n" + \
			"    ybar,\n" + \
			"\tbar width=15pt,\n" + \
			"\tnodes near coords,\n" + \
			"    enlargelimits=0.15,\n" + \
			"    legend style={at={(0.5,-0.2)},anchor=north},\n" + \
			"    ylabel={Integral},\n" + \
			"    xtick=data,\n" + \
			"    x tick label style={rotate=45,anchor=east},\n" + \
			"    ymin=0.0, ymax=1.0\n" + \
			"    ]\n"


    for i in range(len(integral)):
        barchart += "\\addplot+[ybar] plot coordinates {(Test, "+ "{0:.2f}".format(integral[i]) +")};\n"

    barchart += "\\legend{"

    for i in range(len(integral)):
        barchart += "\\strut "+ list_names[i] + ","

    barchart = barchart[0:len(barchart)-1]

    barchart += "}\n" + \
			"\\end{axis}\n" + \
			"\\end{tikzpicture}"

    print barchart



def calc_outperform(ranges, list_series, list_names, to_out_perform, x_max = None, sorted=True, normalize=True):
    outperform = np.zeros(len(list_names))
    sum_labels = np.zeros(len(list_names))

    for i in range(len(list_names)):
        for iteration in range(len(list_series[i])):
            if list_series[i][iteration] >= to_out_perform:
                outperform[i] = ranges[i][iteration]
                break

        if outperform[i] == 0:
            outperform[i] = -1

    #if normalize:
    #    integral /= sum_labels

    if sorted:
        sorted_indeces = (outperform).argsort()
        outperform = outperform[sorted_indeces]
        list_names = np.array(list_names)[sorted_indeces]

    return list_names, outperform


def plot_outperform(ranges, list_series, list_names, title, to_outperform, x_max = None, sorted=True):

    list_names, outperform = calc_outperform(ranges, list_series, list_names, to_outperform, x_max, sorted=True)

    print list_names
    print outperform

    N = len(list_names)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()

    ax.bar(ind, outperform, width, color='r')
    plt.xticks(ind, list_names)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('#Labels')
    ax.set_title(title)

    plt.show()


def plot_outperform_latex(ranges, list_series, list_names, title, to_outperform, x_max = None, sorted=True):

    list_names, outperform = calc_outperform(ranges, list_series, list_names, to_outperform, x_max, sorted=True)

    barchart = "\\begin{tikzpicture}[scale=0.6]\n" + \
			"\\begin{axis}[\n" + \
			"    title={" + title + "},\n" + \
			"    ybar,\n" + \
			"\tbar width=15pt,\n" + \
			"\tnodes near coords,\n" + \
			"    enlargelimits=0.15,\n" + \
			"    legend style={at={(0.5,-0.2)},anchor=north},\n" + \
			"    ylabel={#Labels needed to outperform OpenRefine},\n" + \
			"    xtick=data,\n" + \
			"    x tick label style={rotate=45,anchor=east},\n" + \
			"    ymin=0.0\n" + \
			"    ]\n"


    for i in range(len(outperform)):
        barchart += "\\addplot+[ybar] plot coordinates {(Test, "+ str(outperform[i]) +")};\n"

    barchart += "\\legend{"

    for i in range(len(outperform)):
        barchart += "\\strut "+ list_names[i] + ","

    barchart = barchart[0:len(barchart)-1]

    barchart += "}\n" + \
			"\\end{axis}\n" + \
			"\\end{tikzpicture}"

    print barchart
