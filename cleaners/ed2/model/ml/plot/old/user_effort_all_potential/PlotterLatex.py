import numpy as np


class PlotterLatex:

    def plot_list(self, label, fscore_lists, list_series, list_names, dboost_models, dboost_sizes, dboost_avg_f, xmax, legend_location=None, title=""):
        latex = "\\begin{tikzpicture}[scale=0.6]\n" + \
			    "\\begin{axis}[\n" + \
				"title={"+ title +"},\n" + \
				"xlabel={\\# Labelled Cells},\n" + \
				"ylabel={F1-score},\n" + \
                "xmin=0, xmax="+ str(xmax) +",\n" + \
				"ymin=0.0, ymax=1.0,\n"

        latex += "ytick={0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0},\n" + \
				"legend style={at={(0.5,-0.2)},anchor=north},\n" + \
				"xmajorgrids=true,\n" + \
				"ymajorgrids=true,\n" + \
                "cycle list name=exotic,\n" + \
				"grid style=dashed]\n\n"

        for i in range(len(list_series)):
            if 'dBoost' in list_names[i]:
                latex += "\\addplot+[dashed, mark=x] coordinates{"
            else:
                latex += "\\addplot+[no marks] coordinates{"


            latex += "(" + str(0) + "," + str(list_series[i]) + ")"
            latex += "(" + str(xmax + 1) + "," + str(list_series[i]) + ")"


            latex += "};\n"
            latex += "\\addlegendentry{"+ list_names[i] +"}\n"


        for i in range(len(dboost_models)):
            latex += "\\addplot+[mark=x] coordinates{"

            for l in range(len(dboost_sizes)):
                latex += "(" + str(dboost_sizes[l]) + "," + str(dboost_avg_f[i][l]) + ")"


            latex += "};\n"
            latex += "\\addlegendentry{"+ dboost_models[i] +"}\n"



        f_matrix = np.matrix(fscore_lists)

        lower_quartile = np.percentile(f_matrix, 25, axis=0)
        median = np.percentile(f_matrix, 50, axis=0)
        upper_quartile = np.percentile(f_matrix, 75, axis=0)
        minimum = np.min(f_matrix, axis=0).A1
        maximum = np.max(f_matrix, axis=0).A1

        print "minimum shape: " + str(minimum.shape)


        table = ""

        for i in range(len(lower_quartile)):
            latex += "\\boxplotlabels{"+ str(label[i]) +"}{"+ str(median[i]) +"}{"+ str(lower_quartile[i]) +"}{"+ str(upper_quartile[i]) +"}{"+ str(minimum[i]) +"}{"+ str(maximum[i]) +"}\n"
            #table += str(real_time[i] / 60.0) + " " + str(minimum[i]) + " " + str(lower_quartile[i]) + " " + str(median[i]) + " " + str(upper_quartile[i]) + " " + str(maximum[i]) + "\n"

        latex += "\\addlegendimage{mark=square,black}\n"
        latex += "\\addlegendentry{Our Approach}\n"

        print table




        latex += "\end{axis}\n" + \
		         "\end{tikzpicture}\n"

        return latex

    def __init__(self, data, label, fscore_lists,
                 dboost_models, dboost_sizes, dboost_avg_f,
                 nadeef_fscore,
                 openrefine_fscore,
                 legend_location=8, xmax=None, filename=None):

        self.number_dirty_columns = len(np.where (np.sum(data.matrix_is_error, axis=0) > 0)[0])

        plot_fscore_list = [nadeef_fscore, openrefine_fscore]
        plot_names_list = ['NADEEF', 'OpenRefine']


        latex = self.plot_list(label,
                               fscore_lists,
                               plot_fscore_list,
                               plot_names_list,
                               dboost_models, dboost_sizes, dboost_avg_f,
                               xmax,
                               legend_location,
                               filename)

        print latex
