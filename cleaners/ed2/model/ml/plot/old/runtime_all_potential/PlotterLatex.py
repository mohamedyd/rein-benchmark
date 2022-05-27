import numpy as np


class PlotterLatex:

    def calc_waiting_time(self, real_time):
        save_time = []
        time = real_time[0] + self.init_labels * self.sec_pro_label
        save_time.append(time)
        for i in range(1, self.number_dirty_columns):
            time += (real_time[i] - real_time[i - 1]) + self.init_labels * self.sec_pro_label
            save_time.append(time)

        for i in range(self.number_dirty_columns, len(real_time)):
            time += (real_time[i] - real_time[i - 1]) + self.std_labels * self.sec_pro_label
            save_time.append(time)

        return save_time

    def plot_list(self, real_time, fscore_lists, y_series, list_series, list_names, the_end, legend_location=None, title=""):
        end_min = int(the_end / 60.0)

        latex = "\\begin{tikzpicture}[scale=0.6]\n" + \
			    "\\begin{axis}[\n" + \
				"title={"+ title +"},\n" + \
				"xlabel={Runtime (min)},\n" + \
				"ylabel={F-score},\n" + \
				"xmin=0, xmax=" + str(end_min) + ",\n" + \
				"ymin=0.0, ymax=1.0,\n"

        latex += "ytick={0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0},\n" + \
				"legend style={at={(0.5,-0.2)},anchor=north},\n" + \
				"xmajorgrids=true,\n" + \
				"ymajorgrids=true,\n" + \
                "cycle list name=exotic,\n" + \
				"grid style=dashed]\n\n"

        for i in range(len(list_series)):
            time_val = np.array(y_series[i]) / 60.0

            if 'dBoost' in list_names[i]:
                latex += "\\addplot+[dashed, mark=x] coordinates{"
            else:
                latex += "\\addplot+[mark=x] coordinates{"



            for c in range(len(list_series[i])):
                latex += "(" + str(time_val[c]) + "," + str(list_series[i][c]) + ")"
            latex += "};\n"
            latex += "\\addlegendentry{"+ list_names[i] +"}\n"


        f_matrix = np.matrix(fscore_lists)

        lower_quartile = np.percentile(f_matrix, 25, axis=0)
        median = np.percentile(f_matrix, 50, axis=0)
        upper_quartile = np.percentile(f_matrix, 75, axis=0)
        minimum = np.min(f_matrix, axis=0).A1
        maximum = np.max(f_matrix, axis=0).A1

        print "minimum shape: " + str(minimum.shape)


        table = ""

        for i in range(len(lower_quartile)):
            latex += "\\boxplot{"+ str(real_time[i] / 60.0) +"}{"+ str(median[i]) +"}{"+ str(lower_quartile[i]) +"}{"+ str(upper_quartile[i]) +"}{"+ str(minimum[i]) +"}{"+ str(maximum[i]) +"}\n"
            #table += str(real_time[i] / 60.0) + " " + str(minimum[i]) + " " + str(lower_quartile[i]) + " " + str(median[i]) + " " + str(upper_quartile[i]) + " " + str(maximum[i]) + "\n"

        latex += "\\addlegendimage{mark=square,black}\n"
        latex += "\\addlegendentry{Our Approach}\n"

        print table




        latex += "\end{axis}\n" + \
		         "\end{tikzpicture}\n"

        return latex

    def __init__(self, data, real_time, fscore_lists,
                 runtime_in_sec_list, fscore_res_list, label_rows_list, name_list,
                 nadeef_time_list, nadeef_fscore_list,
                 openrefine_time_in_min, openrefine_fscore_res,
                 legend_location=8, end_time=None, filename=None):

        self.number_dirty_columns = len(np.where (np.sum(data.matrix_is_error, axis=0) > 0)[0])
        number_dirty_columns_min_1 = self.number_dirty_columns - 1



        print number_dirty_columns_min_1

        self.init_labels = 4
        self.std_labels = 10
        self.sec_pro_label = 3.0

        time = self.number_dirty_columns * self.init_labels * self.sec_pro_label

        approx_time = []
        approx_time_non_parallel = self.calc_waiting_time(real_time)


        for i in range(number_dirty_columns_min_1, len(fscore_lists[0])):
            approx_time.append(time)
            time = time + (self.std_labels * self.sec_pro_label)

        print approx_time_non_parallel

        print len(approx_time)


        time_list = []
        for d in range(len(runtime_in_sec_list)):
            time_direct = []
            for tt in range(len(label_rows_list[d])):
                time_direct.append((data.get_number_dirty_columns() * label_rows_list[d][tt] * self.sec_pro_label) + runtime_in_sec_list[d][tt])
            time_list.append(time_direct)

        time_max = []
        for the_time in time_list:
            time_max.extend(the_time)
        time_max.extend(nadeef_time_list)
        time_max.append((openrefine_time_in_min*60))
        time_max.append(approx_time_non_parallel[-1])

        if end_time == None:
            the_end = np.max(time_max) + 60.0
        else:
            the_end = end_time

        nadeef_fscore = nadeef_fscore_list
        nadeef_time = nadeef_time_list

        openrefine_fscore = [0.0, 0.0, openrefine_fscore_res, openrefine_fscore_res]
        openrefine_time = [0, (openrefine_time_in_min - 0.001) * 60, openrefine_time_in_min * 60, the_end]


        plot_time_list = [nadeef_time, openrefine_time]
        plot_fscore_list = [nadeef_fscore, openrefine_fscore]
        plot_names_list = ['NADEEF', 'OpenRefine']


        for d in range(len(time_list)):
            plot_time_list.append(time_list[d])
            plot_fscore_list.append(fscore_res_list[d])
            plot_names_list.append(name_list[d])


        latex = self.plot_list(approx_time_non_parallel, fscore_lists, plot_time_list, plot_fscore_list, plot_names_list, the_end, legend_location, filename)

        print latex
