import matplotlib.pyplot as plt
import numpy as np


class Plotter2:

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

    def plot_list(self, y_series, list_series, list_names, the_end, legend_location=None):
        fig = plt.figure()
        ax = plt.subplot(111)

        for i in range(len(list_series)):
            if 'dBoost' in list_names[i]:
                ax.plot(np.array(y_series[i]) / 60.0, list_series[i], label=list_names[i], linestyle='dashdot')
            else:
                ax.plot(np.array(y_series[i]) / 60.0, list_series[i], label=list_names[i])

        ax.set_ylabel('F-score')
        ax.set_xlabel('Runtime in minutes')

        ax.set_xlim([0, (the_end / 60.0)])
        ax.set_ylim([0.0, 1.0])

        return ax

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


        ax = self.plot_list(plot_time_list, plot_fscore_list, plot_names_list, the_end, legend_location)

        mins = np.min(np.matrix(fscore_lists), axis=0).A1
        maxes = np.max(np.matrix(fscore_lists), axis=0).A1
        means = np.mean(np.matrix(fscore_lists), axis=0).A1
        std = np.std(np.matrix(fscore_lists), axis=0).A1

        # create stacked errorbars:
        ax.errorbar(np.array(approx_time_non_parallel) / 60.0, means, std, label='Our approach', color="black")
        ax.errorbar(np.array(approx_time_non_parallel) / 60.0, means, [means - mins, maxes - means], fmt='.k', ecolor='gray', lw=1)

        if legend_location != None:
            ax.legend(loc=legend_location)

        if filename != None:
            plt.savefig(filename + ".pgf")
        else:
            plt.show()
