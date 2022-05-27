import pickle

from ml.datasets.flights.FlightHoloClean import FlightHoloClean
from ml.datasets.hospital.HospitalHoloClean import HospitalHoloClean
from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
from ml.datasets.salary_data.Salary import Salary
from ml.datasets.luna.book.Book import Book
import numpy as np
from ml.simulate.common.utils import calc_total_f1
import xgboost as xgb


def load_model(dataSet):
	dataset_log_files = {}
	dataset_log_files[HospitalHoloClean().name] = "hospital"
	dataset_log_files[BlackOakDataSetUppercase().name] = "blackoak"
	dataset_log_files[FlightHoloClean().name] = "flight"
	# not yet
	#dataset_log_files[Salary().name] = "hospital"  # be careful
	#dataset_log_files[Book().name] = "hospital"  # be careful

	potential_model_dir = '/home/felix/ExampleDrivenErrorDetection/potential models/unique_false_current_hist'

	return pickle.load(
		open(potential_model_dir + "/model" + dataset_log_files[dataSet.name] + "_" + "XGBoost" + ".p"))


def go_to_next_column_prob(column_id, pred_potential):
	potentials = np.zeros(len(pred_potential))
	map_index_to_key = {}
	map_id = 0
	for key in pred_potential.keys():
		map_index_to_key[map_id] = key
		potentials[map_id] = pred_potential[key]
		map_id += 1

	# print potentials

	# new_potential = np.square(potentials)
	new_potential = potentials
	new_potential = new_potential / np.sum(new_potential)

	# return map_index_to_key[np.random.choice(len(new_potential), 1, p=new_potential)[0]]
	return map_index_to_key[np.argmax(new_potential)]


def get_false_prediction(x, which_features_to_use_n, n, dataSet, runs=41):
	feature_names = ['distinct_values_fraction', 'labels', 'certainty', 'certainty_stddev',
							   'minimum_certainty']

	for i in range(100):
		feature_names.append('certainty_histogram' + str(i))

	feature_names.append('predicted_error_fraction')

	for i in range(7):
		feature_names.append('icross_val' + str(i))

	feature_names.append('mean_cross_val')
	feature_names.append('stddev_cross_val')

	feature_names.append('training_error_fraction')

	for i in range(100):
		feature_names.append('change_histogram' + str(i))

	feature_names.append('mean_squared_certainty_change')
	feature_names.append('stddev_squared_certainty_change')


	feature_names.append('no_change_0')
	feature_names.append('no_change_1')
	feature_names.append('change_0_to_1')
	feature_names.append('change_1_to_0')

	print(str(feature_names))

	size = len(feature_names)
	for s in range(size):
		feature_names.append(feature_names[s] + "_old")

	model = None

	fpfn = np.zeros((n, runs))
	for run in range(runs):
		for col in range(n):
			if run > 0:
				vector = []
				vector.extend(x[col + n * run][which_features_to_use_n])
				vector.extend(x[col + (n - 1) * run][which_features_to_use_n])

				feature_vector_new = np.matrix(vector)

				if model == None:
					model = load_model(dataSet)

				mat_potential = xgb.DMatrix(feature_vector_new, feature_names=feature_names)
				fpfn[col, run] = model.predict(mat_potential)


	return fpfn


def select_by_max_false_prediction(tensor_run, column_states, current_f_list, col_list, fpfn, steps, use_sum=True):
	for s in range(steps):
		max_false_pred = -1.0
		max_false_id = -1
		for col_i in range(len(column_states)):
			if fpfn[col_i][column_states[col_i]] > max_false_pred:
				max_false_pred = fpfn[col_i][column_states[col_i]]
				max_false_id = col_i

		column_states[max_false_id] += 1
		col_list.append(max_false_id)
		current_f_list.append(calc_total_f1(tensor_run, column_states))

	return current_f_list, col_list
