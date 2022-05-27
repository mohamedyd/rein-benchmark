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
	# dataset_log_files[Salary().name] = "hospital"  # be careful
	# dataset_log_files[Book().name] = "hospital"  # be careful

	#potential_model_dir = '/home/felix/ExampleDrivenErrorDetection/potential models/current_total_f'
	potential_model_dir = '/home/felix/ExampleDrivenErrorDetection/potential models/simulation100data'

	tp_model = pickle.load(open(potential_model_dir + "/tp_model_" + "XGBoost" + ".p"))
	fp_model = pickle.load(open(potential_model_dir + "/fp_model_" + "XGBoost" + ".p"))
	fn_model = pickle.load(
		open(potential_model_dir + "/fn_model_XGBoost.p"))

	return tp_model, fp_model, fn_model

def get_estimated_tp_fp_fn(x, n, dataSet, features, which_features_to_use_n, runs=41,tp_model=None, fp_model=None, fn_model=None):

	'''
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

	for i in range(10):
		feature_names.append('batch_certainty' + str(i))

	feature_names.append('no_change_0')
	feature_names.append('no_change_1')
	feature_names.append('change_0_to_1')
	feature_names.append('change_1_to_0')


	#print(str(feature_names))

	size = len(feature_names)
	for s in range(size):
		feature_names.append(feature_names[s] + "_old")
	'''


	f_p = 0
	f_n = 1
	t_p = 2

	estimated_scores = np.zeros((n, runs, 3))
	for run in range(runs):
		for col in range(n):
			if run > 0:
				vector = []
				vector.extend(x[col + n * run])
				vector.extend(x[col + (n - 1) * run])

				feature_vector_new = np.matrix(vector)[:, which_features_to_use_n]

				if tp_model == None:
					tp_model, fp_model, fn_model = load_model(dataSet)


				print feature_vector_new.shape
				print len(features)

				mat_potential = xgb.DMatrix(feature_vector_new, feature_names=features)
				estimated_scores[col, run, t_p] = tp_model.predict(mat_potential)
				estimated_scores[col, run, f_p] = fp_model.predict(mat_potential)
				estimated_scores[col, run, f_n] = fn_model.predict(mat_potential)

	return estimated_scores

def calculateEstimatedImpacts(column_states, estimated_scores):
	tp_all_sum = 0.0
	fp_all_sum = 0.0
	fn_all_sum = 0.0

	f_p = 0
	f_n = 1
	t_p = 2

	for col_i in range(len(column_states)):
		tp_all_sum += estimated_scores[col_i, column_states[col_i], t_p]
		fp_all_sum += estimated_scores[col_i, column_states[col_i], f_p]
		fn_all_sum += estimated_scores[col_i, column_states[col_i], f_n]

	estimated_total_fscore = (2 * tp_all_sum) / ((2 * tp_all_sum) + (fp_all_sum + fn_all_sum))

	#print "estimated F1-score: " + str(estimated_total_fscore)

	estimated_possible_total_fscore = np.zeros(len(column_states))
	estimated_difference_total_fscore = np.zeros(len(column_states))
	for key_column in range(len(column_states)):
		estimated_possible_total_fscore[key_column] = (2 * (tp_all_sum + estimated_scores[key_column, column_states[key_column], f_n])) / (
				(2 * (tp_all_sum + estimated_scores[key_column, column_states[key_column], f_n])) + (
				(fp_all_sum - estimated_scores[key_column, column_states[key_column], f_p]) + (fn_all_sum - estimated_scores[key_column, column_states[key_column], f_n])))
		estimated_difference_total_fscore[key_column] = estimated_possible_total_fscore[
															key_column] - estimated_total_fscore

	return estimated_difference_total_fscore


def select_by_estimated_max_f_impact(tensor_run, column_states, current_f_list, col_list, estimated_scores, steps, use_sum=True):
	for s in range(steps):
		estimated_impact = calculateEstimatedImpacts(column_states, estimated_scores)

		chosen_id = np.argmax(estimated_impact)

		column_states[chosen_id] += 1
		col_list.append(chosen_id)
		current_f_list.append(calc_total_f1(tensor_run, column_states))

	return current_f_list, col_list
