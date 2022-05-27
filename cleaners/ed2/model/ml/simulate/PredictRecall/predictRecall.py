import pickle

from ml.datasets.flights.FlightHoloClean import FlightHoloClean
from ml.datasets.hospital.HospitalHoloClean import HospitalHoloClean
from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
from ml.datasets.salary_data.Salary import Salary
from ml.datasets.luna.book.Book import Book
import numpy as np
from ml.simulate.common.utils import calc_total_f1
import xgboost as xgb

def calculateEstimatedRecallImpacts(column_states, estimated_scores):
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

	estimated_total_recall = tp_all_sum / (tp_all_sum + fn_all_sum)

	#print "estimated F1-score: " + str(estimated_total_fscore)

	estimated_possible_total_recall = np.zeros(len(column_states))
	estimated_difference_total_recall = np.zeros(len(column_states))
	for key_column in range(len(column_states)):
		estimated_possible_total_recall[key_column] = (tp_all_sum + estimated_scores[key_column, column_states[key_column], f_n]) / (
				(tp_all_sum + estimated_scores[key_column, column_states[key_column], f_n]) + (fn_all_sum - estimated_scores[key_column, column_states[key_column], f_n]))
		estimated_difference_total_recall[key_column] = estimated_possible_total_recall[
															key_column] - estimated_total_recall

	return estimated_difference_total_recall


def select_by_estimated_max_recall_impact(tensor_run, column_states, current_f_list, col_list, estimated_scores, steps, use_sum=True):
	for s in range(steps):
		estimated_impact = calculateEstimatedRecallImpacts(column_states, estimated_scores)

		chosen_id = np.argmax(estimated_impact)

		column_states[chosen_id] += 1
		col_list.append(chosen_id)
		current_f_list.append(calc_total_f1(tensor_run, column_states))

	return current_f_list, col_list
