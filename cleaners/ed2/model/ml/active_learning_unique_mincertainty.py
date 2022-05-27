import pickle

from ml.active_learning.library import *
import xgboost as xgb
from sklearn.metrics import confusion_matrix


# best version
def go_to_next_column_prob1(column_id, pred_potential):

	minimum_pred = 0.0
	for column_step in range(len(pred_potential)):
		if pred_potential[column_step] != -1.0:
			if pred_potential[column_step] < minimum_pred:
				minimum_pred = pred_potential[column_step]

	new_potential = pred_potential - minimum_pred

	for column_step in range(len(pred_potential)):
		if pred_potential[column_step] == -1.0:
			new_potential[column_step] = 0.0


	# print str(new_potential)
	# print str(np.sum(new_potential))

	new_potential = np.square(new_potential)
	new_potential = new_potential / np.sum(new_potential)

	print "pot: " + str(new_potential) + " sum: " + str(np.sum(new_potential))

	# return np.random.choice(len(new_potential), 1, p=new_potential)[0]
	return np.argmax(new_potential)


def go_to_next_column_prob(id_next, avg_certainty):
	import operator
	return min(avg_certainty.iteritems(), key=operator.itemgetter(1))[0]




def go_to_next_column_round(column_id):
	column_id = column_id + 1
	if column_id == dataSet.shape[1]:
		column_id = 0
	return column_id


def load_model(dataSet, classifier):
	dataset_log_files = {}
	dataset_log_files[HospitalHoloClean().name] = "hospital"
	dataset_log_files[BlackOakDataSetUppercase().name] = "blackoak"
	dataset_log_files[FlightHoloClean().name] = "flight"
	# not yet
	dataset_log_files[Salary().name] = "hospital"  # be careful
	dataset_log_files[Book().name] = "hospital"  # be careful

	potential_model_dir = Config.get("column.potential.models")

	return pickle.load(
		open(potential_model_dir + "/model" + dataset_log_files[dataSet.name] + "_" + classifier.name + ".p"))


def add_lstm_features(data, use_lstm_only, all_matrix_train, feature_name_list):
	lstm_path = ""
	if dataSet.name == 'Flight HoloClean':
		lstm_path = "/home/felix/SequentialPatternErrorDetection/deepfeatures/Flights/last/"
	elif dataSet.name == 'HospitalHoloClean':
		lstm_path = "/home/felix/SequentialPatternErrorDetection/deepfeatures/HospitalHoloClean/last/"
	elif dataSet.name == 'BlackOakUppercase':
		lstm_path = "/home/felix/SequentialPatternErrorDetection/deepfeatures/BlackOakUppercase/last/"
	else:
		raise Exception('We have no potential model for this dataset yet')

	all_matrix_train_deep = read_compressed_deep_features(lstm_path)

	all_matrix_test = None
	feature_name_list_deep = ['deep ' + str(dfeature) for dfeature in range(all_matrix_train_deep.shape[1])]

	if use_lstm_only:
		all_matrix_train = all_matrix_train_deep
		feature_name_list = feature_name_list_deep
	else:
		all_matrix_train = hstack((all_matrix_train, all_matrix_train_deep)).tocsr()
		feature_name_list.extend(feature_name_list_deep)

	return all_matrix_train, all_matrix_test, feature_name_list


# input

start_time = time.time()

from ml.datasets.flights.FlightHoloClean import FlightHoloClean
#dataSet = FlightHoloClean()
from ml.datasets.hospital.HospitalHoloClean import HospitalHoloClean
#dataSet = HospitalHoloClean()
from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
dataSet = BlackOakDataSetUppercase()

from ml.datasets.salary_data.Salary import Salary
#dataSet = Salary()

from ml.datasets.luna.book.Book import Book
#dataSet = Book()

from ml.datasets.luna.restaurant.Restaurant import Restaurant

#dataSet = Restaurant()

'''
from ml.datasets.synthetic.Synthetic import Synthetic
from ml.datasets.synthetic.ReplaceError import ReplaceError
rows = 2000
datasets =[BlackOakDataSetUppercase(), BlackOakDataSetUppercase(), BlackOakDataSetUppercase(), BlackOakDataSetUppercase(), BlackOakDataSetUppercase(), BlackOakDataSetUppercase(), BlackOakDataSetUppercase(), BlackOakDataSetUppercase(), BlackOakDataSetUppercase(), BlackOakDataSetUppercase()]
columns = [4,4,4,4,4,4,4,4,4,4]
error_fraction = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
error_types = [ReplaceError, ReplaceError,ReplaceError, ReplaceError,ReplaceError, ReplaceError,ReplaceError, ReplaceError,ReplaceError, ReplaceError]
seed_synth = 41
dataSet = Synthetic(rows, datasets, columns, error_fraction, error_types, seed_synth)
'''

'''
from ml.datasets.synthetic.Synthetic import Synthetic
from ml.datasets.synthetic.ReplaceError import ReplaceError
rows = 2000
datasets =[BlackOakDataSetUppercase()]
columns = [4]
error_fraction = [0.9]
error_types = [ReplaceError]
seed_synth = 41
dataSet = Synthetic(rows, datasets, columns, error_fraction, error_types, seed_synth)
'''

print("read: %s seconds ---" % (time.time() - start_time))

start_time = time.time()

number_of_round_robin_rounds = 2

train_fraction = 1.0
ngrams = 1
runSVD = False
use_metadata = True
use_metadata_only = False
use_lstm = False
user_error_probability = 0.00
step_size = 10
cross_validation_rounds = 1  # 1

use_change_features = True

checkN = 10  # 5
# total runs
label_iterations = 6  # 6

run_round_robin = False
if run_round_robin:
	number_of_round_robin_rounds = 10000
	label_iterations = 41
	checkN = 10

feature_names_potential = ['distinct_values_fraction', 'labels', 'certainty', 'certainty_stddev', 'minimum_certainty']

for i in range(100):
	feature_names_potential.append('certainty_histogram' + str(i))

feature_names_potential.append('predicted_error_fraction')

for i in range(7):
	feature_names_potential.append('icross_val' + str(i))

feature_names_potential.append('mean_cross_val')
feature_names_potential.append('stddev_cross_val')

feature_names_potential.append('training_error_fraction')

for i in range(100):
	feature_names_potential.append('change_histogram' + str(i))

feature_names_potential.append('mean_squared_certainty_change')
feature_names_potential.append('stddev_squared_certainty_change')

for i in range(10):
	feature_names_potential.append('batch_certainty_' + str(i))

if use_change_features:
	feature_names_potential.append('no_change_0')
	feature_names_potential.append('no_change_1')
	feature_names_potential.append('change_0_to_1')
	feature_names_potential.append('change_1_to_0')

print(str(feature_names_potential))

size = len(feature_names_potential)
for s in range(size):
	feature_names_potential.append(feature_names_potential[s] + "_old")

which_features_to_use = []
for feature_index in range(len(feature_names_potential)):
	if True: #not 'histogram' in feature_names_potential[feature_index]:
		which_features_to_use.append(feature_index)
print which_features_to_use

feature_names_potential = [i for j, i in enumerate(feature_names_potential) if j in which_features_to_use]

feature_gen_time = 0.0

for check_this in range(checkN):

	f = open("/home/felix/ExampleDrivenErrorDetection/log_progress_" + dataSet.name + "_" + str(check_this) + ".csv",
			 'w+')

	train_indices, test_indices = split_data_indices(dataSet, train_fraction, fold_number=check_this)

	total_start_time = time.time()

	feature_gen_start = time.time()

	all_matrix_train, all_matrix_test, feature_name_list = create_features(dataSet, train_indices, test_indices, ngrams,
																		   runSVD)

	if use_metadata:
		all_matrix_train, all_matrix_test, feature_name_list = add_metadata_features(dataSet, train_indices,
																					 test_indices, all_matrix_train,
																					 all_matrix_test, feature_name_list,
																					 use_metadata_only)

	if use_lstm:
		all_matrix_train, all_matrix_test, feature_name_list = add_lstm_features(dataSet, False, all_matrix_train,
																				 feature_name_list)

	print("features: %s seconds ---" % (time.time() - start_time))

	data_result = []

	column_id = 0

	feature_matrix = all_matrix_train.tocsr()

	from ml.active_learning.classifier.XGBoostClassifier import XGBoostClassifier

	classifier = XGBoostClassifier(all_matrix_train, all_matrix_test)
	from ml.active_learning.classifier.LinearSVMClassifier import LinearSVMClassifier
	# classifier = LinearSVMClassifier(all_matrix_train, all_matrix_test)
	from ml.active_learning.classifier.NaiveBayesClassifier import NaiveBayesClassifier

	# classifier = NaiveBayesClassifier(all_matrix_train, all_matrix_test)

	all_error_status = np.zeros((all_matrix_train.shape[0], dataSet.shape[1]), dtype=bool)
	if all_matrix_test != None:
		all_error_status_test = np.zeros((all_matrix_test.shape[0], dataSet.shape[1]), dtype=bool)

	feature_gen_time = time.time() - feature_gen_start
	print("Feature Generation Time: " + str(feature_gen_time))

	save_fscore = []
	save_labels = []
	save_certainty = []
	save_fscore_general = []
	save_time = []

	our_params = {}
	train = {}
	train_target = {}

	train_chosen_ids = {}

	y_pred = {}
	certainty = {}
	min_certainty = {}
	final_gb = {}
	res = {}
	feature_array_all = {}

	zero_change_count = {}

	rounds_per_column = {}

	model = None

	pred_potential = {}

	y_next = {}
	x_next = {}
	id_next = {}
	diff_certainty = {}

	avg_certainty = {}



	for round in range(label_iterations * dataSet.shape[1]):
		print("round: " + str(round))

		if column_id in rounds_per_column:
			current_rounds = rounds_per_column[column_id]
			current_rounds += 1
			rounds_per_column[column_id] = current_rounds
		else:
			rounds_per_column[column_id] = 1

		# switch to column
		target_run, target_test = getTarget(dataSet, column_id, train_indices, test_indices)

		if rounds_per_column[column_id] == 1:
			start_time = time.time()

			num_errors = 2
			train[column_id], train_target[column_id], train_chosen_ids[column_id] = create_user_start_data(feature_matrix.tocsr(), target_run,
																			   num_errors, return_ids=True)
			if train[column_id] == None:
				certainty[column_id] = 1.0
				#pred_potential[column_id] = -1.0
				column_id = go_to_next_column_round(column_id)
				continue

			print("Number of errors in training: " + str(np.sum(train_target[column_id])))
			print("clustering: %s seconds ---" % (time.time() - start_time))

			# cross-validation
			start_time = time.time()
			classifier.run_cross_validation(train[column_id], train_target[column_id], num_errors, column_id)
			print("cv: %s seconds ---" % (time.time() - start_time))

			min_certainty[column_id] = 0.0

			eval_scores = np.zeros(7)

		else:
			if train[column_id] == None:
				if round < dataSet.shape[1] * number_of_round_robin_rounds:
					column_id = go_to_next_column_round(column_id)
				else:
					column_id = go_to_next_column_prob(id_next, avg_certainty)
				continue

			# change column

			if column_id in certainty:
				min_certainty[column_id] = np.min(np.absolute(y_pred[column_id] - 0.5))
			else:
				min_certainty[column_id] = 0.0

			diff = np.absolute(y_pred[column_id] - 0.5)
			print("min certainty: " + str(np.min(diff)))


			'''
			train[column_id], train_target[column_id], certainty[column_id], train_chosen_ids[column_id] = create_next_data(train[column_id],
																							   train_target[column_id],
																							   feature_matrix,
																							   target_run,
																							   y_pred[column_id],
																							   step_size,
																							   dataSet,
																							   column_id,
																							   user_error_probability,
																							   train_chosen_ids[column_id])


			'''

			train[column_id], train_target[column_id], train_chosen_ids[column_id] = add_data_next(
										      train[column_id],
											  train_target[column_id],
											  train_chosen_ids[column_id],
											  x_next[column_id],
											  y_next[column_id],
											  id_next[column_id])

			#print "len: " + str(len(train[column_id])) + " - " + str(len(train_target[column_id]))

			# cross-validation
			if round < dataSet.shape[1] * cross_validation_rounds:
				our_params[column_id] = classifier.run_cross_validation(train[column_id], train_target[column_id],
																		num_errors, column_id)
			# print("cv: %s seconds ---" % (time.time() - start_time))

			eval_scores = classifier.run_cross_validation_eval(train[column_id], train_target[column_id], 7, column_id)

		start_time = time.time()
		# train
		# predict
		y_pred_current_prediction, res_new = classifier.train_predict(train[column_id], train_target[column_id],
																	  column_id)

		if column_id in y_pred:
			prediction_change_y_pred = np.square(y_pred_current_prediction - y_pred[column_id])
		else:
			prediction_change_y_pred = np.zeros(len(y_pred_current_prediction))

		y_pred[column_id] = y_pred_current_prediction

		x_next[column_id], y_next[column_id], diff_certainty[column_id], id_next[column_id] = create_next_part(
			feature_matrix,
			target_run,
			y_pred[column_id],
			step_size,
			dataSet,
			column_id,
			user_error_probability,
			train_chosen_ids[column_id])

		print "size x: " + str(len(x_next[column_id]))


		if column_id in res:
			no_change_0, no_change_1, change_0_to_1, change_1_to_0 = compare_change(res[column_id], res_new)

			print("no change 0: " + str(no_change_0) + " no change 1: " + str(no_change_1) + " sum no change: " + str(
				no_change_0 + no_change_1))
			print("change 0 ->1: " + str(change_0_to_1) + " change 1->0: " + str(change_1_to_0) + " sum change: " + str(
				change_0_to_1 + change_1_to_0))
		else:
			no_change_0, no_change_1, change_0_to_1, change_1_to_0 = compare_change(np.zeros(len(res_new)), res_new)

		res[column_id] = res_new
		all_error_status[:, column_id] = res[column_id]
		print("train & predict: %s seconds ---" % (time.time() - start_time))

		if all_matrix_test != None:
			y_pred_test, res_gen = classifier.predict(column_id)

			all_error_status_test[:, column_id] = res_gen

		# visualize_model(dataSet, column_id, final_gb, feature_name_list, train, target_run, res)

		print ("current train shape: " + str(train[column_id].shape))

		print ("column: " + str(column_id))
		print_stats(target_run, res[column_id])
		print_stats_whole(dataSet.matrix_is_error[train_indices, :], all_error_status, "run all")
		calc_my_fscore(dataSet.matrix_is_error[train_indices, :], all_error_status, dataSet)
		if all_matrix_test != None:
			print_stats_whole(dataSet.matrix_is_error[test_indices, :], all_error_status_test, "test general")

		number_samples = 0
		for key, value in train.iteritems():
			if value != None:
				number_samples += value.shape[0]
		print("total labels: " + str(number_samples) + " in %: " + str(
			float(number_samples) / (dataSet.shape[0] * dataSet.shape[1])))

		sum_certainty = 0.0
		for key, value in certainty.iteritems():
			if value != None:
				sum_certainty += value
		sum_certainty /= dataSet.shape[1]
		print("total certainty: " + str(sum_certainty))

		save_fscore.append(f1_score(dataSet.matrix_is_error[train_indices, :].flatten(), all_error_status.flatten()))
		if all_matrix_test != None:
			save_fscore_general.append(
				f1_score(dataSet.matrix_is_error[test_indices, :].flatten(), all_error_status_test.flatten()))
		save_labels.append(number_samples)
		save_certainty.append(sum_certainty)

		num_hist_bin = 100
		diff = np.absolute(y_pred[column_id] - 0.5)
		certainty_here = (np.sum(diff) / len(diff)) * 2

		distinct_values_fraction = float(
			len(dataSet.dirty_pd[dataSet.dirty_pd.columns[column_id]].unique())) / float(dataSet.shape[0])

		feature_array = []
		feature_array.append(distinct_values_fraction)
		feature_array.append(train[column_id].shape[0])
		feature_array.append(certainty_here)
		avg_certainty[column_id] = certainty_here
		feature_array.append(np.std(diff))
		feature_array.append(np.min(np.absolute(y_pred[column_id] - 0.5)))

		for i in range(num_hist_bin):
			feature_array.append(float(len(
				diff[np.logical_and(diff >= i * (0.5 / num_hist_bin), diff < (i + 1) * (0.5 / num_hist_bin))])) / len(
				diff))

		predicted_error_fraction = float(np.sum(y_pred[column_id] > 0.5)) / float(len(y_pred[column_id]))
		print "predicted error fraction: " + str(predicted_error_fraction)
		feature_array.append(predicted_error_fraction)

		for score in eval_scores:
			feature_array.append(score)

		feature_array.append(np.mean(eval_scores))
		feature_array.append(np.std(eval_scores))

		training_error_fraction = float(np.sum(train_target[column_id])) / float(len(train_target[column_id]))
		print "training error fraction: " + str(training_error_fraction)
		feature_array.append(training_error_fraction)

		hist_pred_change = []
		for histogram_i in range(num_hist_bin):
			feature_array.append(float(len(prediction_change_y_pred[np.logical_and(
				prediction_change_y_pred >= histogram_i * (1.0 / num_hist_bin),
				prediction_change_y_pred < (histogram_i + 1) * (1.0 / num_hist_bin))])) / len(prediction_change_y_pred))

			hist_pred_change.append(float(len(prediction_change_y_pred[np.logical_and(
				prediction_change_y_pred >= histogram_i * (1.0 / num_hist_bin),
				prediction_change_y_pred < (histogram_i + 1) * (1.0 / num_hist_bin))])) / len(prediction_change_y_pred))

		feature_array.append(np.mean(prediction_change_y_pred))
		feature_array.append(np.std(prediction_change_y_pred))
		print "Mean Squared certainty change: " + str(np.mean(prediction_change_y_pred))

		batch_certainties = diff_certainty[column_id][id_next[column_id]]
		assert len(batch_certainties) == 10
		for batch_certainty in batch_certainties:
			feature_array.append(batch_certainty)


		# print "hist: pred: " + str(hist_pred_change)
		# plt.bar(range(100), hist_pred_change)
		# plt.show()

		if use_change_features:
			feature_array.append(no_change_0)
			feature_array.append(no_change_1)
			feature_array.append(change_0_to_1)
			feature_array.append(change_1_to_0)

		feature_vector = []

		if column_id in feature_array_all:
			if not run_round_robin:
				column_list = feature_array_all[column_id]
				column_list.append(feature_array)
				feature_array_all[column_id] = column_list

				feature_vector.extend(feature_array)
				feature_vector.extend(column_list[len(column_list) - 2])

				feature_vector_new = np.matrix(feature_vector)[0, which_features_to_use]

				'''
				if model == None:
					model = load_model(dataSet, classifier)

				mat_potential = xgb.DMatrix(feature_vector_new, feature_names=feature_names_potential)
				pred_potential[column_id] = model.predict(mat_potential)
				print("prediction: " + str(pred_potential[column_id]))
				'''

		else:
			column_list = []
			column_list.append(feature_array)
			feature_array_all[column_id] = column_list

		for feature_e in feature_array:
			f.write(str(feature_e) + ",")

		tn, fp, fn, tp = confusion_matrix(target_run, res[column_id]).ravel()

		# tn = float(tn) / float(len(target_run))
		fp = float(fp) / float(len(target_run))
		fn = float(fn) / float(len(target_run))
		tp = float(tp) / float(len(target_run))

		f.write(str(f1_score(target_run, res[column_id])) + "," + str(fp) + "," + str(fn) + "," + str(tp) + '\n')

		if round < dataSet.shape[1] * number_of_round_robin_rounds:
			column_id = go_to_next_column_round(column_id)
		else:
			print ("start using prediction")
			column_id = go_to_next_column_prob(id_next, avg_certainty)

		current_runtime = (time.time() - total_start_time)
		print("iteration end: %s seconds ---" % current_runtime)
		save_time.append(current_runtime)

	print (save_fscore)
	print (save_fscore_general)
	print (save_labels)
	print (save_certainty)
	print (save_time)
	f.close()
