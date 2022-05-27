import pickle

from ml.active_learning.library import *
import sys
from sklearn.metrics import confusion_matrix
from ml.Word2VecFeatures.Word2VecFeatures import Word2VecFeatures
from ml.features.ActiveCleanFeatures import ActiveCleanFeatures
from ml.features.ValueCorrelationFeatures import ValueCorrelationFeatures
from ml.features.BoostCleanMetaFeatures import BoostCleanMetaFeatures
import operator

def go_to_next_column_prob(diff_certainty):
	certainty_columns ={}

	for key in diff_certainty.keys():
		certainty_columns[key] = (np.sum(diff_certainty[key]) / len(diff_certainty[key])) * 2

	return min(certainty_columns.items(), key=operator.itemgetter(1))[0]




def go_to_next_column_round(column_id, dataSet):
	column_id = column_id + 1
	if column_id == dataSet.shape[1]:
		column_id = 0
	return column_id

def go_to_next_column_random(dataSet):
	my_list = dataSet.get_applicable_columns()
	id = np.random.randint(len(my_list))
	return my_list[id]

def go_to_next_column(dataSet, statistics,
					  use_max_pred_change_column_selection,
					  use_max_error_column_selection,
					  use_min_certainty_column_selection,
					  use_random_column_selection):
	if use_min_certainty_column_selection:
		return go_to_next_column_prob(statistics['certainty'])
	if use_max_pred_change_column_selection:
		return max(statistics['change'].items(), key=operator.itemgetter(1))[0]
	if use_max_error_column_selection:
		return min(statistics['cross_val_f'].items(), key=operator.itemgetter(1))[0]
	if use_random_column_selection:
		return go_to_next_column_random(dataSet)


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
	if data.name == 'Flight HoloClean':
		lstm_path = Config.get('lstm.folder') + "/Flights/last/"
	elif data.name == 'HospitalHoloClean':
		lstm_path = Config.get('lstm.folder') + "/HospitalHoloClean/last/"
	elif data.name == 'BlackOakUppercase':
		lstm_path = Config.get('lstm.folder') + "/BlackOakUppercase/last/"
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


def augment_features_with_predictions(data_x_matrix, all_matrix_train, current_predictions, column_id, train_chosen_ids):
	x_all = all_matrix_train.copy()
	for column_number, last_predictions in current_predictions.items():
		if column_number != column_id:
			select_predictions = np.matrix(last_predictions).transpose()
			try:
				data_x_matrix = hstack((data_x_matrix, select_predictions[train_chosen_ids[column_id], :])).tocsr()
				x_all = hstack((x_all, select_predictions)).tocsr()
			except:
				data_x_matrix = np.hstack((data_x_matrix, select_predictions[train_chosen_ids[column_id], :]))
				x_all= np.hstack((x_all, select_predictions))
	return data_x_matrix, x_all


def augment_features_with_predictions_test(all_matrix_test, current_predictions_test, column_id):
	x_all = all_matrix_test.copy()
	for column_number, last_predictions in current_predictions_test.items():
		if column_number != column_id:
			select_predictions = np.matrix(last_predictions).transpose()
			try:
				x_all = hstack((x_all, select_predictions)).tocsr()
			except:
				x_all= np.hstack((x_all, select_predictions))
	return x_all



def run_multi( params):
	try:
		return run(**params)
	except:
		return_dict = {}
		return_dict['labels'] = []
		return_dict['fscore'] = []
		return_dict['precision'] = []
		return_dict['recall'] = []
		return_dict['time'] = []
		return_dict['error'] = "Unexpected error:" + str(sys.exc_info()[0])

		return return_dict

def run(dataSet,
			 classifier_model,
			 number_of_round_robin_rounds=2,
			 train_fraction=1.0,
			 ngrams=1,
			 runSVD=False,
			 is_word=False,
			 use_metadata = True,
			 use_metadata_only = False,
			 use_lstm=False,
             use_lstm_only=False,
			 user_error_probability=0.00,
			 step_size=10,
			 cross_validation_rounds=1,
			 checkN=1,#originally 10
			 label_iterations=100, # algorithm termination is mainly by labeling_threshold determined. 
			label_threshold = 200,
			 run_round_robin=False,
			 correlationFeatures=True,
			 use_tf_idf=True,
			 use_word2vec=False,
			 use_word2vec_only=False,
			 w2v_size=100,
			 use_active_clean=False,
			 use_activeclean_only=False,
			 use_cond_prob=False,
			 use_cond_prob_only=False,
		     use_boostclean_metadata=False,
		     use_boostclean_metadata_only=False,
             store_results=False,
		     use_random_column_selection=False,
		     use_max_pred_change_column_selection=False,
             use_max_error_column_selection=False,
			 use_min_certainty_column_selection=True,
             visualize_models=False,
		     output_detection_result=0,
			 ):
    
	start_time = time.time()

	all_fscore = []
	all_precision = []
	all_recall = []

	all_fscore_test = []
	all_precision_test = []
	all_recall_test = []


	all_time = []

	use_change_features = True

	if run_round_robin:
		number_of_round_robin_rounds = 10000
		label_iterations = 71
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
	print(which_features_to_use)

	feature_names_potential = [i for j, i in enumerate(feature_names_potential) if j in which_features_to_use]

	feature_gen_time = 0.0

	for check_this in range(checkN):

		ts = time.time()
		#f = open(Config.get("logging.folder") + "/logging_output/label_log_progress_" + dataSet.name + "_" + str(check_this) + "_" +  str(ts) + ".csv",
		#		 'w+')

		train_indices, test_indices = split_data_indices(dataSet, train_fraction, fold_number=check_this)

		total_start_time = time.time()

		feature_gen_start = time.time()

		all_matrix_train, all_matrix_test, feature_name_list = create_features(dataSet, train_indices, test_indices, ngrams,
																			   runSVD, is_word, use_tf_idf)

		if use_metadata:
			all_matrix_train, all_matrix_test, feature_name_list = add_metadata_features(dataSet, train_indices,
																						 test_indices, all_matrix_train,
																						 all_matrix_test, feature_name_list,
																						 use_metadata_only)

		if use_lstm:
			all_matrix_train, all_matrix_test, feature_name_list = add_lstm_features(dataSet, use_lstm_only, all_matrix_train,
																					 feature_name_list)


		if use_word2vec:
			w2v_features = Word2VecFeatures(vector_size=w2v_size)
			all_matrix_train, all_matrix_test, feature_name_list = w2v_features.add_word2vec_features(dataSet, train_indices,
																						 test_indices,
																						 all_matrix_train,
																						 all_matrix_test,
																						 feature_name_list,
																						 use_word2vec_only)


		if use_active_clean:
			ac_features = ActiveCleanFeatures()  # active clean
			all_matrix_train, all_matrix_test, feature_name_list = ac_features.add_features(dataSet,
																									  train_indices,
																									  test_indices,
																									  all_matrix_train,
																									  all_matrix_test,
																									  feature_name_list,
																									  use_activeclean_only)

		if use_cond_prob:
			ac_features = ValueCorrelationFeatures()
			all_matrix_train, all_matrix_test, feature_name_list = ac_features.add_features(dataSet,
																							train_indices,
																							test_indices,
																							all_matrix_train,
																							all_matrix_test,
																							feature_name_list,
																							use_cond_prob_only)


		if use_boostclean_metadata:
			ac_features = BoostCleanMetaFeatures()  # boost clean metadatda
			all_matrix_train, all_matrix_test, feature_name_list = ac_features.add_features(dataSet,
																									  train_indices,
																									  test_indices,
																									  all_matrix_train,
																									  all_matrix_test,
																									  feature_name_list,
																							          use_boostclean_metadata_only)

		print("features: %s seconds ---" % (time.time() - start_time))

		column_id = 0

		try:
			feature_matrix = all_matrix_train.tocsr()
		except:
			feature_matrix = all_matrix_train

		classifier = classifier_model(all_matrix_train, all_matrix_test)

		all_error_status = np.zeros((all_matrix_train.shape[0], dataSet.shape[1]), dtype=bool)
		if all_matrix_test != None:
			all_error_status_test = np.zeros((all_matrix_test.shape[0], dataSet.shape[1]), dtype=bool)

		feature_gen_time = time.time() - feature_gen_start
		print("Feature Generation Time: " + str(feature_gen_time))

		save_fscore = []
		save_precision = []
		save_recall = []

		save_fscore_test = []
		save_precision_test = []
		save_recall_test = []


		save_labels = []
		save_certainty = []
		save_time = []

		our_params = {}
		train = {}
		train_target = {}

		train_chosen_ids = {}

		y_pred = {}
		certainty = {}
		min_certainty = {}
		res = {}
		feature_array_all = {}

		rounds_per_column = {}

		y_next = {}
		x_next = {}
		id_next = {}
		diff_certainty = {}

		current_predictions = {}
		current_predictions_test = {}


		statistics = {}
		statistics['change'] = {}
		statistics['certainty'] = {}
		statistics['cross_val_f'] = {}




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

			# user has to label initial sample for every column
			if rounds_per_column[column_id] == 1:
				start_time = time.time()

				num_errors = 2
				train[column_id], train_target[column_id], train_chosen_ids[column_id] = create_user_start_data(feature_matrix, target_run,
																				   num_errors, return_ids=True)

	
				if type(train[column_id]) == type(None):
					certainty[column_id] = 1.0
					#pred_potential[column_id] = -1.0

					print("column " + str(column_id) + " is not applicable" + " errors: " + str(np.sum(dataSet.matrix_is_error[:,column_id])))

					all_error_status[:, column_id] = dataSet.matrix_is_error[train_indices, column_id]

					if use_random_column_selection:
						column_id = go_to_next_column_random(dataSet)
					else:
						column_id = go_to_next_column_round(column_id, dataSet)
					continue

				print("Number of errors in training: " + str(np.sum(train_target[column_id])))
				print("clustering: %s seconds ---" % (time.time() - start_time))

				# cross-validation
				start_time = time.time()
				classifier.run_cross_validation(train[column_id], train_target[column_id], num_errors, column_id)
				print("cv: %s seconds ---" % (time.time() - start_time))

				min_certainty[column_id] = 0.0

				eval_scores = np.zeros(7)

				data_x_matrix = train[column_id].copy()
				x_all = all_matrix_train.copy()
				if type(all_matrix_test) != type(None):
					x_all_test = all_matrix_test.copy()
				else:
					x_all_test = None

			else:
				if type(train[column_id]) == type(None):
					if round < dataSet.shape[1] * number_of_round_robin_rounds:
						if use_random_column_selection:
							column_id = go_to_next_column_random(dataSet)
						else:
							column_id = go_to_next_column_round(column_id, dataSet)
					else:
						column_id = go_to_next_column(dataSet, statistics, use_max_pred_change_column_selection, use_max_error_column_selection, use_min_certainty_column_selection, use_random_column_selection)
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


				print("id next: " + str(len(id_next[column_id])))

				train[column_id], train_target[column_id], train_chosen_ids[column_id] = add_data_next(
												  train[column_id],
												  train_target[column_id],
												  train_chosen_ids[column_id],
												  x_next[column_id],
												  y_next[column_id],
												  id_next[column_id])


				print(train_chosen_ids[column_id])
				print("train: " + str(train[column_id].shape))


				##todo: the same for test
				data_x_matrix = train[column_id].copy()
				x_all = all_matrix_train
				x_all_test = all_matrix_test
				if correlationFeatures:
					data_x_matrix, x_all = augment_features_with_predictions(data_x_matrix, all_matrix_train, current_predictions, column_id, train_chosen_ids)
					if type(None) != type(all_matrix_test):
						x_all_test = augment_features_with_predictions_test(all_matrix_test, current_predictions_test, column_id)


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

			#todo check

			#y_pred_current_prediction, res_new = classifier.train_predict_all(data_x_matrix, train_target[column_id], column_id, x_all,
			#																  feature_name_list, dataSet.clean_pd.columns)

			y_pred_current_prediction, res_new, y_pred_current_prediction_test, res_new_test = classifier.train_predict_all(data_x_matrix,
																			  train_target[column_id], column_id,
																			  x_all, x_all_test)




			current_predictions[column_id] = y_pred_current_prediction
			current_predictions_test[column_id] = y_pred_current_prediction_test

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
				train_chosen_ids[column_id],
				check_this)

			print("size x: " + str(len(x_next[column_id])))


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

			print("resnew shape: " + str(res_new.shape) + " - allerror status" + str(all_error_status.shape))

			print("train & predict: %s seconds ---" % (time.time() - start_time))



			if all_matrix_test != None:
				all_error_status_test[:, column_id] = res_new_test

			if visualize_models:
				visualize_model(dataSet, column_id, classifier.model, feature_name_list, train, target_run, res)

			print ("current train shape: " + str(train[column_id].shape))

			print ("column: " + str(column_id))
			"""
			print(stats(target_run, res[column_id]))
			print(stats_whole(dataSet.matrix_is_error[train_indices, :], all_error_status, "run all"))
			calc_my_fscore(dataSet.matrix_is_error[train_indices, :], all_error_status, dataSet)
			if all_matrix_test != None:
				print_stats_whole(dataSet.matrix_is_error[test_indices, :], all_error_status_test, "test general")
			"""
			number_samples = 0
			for key, value in train.items():
				if type(value) != type(None):
					number_samples += value.shape[0]
			print("total labels: " + str(number_samples) + " in %: " + str(
				float(number_samples) / (dataSet.shape[0] * dataSet.shape[1])))

			sum_certainty = 0.0
			for key, value in certainty.items():
				if value != None:
					sum_certainty += value
			sum_certainty /= dataSet.shape[1]
			print("total certainty: " + str(sum_certainty))

			save_fscore.append(f1_score(dataSet.matrix_is_error[train_indices, :].flatten(), all_error_status.flatten()))
			save_precision.append(precision_score(dataSet.matrix_is_error[train_indices, :].flatten(), all_error_status.flatten()))
			save_recall.append(recall_score(dataSet.matrix_is_error[train_indices, :].flatten(), all_error_status.flatten()))

			#save_fscore_test.append(
			#	f1_score(dataSet.matrix_is_error[test_indices, :].flatten(), all_error_status_test.flatten()))
			#save_precision_test.append(
			#	precision_score(dataSet.matrix_is_error[test_indices, :].flatten(), all_error_status_test.flatten()))
			#save_recall_test.append(
			#	recall_score(dataSet.matrix_is_error[test_indices, :].flatten(), all_error_status_test.flatten()))



			if output_detection_result>0 and output_detection_result == number_samples:
				np.save('/tmp/ed2_results.npy', all_error_status)

			if store_results:
				np.save(Config.get("logging.folder") + "/results/result" + dataSet.name + "_" + str(check_this) + "_" +  str(ts), all_error_status)


###
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
			feature_array.append(np.std(diff))
			feature_array.append(np.min(np.absolute(y_pred[column_id] - 0.5)))

			for i in range(num_hist_bin):
				feature_array.append(float(len(
					diff[np.logical_and(diff >= i * (0.5 / num_hist_bin), diff < (i + 1) * (0.5 / num_hist_bin))])) / len(
					diff))

			predicted_error_fraction = float(np.sum(y_pred[column_id] > 0.5)) / float(len(y_pred[column_id]))
			print("predicted error fraction: " + str(predicted_error_fraction))
			feature_array.append(predicted_error_fraction)

			for score in eval_scores:
				feature_array.append(score)

			feature_array.append(np.mean(eval_scores))
			feature_array.append(np.std(eval_scores))

			training_error_fraction = float(np.sum(train_target[column_id])) / float(len(train_target[column_id]))
			print("training error fraction: " + str(training_error_fraction))
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
			print("Mean Squared certainty change: " + str(np.mean(prediction_change_y_pred)))

			batch_certainties = diff_certainty[column_id][id_next[column_id]]
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


			statistics['change'][column_id] = change_0_to_1 + change_1_to_0
			statistics['certainty'][column_id] = diff_certainty[column_id]
			statistics['cross_val_f'][column_id] = np.mean(eval_scores)

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

			"""
			for feature_e in feature_array:
				f.write(str(feature_e) + ",")
			"""

			tn, fp, fn, tp = confusion_matrix(target_run, res[column_id]).ravel()

			# tn = float(tn) / float(len(target_run))
			fp = float(fp) / float(len(target_run))
			fn = float(fn) / float(len(target_run))
			tp = float(tp) / float(len(target_run))

			#f.write(str(f1_score(target_run, res[column_id])) + "," + str(fp) + "," + str(fn) + "," + str(tp) + '\n')

			if round < dataSet.shape[1] * number_of_round_robin_rounds:
				if use_random_column_selection:
					column_id = go_to_next_column_random(dataSet)
				else:
					column_id = go_to_next_column_round(column_id, dataSet)
			else:
				print ("start using prediction")
				column_id = go_to_next_column(dataSet, statistics, use_max_pred_change_column_selection, use_max_error_column_selection, use_min_certainty_column_selection, use_random_column_selection)

			current_runtime = (time.time() - total_start_time)
			print("iteration end: %s seconds ---" % current_runtime)
			save_time.append(current_runtime)
			print("all_errors_states", all_error_status)
   
			# if the threshold of labels we want is passed break the loop
			# with step_size = 10, the #labels can be at maximum 9 higher than the threshold
			if number_samples >= label_threshold:
				break

		print ("save fscore: ", save_fscore)
		print ("save labels: ", save_labels)
		print (save_certainty)
		print (save_time)
		#f.close()

		all_fscore.append(save_fscore)
		all_precision.append(save_precision)
		all_recall.append(save_recall)

		all_fscore_test.append(save_fscore_test)
		all_precision_test.append(save_precision_test)
		all_recall_test.append(save_recall_test)

		all_time.append(save_time)

	##################################################################
	return_dict = {}
	return_dict['labels'] = save_labels

	return_dict['fscore'] = all_fscore
	return_dict['precision'] = all_precision
	return_dict['recall'] = all_recall

	return_dict['fscore_test'] = all_fscore_test
	return_dict['precision_test'] = all_precision_test
	return_dict['recall_test'] = all_recall_test

	return_dict['time'] = all_time
	return_dict['all_error_status'] = all_error_status

	return return_dict, all_error_status
