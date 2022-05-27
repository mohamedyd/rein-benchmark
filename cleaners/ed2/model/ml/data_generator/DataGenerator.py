import numpy as np
from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
from ml.datasets.flights.FlightHoloClean import FlightHoloClean
from ml.datasets.products.Products import Products
from ml.datasets.luna.book.Book import Book
from ml.datasets.electronics.Electronics import Electronics
from ml.datasets.salary_data.Salary import Salary
import pandas as pd
import csv
from ml.data_generator.generate_bart_config import generate_bart_config
from shutil import copyfile



datasets = [BlackOakDataSetUppercase().clean_pd.values,
			FlightHoloClean().clean_pd.values,
		    Salary().clean_pd.values,
			Electronics().clean_pd.values,
			Book().clean_pd.values,
			Products().clean_pd.values
			]


for n in range(1000):
	# select dataset
	dataset_id = np.random.randint(len(datasets))
	dataset = datasets[dataset_id]


	# select number of rows
	max_rows = 2000
	if datasets[dataset_id].shape[0] < max_rows:
		max_rows = datasets[dataset_id].shape[0]

	row_size = np.random.randint(low=500, high=max_rows)

	arr = np.arange(dataset.shape[0])
	np.random.shuffle(arr)
	sized_data = dataset[arr[0:row_size], :]

	print sized_data.shape

	# select which attributes to choose
	attribute_size = np.random.randint(low=2, high=dataset.shape[1])
	arr = np.arange(dataset.shape[1])
	np.random.shuffle(arr)

	attributes_selected = sized_data[:, arr[0:attribute_size]]

	column_all = np.array(['a','b','c','d','e','f','g','h','i','j','k', 'l', 'm','n', 'o', 'p', 'q', 'r', 's', 't', 'u'])

	header = column_all[arr[0:attribute_size]]

	data = pd.DataFrame(data=attributes_selected, columns=header)
	data.to_csv('/tmp/generated_data.csv', index=False, quoting=csv.QUOTE_ALL)
	data.to_csv('/tmp/my_data/clean_data' + str(n) +'.csv', index=False, quoting=csv.QUOTE_ALL)


	# select whether attribute is dirty
	number_dirty = np.random.randint(low=1, high=attributes_selected.shape[1])
	arr = np.arange(attributes_selected.shape[1])
	np.random.shuffle(arr)
	dirty_column_ids = arr[0:number_dirty]


	#select error type and error fraction
	# first only different random errors
	error_strategies = np.random.choice(5, len(dirty_column_ids))
	error_fraction = np.random.uniform(low=1.0 / row_size, high=1.0, size=len(dirty_column_ids))

	print header

	generate_bart_config(header, dirty_column_ids, error_fraction, error_strategies)

	copyfile('/tmp/dirty_person.csv', '/tmp/my_data/dirty_data' + str(n)+ '.csv')
