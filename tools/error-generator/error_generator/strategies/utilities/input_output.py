import pandas


class Read_Write():
    def __init__(self):
        pass

    def read_csv_dataset(dataset_path, header_exists=True):
        """
        The method reads a dataset from a csv file path.
        """

        dataset_dataframe_version = pandas.read_csv(dataset_path)
        if header_exists:
            dataset_dataframe = pandas.read_csv(dataset_path, sep=",", header="infer", encoding="utf-8", dtype=str,
                                                keep_default_na=False, low_memory=False)

            dataset_dataframe = dataset_dataframe.apply(lambda x: x.str.strip())
            return [list(dataset_dataframe.columns.to_numpy())] + list(dataset_dataframe.to_numpy()), dataset_dataframe_version
        else:
            dataset_dataframe = pandas.read_csv(dataset_path, sep=",", header=None, encoding="utf-8", dtype=str,
                                                keep_default_na=False)

            dataset_dataframe = dataset_dataframe.apply(lambda x: x.str.strip())
            return list(dataset_dataframe.values()),dataset_dataframe_version

    def write_csv_dataset(dataset_path, dataset_table):
        """
        The method writes a dataset to a csv file path.
        """
        dataset_dataframe = pandas.DataFrame(data=dataset_table[1:], columns=dataset_table[0])
        dataset_dataframe.to_csv(dataset_path, sep=",", header=True, index=False, encoding="utf-8")


