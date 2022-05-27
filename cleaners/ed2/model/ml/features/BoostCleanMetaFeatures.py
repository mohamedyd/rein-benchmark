import numpy as np
from scipy.sparse import hstack
import dateutil
import usaddress

class BoostCleanMetaFeatures():
    def __init__(self):
        self.method_list = [
            self.toNumber,  # Numerical Attributes
            self.has_alphanumeric_characters, self.is_empty_string, self.is_NAN, self.is_INF,  # missing values
            # Parsing/Type Errors:
            self.hasDay, self.hasMonth, self.hasYear,  # date
            self.hasStreet, self.hasCity, self.hasState  # address
        ]
        pass

    def add_features(self, dataSet,
                              train_indices, test_indices,
                              all_matrix_train, all_matrix_test,
                              feature_name_list,
                              features_only):

        data_train = dataSet.dirty_pd.values[train_indices, :]
        data_test = dataSet.dirty_pd.values[test_indices, :]

        self.fit(data_train)

        features_train = self.transform(data_train)
        names = self.get_feature_names(dataSet)

        if features_only:
            all_features_train_new = features_train
            feature_name_list = names
            all_features_test_new = all_matrix_test

        else:
            try:
                all_features_train_new = hstack((all_matrix_train, features_train)).tocsr()
            except:
                all_features_train_new = np.hstack((all_matrix_train, features_train))
            feature_name_list.extend(names)

            if data_test.shape[0] > 0:
                features_test = self.transform(data_test)
                all_features_test_new = hstack((all_matrix_test, features_test)).tocsr()
            else:
                all_features_test_new = all_matrix_test

        return all_features_train_new, all_features_test_new, feature_name_list


    def get_feature_names(self, dataSet):
        feature_names = []
        for col_i in range(dataSet.shape[1]):
            for m in self.method_list:
                feature_names.append(str(dataSet.clean_pd.columns[col_i]) + '_' + str(m.__name__))
        return feature_names


    def fit(self, data):
        pass

    # Numeric
    def toNumber(self, value):
        try:
            return float(value)
        except:
            return 0.0

    #Parsing/Type Errors
    #date
    def hasMonth(self, value):
        try:
            return dateutil.parser.parse(str(value)).month
        except:
            return -1

    def hasDay(self, value):
        try:
            return dateutil.parser.parse(str(value)).day
        except:
            return -1

    def hasYear(self, value):
        try:
            return dateutil.parser.parse(str(value)).year
        except:
            return -1

    #address
    def hasStreet(self, value):
        try:
            address = usaddress.parse(str(value))
            for tuple in address:
                if tuple[1] == 'StreetName':
                    return True
        except:
            return False
        return False

    def hasCity(self, value):
        try:
            address = usaddress.parse(str(value))
            for tuple in address:
                if tuple[1] == 'PlaceName':
                    return True
        except:
            return False
        return False

    def hasState(self, value):
        try:
            address = usaddress.parse(str(value))
            for tuple in address:
                if tuple[1] == 'StateName':
                    return True
        except:
            return False
        return False



    '''
    empty string, NaN, Inf;
    or values whose string representations lack alphanumeric characters.
    '''
    def has_alphanumeric_characters(self, value):
        for i in value:
            if i.isalnum():
                return True
        return False

    def is_empty_string(self, value):
        try:
            return len(str(value)) == 0
        except UnicodeEncodeError:
            return len(value) == 0

    def is_NAN(self, value):
        try:
            return str(value).upper() == 'NAN'
        except UnicodeEncodeError:
            return value.upper() == 'NAN'

    def is_INF(self, value):
        try:
            return str(value).upper() == 'INF'
        except UnicodeEncodeError:
            return value.upper() == 'INF'

    def transform(self, data):
        result = np.zeros((data.shape[0], data.shape[1] * len(self.method_list) ))
        for row_i in range(data.shape[0]):
            f_counter = 0
            for col_i in range(data.shape[1]):
                for m in self.method_list:
                    result[row_i, f_counter] = m(data[row_i, col_i])
                    f_counter += 1

        return result


if __name__ == '__main__':
    from ml.datasets.flights.FlightHoloClean import FlightHoloClean

    data = FlightHoloClean()

    f = BoostCleanMetaFeatures()
    f.fit(data.dirty_pd.values)
    print(f.transform(data.dirty_pd.values))
    print(f.get_feature_names(data))
    print(len(f.get_feature_names(data)))