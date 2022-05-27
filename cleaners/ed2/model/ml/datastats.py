import numpy as np

from ml.datasets.hospital import HospitalHoloClean
'''
#data = BlackOakDataSet()
data = HospitalHoloClean()
#data = FlightHoloClean()

#data = BlackOakDataSetUppercase()

#data.dirty_pd.to_csv('hosp.csv', index=False)
#data.dirty_pd.to_csv('flight.csv', index=False)

#data.clean_pd.to_csv('blackoak_clean.csv', index=False, sep=',', escapechar='\\', quotechar='\'', na_rep="NA")
#data.clean_pd.to_csv('hosp_clean.csv', index=False, sep=',', escapechar='\\', quotechar='\'', na_rep="NA")
#data.clean_pd.to_csv('flight_clean.csv', index=False, sep=',', escapechar='\\', quotechar='\'', na_rep="NA")

print len(data.clean_pd[data.clean_pd.columns[0]].unique())

print data.shape

print float(np.sum(data.matrix_is_error)) / float(data.shape[0] * data.shape[1])
'''

import csv
from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase

#data = BlackOakDataSetUppercase()

#data.clean_pd.columns =['RecID','FirstName','MiddleName','LastName','Address','City','State','ZIP','POBox','POCityStateZip','SSN','DOB']

#data.clean_pd.to_csv('/tmp/address.csv', index=False, quoting=csv.QUOTE_ALL)
#data.clean_pd.to_csv('/tmp/address.csv', index=False)


from ml.datasets.hospital.HospitalHoloClean import HospitalHoloClean
#data = HospitalHoloClean()


from ml.datasets.HospitalDomainError.HospitalDomainError import HospitalDomainError
#data = HospitalDomainError()
#data.dirty_pd.to_csv('/tmp/hospitaldomain.csv', index=False)

#print np.sum(data.matrix_is_error,axis=0)

import csv
from ml.datasets.RestaurantMohammad.Restaurant import Restaurant
#data = Restaurant()
#data.clean_pd.to_csv('/home/felix/Software/Metanome/deployment/target/test/backend/WEB-INF/classes/inputData/restaurant.csv', index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)#
#print data.shape


from ml.datasets.salary_data.Salary import Salary
#data = Salary()

#data.clean_pd.to_csv('/tmp/Salary_Felix_clean.csv', index=False, encoding='utf-8')
#data.dirty_pd.to_csv('/tmp/Salary_Felix_dirty.csv', index=False, encoding='utf-8')



from ml.datasets.MoviesMohammad.Movies import Movies
#data = Movies()


#print list(np.sum(data.matrix_is_error, axis=0) / float(data.shape[0]))


#data.clean_pd.to_csv('/home/felix/Software/Metanome/deployment/target/test/backend/WEB-INF/classes/inputData/movies.csv', index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
#print data.shape

from ml.datasets.Citations.Citation import Citation
#data = Citation()
#data.clean_pd.to_csv('/home/felix/Software/Metanome/deployment/target/test/backend/WEB-INF/classes/inputData/citation.csv', index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)




#from ml.datasets.BeerDataset.Beers import Beers
#data = Beers()
#data.clean_pd.to_csv('/home/felix/Software/Metanome/deployment/target/test/backend/WEB-INF/classes/inputData/beers.csv', index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
#print data.shape


from ml.datasets.BeersMohammad.Beers import Beers
data = Beers()


from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
#data = BlackOakDataSetUppercase()
#data.clean_pd.to_csv('/home/felix/Software/Metanome/deployment/target/test/backend/WEB-INF/classes/inputData/address.csv', index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)

#from ml.datasets.flights.FlightHoloClean import FlightHoloClean
#data = FlightHoloClean()
#data.clean_pd.to_csv('/home/felix/Software/Metanome/deployment/target/test/backend/WEB-INF/classes/inputData/flights.csv', index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)


from ml.datasets.hospital.HospitalHoloClean import HospitalHoloClean
#data = HospitalHoloClean()
#data.clean_pd.to_csv('/home/felix/Software/Metanome/deployment/target/test/backend/WEB-INF/classes/inputData/hospital.csv', index=False, quoting=csv.QUOTE_ALL)





per_column = np.sum(data.matrix_is_error, axis=0) / float(data.shape[0])





'''


from ml.datasets.salary_data.Salary import Salary

data = Salary()
all_columns = list(data.clean_pd.columns)
all_columns[0] = "mygreatid"
all_columns[8] += " (integer)"
data.clean_pd.columns = all_columns

def convert_to_int(value):
    return int(float(value))

data.clean_pd[all_columns[8]] = data.clean_pd[all_columns[8]].apply(convert_to_int)


print list(data.clean_pd.columns)

data.clean_pd.to_csv('/tmp/salary.csv', index=False, quoting=csv.QUOTE_ALL)


from ml.datasets.BartDataset.BartDataSet import BartDataset
data = BartDataset(BlackOakDataSetUppercase(), "CityFD_30percent")
data.clean_pd.to_csv('/tmp/salary.csv', index=False, quoting=csv.QUOTE_ALL)
'''


