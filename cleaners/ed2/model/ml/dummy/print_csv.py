from ml.datasets.hospital.HospitalHoloClean import HospitalHoloClean

data = HospitalHoloClean()

data.dirty_pd.to_csv('/tmp/hospital.csv', index=False)