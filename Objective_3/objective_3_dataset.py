
import kaggle

kaggle.api.authenticate()

# downloading the employee promotion dataset from kaggle

kaggle.api.dataset_download_files('mfaisalqureshi/hr-analytics-and-job-prediction', path='dataset/', unzip=True)



















