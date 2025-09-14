
import kaggle

kaggle.api.authenticate()

# downloading the employee promotion dataset from kaggle

kaggle.api.dataset_download_files('muhammadimran112233/employees-evaluation-for-promotion', path='dataset/', unzip=True)

# downloading the employee performance for hr analytics dataset from kaggle

kaggle.api.dataset_download_files('sanjanchaudhari/employees-performance-for-hr-analytics', path='dataset/', unzip=True)


















