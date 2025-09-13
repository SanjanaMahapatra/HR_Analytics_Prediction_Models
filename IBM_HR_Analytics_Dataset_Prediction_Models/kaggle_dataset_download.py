
import kaggle

kaggle.api.authenticate()

kaggle.api.dataset_download_files('pavansubhasht/ibm-hr-analytics-attrition-dataset', path='dataset/', unzip=True)