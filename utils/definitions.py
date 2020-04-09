import os

# paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_DIR, 'data')

MODEL_SAVE_DIR = os.path.join(PROJECT_DIR, 'models')
FIGURES_SAVE_DIR = os.path.join(PROJECT_DIR, 'reports', 'figures')

# API keys
COMET_API_KEY = "HIZapbzNjFips0c32Co7gXkQZ"
COMET_PROJECT_NAME= "richgans"
COMET_WORKSPACE = "maximartemev"

# create all dirs on import
# if __name__ == '__main__':
os.makedirs(os.path.join(PROCESSED_RAW_DATA_PATH, 'scalers'), exist_ok=True)
os.makedirs(FIGURES_SAVE_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

