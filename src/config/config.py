from pathlib import Path

class Config:
    # Project paths
    ROOT_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = ROOT_DIR / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    MODELS_DIR = ROOT_DIR / 'models' / 'saved'

    # API settings
    API_HOST = '0.0.0.0'
    API_PORT = 8000
    API_WORKERS = 4

    # Model settings
    MODEL_NAME = 'incident-classifier-v1'
    RANDOM_STATE = 42

