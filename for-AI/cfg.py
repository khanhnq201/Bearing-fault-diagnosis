import torch
#--- Data loading ---
BASE_PATH = 'CWRU-dataset-main'
SAMPLE_LENGTH = 2048
PREPROCESSING = False
OVERLAPPING_RATIO = 0
RANDOM_STATE = 42

#--- Model parameter --- 
CNN1D_INPUT = True
MODEL_TYPE = '1D' #student/ teacher/ 1D
NUM_CLASSES = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 400
LEARNING_RATE = 0.000005
BATCH_SIZE = 128

CLASS_NAMES = ['Normal','IR', 'OR'] 
LABEL_MAP = {
    'Normal': 0,
    'IR': 2,
    'OR': 3
}

