#config.py
import os

# 기본 설정
BASE_PATH = 'data'
TRAIN_PATH = os.path.join(BASE_PATH, 'train')
MODEL_SAVE_PATH = 'models'
LOG_PATH = 'logs'

# 학습 파라미터
BATCH_SIZE = 16
LEARNING_RATE = 5e-4
NUM_EPOCHS = 50
IMAGE_SIZE = 256
NUM_WORKERS = 4

# 모델 파라미터
LOSS_WEIGHTS = {
    'dice': 0.6,
    'focal': 0.3,
    'bce': 0.1
}