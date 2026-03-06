# src/config.py

RANDOM_SEED = 42

# =========================
# DATA SETTINGS
# =========================

DEV_MODE = False  # Switch to False for full dataset

if DEV_MODE:
    TRAIN_SIZE = 80000
    VAL_SIZE = 10000
    TEST_SIZE = 20000
else:
    TRAIN_SIZE = 240000
    VAL_SIZE = 30000
    TEST_SIZE = 60000


# =========================
# BERT SETTINGS
# =========================

MODEL_NAME = "bert-base-uncased"

MAX_LENGTH = 256
BATCH_SIZE = 8  # Safe for 8GB GPU
LEARNING_RATE = 2e-5
EPOCHS = 2


# =========================
# OUTPUT
# =========================

MODEL_SAVE_PATH = "models/"

RUN_TRAINING = False # Set to True only when retraining BERT