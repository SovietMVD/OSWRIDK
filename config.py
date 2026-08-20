import os

ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.path.join(ROOT, "data")
TRAIN_DIR = "train"
TEST_DIR = "test"
OOD_DIR = "ood"
WEIGHTS_DIR = os.path.join(ROOT, "weights")

PATCH_SIZE = 256
INPUT_SIZE = 224
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)

BACKBONE = "resnet50"
PRETRAINED = True

EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
PATIENCE = 2
DELTA = 0.05
VAL_FRAC = 0.2

PSI_RATIO = 1.0 / 3.0
T = 200

SEED = 42
DEVICE = "cuda"
