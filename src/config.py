import torch
import os

class Config:
    # We will set these dynamically via argparse
    TRAIN_IMG_PATH = ""
    OUTPUT_DIR = "output"
    CHECKPOINT_PATH = "" # Will be set to OUTPUT_DIR/best_diffusion_checkpoint.pth

    # Data
    SLICE_AXIS = -1
    PADDING_SIZE = (320, 272)
    CROP_SIZE = (320, 260)

    # Model
    IN_CHANNELS = 3
    OUT_CHANNELS = 1
    CHANNELS = (32, 64, 128, 256)
    ATTENTION_LEVELS = (False, False, False, True)
    NUM_RES_BLOCKS = 2
    NUM_HEAD_CHANNELS = 32

    # Training
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 1
    ACCUMULATION_STEPS = 4
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_TRAIN_TIMESTEPS = 1000
    BETA_START = 0.0015
    BETA_END = 0.0195

    # Inference
    NUM_INFERENCE_STEPS = 50