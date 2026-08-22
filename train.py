import logging
import torch
import os
import argparse
from torch.utils.data import DataLoader
from src.config import Config
from src.data.loader import extract_slices
from src.data.dataset import SliceInterpolationDataset
from src.models.components import build_unet, build_ddpm_scheduler
from src.training.trainer import train_model

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Train 3D Volume Interpolation Diffusion Model")
    parser.add_argument("--img_path", type=str, required=True, help="Path to training NIfTI image")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save checkpoints")
    args = parser.parse_args()

    # Initialize and dynamically update config
    cfg = Config()
    cfg.TRAIN_IMG_PATH = args.img_path
    cfg.OUTPUT_DIR = args.output_dir
    cfg.CHECKPOINT_PATH = os.path.join(cfg.OUTPUT_DIR, "best_diffusion_checkpoint.pth")

    # Create output directory if it doesn't exist
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # 1. Load Data
    all_slices, global_min, global_max = extract_slices(cfg.TRAIN_IMG_PATH, cfg.SLICE_AXIS)
    dataset = SliceInterpolationDataset(all_slices, global_min, global_max, cfg.PADDING_SIZE)
    train_loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, pin_memory=True)

    # 2. Build Model Components
    model = build_unet(cfg).to(cfg.DEVICE)
    scheduler = build_ddpm_scheduler(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(device='cuda')

    # 3. Train
    train_model(model, train_loader, scheduler, optimizer, scaler, cfg)