import argparse
import os
import torch
import nibabel as nib
import numpy as np
import logging
from src.config import Config
from src.data.loader import extract_slices
from src.data.dataset import SliceInterpolationDataset
from src.models.components import build_unet, build_ddim_scheduler
from src.inference.generator import generate_intermediate_slice

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Run Inference for 3D Volume Interpolation")
    parser.add_argument("--img_path", type=str, required=True, help="Path to target NIfTI image")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save generated NIfTI")
    args = parser.parse_args()

    cfg = Config()
    cfg.TRAIN_IMG_PATH = args.img_path
    cfg.OUTPUT_DIR = args.output_dir
    cfg.CHECKPOINT_PATH = os.path.join(cfg.OUTPUT_DIR, "best_diffusion_checkpoint.pth")

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # 1. Load Data bounds and get a sample triplet
    all_slices, global_min, global_max = extract_slices(cfg.TRAIN_IMG_PATH, cfg.SLICE_AXIS)
    dataset = SliceInterpolationDataset(all_slices, global_min, global_max, cfg.PADDING_SIZE)
    
    sample_triplet = dataset[10] # Grab a test triplet (e.g., index 10)
    condition_tensor = sample_triplet["condition"] 
    slice_prev = condition_tensor[0:1].unsqueeze(0) 
    slice_next = condition_tensor[1:2].unsqueeze(0) 

    # 2. Build Model and DDIM Scheduler
    model = build_unet(cfg).to(cfg.DEVICE)
    scheduler = build_ddim_scheduler(cfg)

    # 3. Load Checkpoint
    checkpoint = torch.load(cfg.CHECKPOINT_PATH, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from {cfg.CHECKPOINT_PATH}")

    # 4. Generate
    generated_slice = generate_intermediate_slice(
        model, scheduler, slice_prev, slice_next, global_min, global_max, cfg
    )
    
    # 5. Save generated slice as NIfTI in the output folder
    # We load the original image just to grab its affine matrix for proper spatial coordinates
    original_img = nib.load(cfg.TRAIN_IMG_PATH)
    original_affine = original_img.affine
    
    # Add a dummy Z dimension so it is formally a 3D volume (H, W, 1)
    generated_3d = np.expand_dims(generated_slice, axis=-1)
    
    nii_img = nib.Nifti1Image(generated_3d, affine=original_affine)
    out_file = os.path.join(cfg.OUTPUT_DIR, "generated_slice.nii.gz")
    nib.save(nii_img, out_file)
    logger.info(f"Successfully saved generated slice to {out_file}")