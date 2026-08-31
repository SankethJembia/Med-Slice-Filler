import streamlit as st
import torch
import os
import tempfile
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

#pipeline components
from src.config import Config
from src.data.loader import extract_slices
from src.data.dataset import SliceInterpolationDataset
from src.models.components import build_unet, build_ddpm_scheduler, build_ddim_scheduler
from src.training.trainer import train_model
from src.inference.generator import generate_intermediate_slice

st.set_page_config(page_title="Volume Interpolation", page_icon="😷", layout="centered")

# Custom CSS for a clean, dark medical theme
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    h1, h2, h3 { color: #40e0d0; font-weight: 300; }
    .stButton>button { border-color: #40e0d0; color: #40e0d0; }
    .stButton>button:hover { background-color: #40e0d0; color: #0e1117; }
    </style>
""", unsafe_allow_html=True)

st.title("🥼 Med-Slice Filler")
st.write("Upload a NIfTI volume. The model will train on the slices and predict a missing target slice.")

uploaded_file = st.file_uploader("Upload NIfTI (.nii, .nii.gz)", type=["nii", "nii.gz"])

if uploaded_file is not None:
    # Streamlit holds files in memory, so we write it to a temporary file for Nibabel
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    # Initialize Config
    cfg = Config()
    cfg.TRAIN_IMG_PATH = tmp_path
    cfg.OUTPUT_DIR = "app_output"
    cfg.CHECKPOINT_PATH = os.path.join(cfg.OUTPUT_DIR, "best_diffusion_checkpoint.pth")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    if st.button("Start Pipeline"):
        
        
        with st.status("Initializing MLOps Pipeline...", expanded=True) as status:
            
            st.write("⚙️ Extracting slices from NIfTI volume...")
            all_slices, global_min, global_max = extract_slices(cfg.TRAIN_IMG_PATH, cfg.SLICE_AXIS)
            dataset = SliceInterpolationDataset(all_slices, global_min, global_max, cfg.PADDING_SIZE)
            train_loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
            
            st.write("🧠 Building Neural Networks...")
            model = build_unet(cfg).to(cfg.DEVICE)
            train_scheduler = build_ddpm_scheduler(cfg)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
            scaler = torch.amp.GradScaler(device='cuda')
            
            st.write("🔥 Training Diffusion Model (This will take a while)...")
            train_model(model, train_loader, train_scheduler, optimizer, scaler, cfg)
            
            st.write("✨ Running DDIM Inference...")
            infer_scheduler = build_ddim_scheduler(cfg)
            
            # Pick a triplet from the middle of the volume to visualize
            target_idx = len(dataset) // 2
            sample_triplet = dataset[target_idx]
            
            # Format inputs for inference
            condition_tensor = sample_triplet["condition"] 
            slice_prev = condition_tensor[0:1].unsqueeze(0) 
            slice_next = condition_tensor[1:2].unsqueeze(0) 
            
            # Grab the actual target for the bottom grid
            actual_target = sample_triplet["target"].squeeze().numpy() 
            actual_target = actual_target * (global_max - global_min) + global_min # Denormalize
            
            # Load the newly trained best checkpoint
            if os.path.exists(cfg.CHECKPOINT_PATH):
                checkpoint = torch.load(cfg.CHECKPOINT_PATH, weights_only=True)
                model.load_state_dict(checkpoint['model_state_dict'])

            # Generate
            generated_slice = generate_intermediate_slice(
                model, infer_scheduler, slice_prev, slice_next, global_min, global_max, cfg
            )
            
            status.update(label="Pipeline Complete!", state="complete", expanded=False)

        
        st.markdown("### Interpolation Results")
        
        
        def plot_slice(slice_array, title):
            fig, ax = plt.subplots(figsize=(5, 5))
            # Transpose with .T and use origin="lower" for standard medical orientation
            ax.imshow(slice_array.T, cmap="bone", origin="lower") 
            ax.axis("off")
            ax.set_title(title, color="#40e0d0", pad=15)
            fig.patch.set_facecolor('#0e1117')
            return fig

        # Top Row: Conditional Slices
        col1, col2 = st.columns(2)
        with col1:
            # We fetch the raw, un-padded slices directly from all_slices
            st.pyplot(plot_slice(all_slices[target_idx], "Top Condition Slice (Z-1)"))
        with col2:
            st.pyplot(plot_slice(all_slices[target_idx+2], "Bottom Condition Slice (Z+1)"))
            
        st.markdown("<br>", unsafe_allow_html=True) # Spacer
        
        # Bottom Row: Generated vs Actual
        col3, col4 = st.columns(2)
        with col3:
            # Note: actual_target still has the padding from the dataset, we crop it for UI
            h, w = cfg.CROP_SIZE
            st.pyplot(plot_slice(all_slices[target_idx+1], "Actual Target Slice (Ground Truth)"))
        with col4:
            st.pyplot(plot_slice(generated_slice, "AI Generated Slice"))