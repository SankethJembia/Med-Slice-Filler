import torch
from monai.transforms import CenterSpatialCrop

def generate_intermediate_slice(model, scheduler, slice_prev, slice_next, global_min, global_max, cfg):
    model.eval()
    cropper = CenterSpatialCrop(roi_size=cfg.CROP_SIZE)
    condition = torch.cat([slice_prev, slice_next], dim=1).to(cfg.DEVICE)
    
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            x_t = torch.randn((1, 1, cfg.PADDING_SIZE[0], cfg.PADDING_SIZE[1]), device=cfg.DEVICE)
            
            for t in scheduler.timesteps:
                timestep = torch.tensor([t], device=cfg.DEVICE).long()
                network_input = torch.cat((x_t, condition), dim=1)
                noise_pred = model(network_input, timesteps=timestep)
                x_t, _ = scheduler.step(model_output=noise_pred, timestep=t, sample=x_t)
            
    generated_tensor = x_t.detach().cpu().squeeze(0)
    cropped_tensor = cropper(generated_tensor)
    
    final_array = cropped_tensor.squeeze().numpy()
    final_array = final_array * (global_max - global_min) + global_min
    
    return final_array