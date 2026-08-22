import torch
import torch.nn.functional as F
import logging
import os

logger = logging.getLogger(__name__)

def train_model(model, train_loader, scheduler, optimizer, scaler, cfg):
    best_loss = float('inf')
    start_epoch = 0

    # Resume from checkpoint if it exists
    if os.path.exists(cfg.CHECKPOINT_PATH):
        checkpoint = torch.load(cfg.CHECKPOINT_PATH, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        logger.info(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.EPOCHS):
        model.train()
        epoch_loss = 0.0
        torch.cuda.empty_cache()
        
        for step, batch in enumerate(train_loader):
            target = batch["target"].to(cfg.DEVICE)       
            condition = batch["condition"].to(cfg.DEVICE) 
            
            with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                noise = torch.randn_like(target).to(cfg.DEVICE)
                timesteps = torch.randint(0, scheduler.num_train_timesteps, (target.shape[0],), device=cfg.DEVICE).long()
                
                noisy_target = scheduler.add_noise(target, noise, timesteps)
                network_input = torch.cat((noisy_target, condition), dim=1)
                noise_pred = model(network_input, timesteps)
                
                loss = F.mse_loss(noise_pred.float(), noise.float()) / cfg.ACCUMULATION_STEPS
                
            scaler.scale(loss).backward()
            
            if (step + 1) % cfg.ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                
            epoch_loss += (loss.item() * cfg.ACCUMULATION_STEPS)
            
        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"End of Epoch {epoch+1} | Avg Loss: {avg_epoch_loss:.5f}")
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_loss': best_loss
            }, cfg.CHECKPOINT_PATH)