from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

def build_unet(cfg):
    return DiffusionModelUNet(
        spatial_dims=2,
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        num_channels=cfg.CHANNELS,
        attention_levels=cfg.ATTENTION_LEVELS,
        num_res_blocks=cfg.NUM_RES_BLOCKS,
        num_head_channels=cfg.NUM_HEAD_CHANNELS,
    )

def build_ddpm_scheduler(cfg):
    return DDPMScheduler(
        num_train_timesteps=cfg.NUM_TRAIN_TIMESTEPS,
        schedule="scaled_linear_beta",
        beta_start=cfg.BETA_START,
        beta_end=cfg.BETA_END
    )

def build_ddim_scheduler(cfg):
    scheduler = DDIMScheduler(
        num_train_timesteps=cfg.NUM_TRAIN_TIMESTEPS, 
        schedule="scaled_linear_beta",
        beta_start=cfg.BETA_START,
        beta_end=cfg.BETA_END
    )
    scheduler.set_timesteps(num_inference_steps=cfg.NUM_INFERENCE_STEPS)
    return scheduler