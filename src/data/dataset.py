import torch
from torch.utils.data import Dataset
from monai.transforms import SpatialPad

class SliceInterpolationDataset(Dataset):
    def __init__(self, slice_list, global_min, global_max, pad_size=(320, 272)):
        self.slice_list = slice_list
        self.num_triplets = len(slice_list) - 2
        self.global_min = global_min
        self.global_max = global_max
        self.padder = SpatialPad(spatial_size=pad_size, mode="minimum")

    def __len__(self):
        return self.num_triplets

    def __getitem__(self, idx):
        slice_prev = self.slice_list[idx]
        slice_target = self.slice_list[idx + 1]
        slice_next = self.slice_list[idx + 2]
        
        slice_prev = (slice_prev - self.global_min) / (self.global_max - self.global_min)
        slice_target = (slice_target - self.global_min) / (self.global_max - self.global_min)
        slice_next = (slice_next - self.global_min) / (self.global_max - self.global_min)
        
        tensor_prev = torch.tensor(slice_prev, dtype=torch.float32).unsqueeze(0)
        tensor_target = torch.tensor(slice_target, dtype=torch.float32).unsqueeze(0)
        tensor_next = torch.tensor(slice_next, dtype=torch.float32).unsqueeze(0)
        
        tensor_prev = self.padder(tensor_prev)
        tensor_target = self.padder(tensor_target)
        tensor_next = self.padder(tensor_next)
        
        condition = torch.cat([tensor_prev, tensor_next], dim=0)
        
        return {"target": tensor_target, "condition": condition}