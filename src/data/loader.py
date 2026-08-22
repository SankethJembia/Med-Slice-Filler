import nibabel as nib
import numpy as np

def extract_slices(nii_file_path, slice_axis=-1):
    img = nib.load(nii_file_path)
    volume = img.get_fdata()
    
    if len(volume.shape) > 3:
        volume = np.squeeze(volume)

    num_slices = volume.shape[slice_axis]
    slice_list = [np.take(volume, i, axis=slice_axis) for i in range(num_slices)]
    
    return slice_list, volume.min(), volume.max()