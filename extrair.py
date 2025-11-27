import os
import glob
import nibabel as nib
import numpy as np
from PIL import Image

input_folder = "axl"
output_folder = "imgs"

os.makedirs(output_folder, exist_ok=True)

nii_files = glob.glob(os.path.join(input_folder, "*.nii"))
nii_gz_files = glob.glob(os.path.join(input_folder, "*.nii.gz"))
all_files = nii_files + nii_gz_files

for nii_path in all_files:
    filename = os.path.basename(nii_path)
    if filename.endswith(".nii.gz"):
        base_name = filename[:-7]
    else:
        base_name = os.path.splitext(filename)[0]
    
    img = nib.load(nii_path)
    data = img.get_fdata()
    
    if len(data.shape) == 3:
        middle_slice = data.shape[2] // 2
        slice_data = data[:, :, middle_slice]
    else:
        slice_data = data
    
    slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
    slice_data = (slice_data * 255).astype(np.uint8)
    
    image = Image.fromarray(slice_data)
    
    output_path = os.path.join(output_folder, f"{base_name}.png")
    image.save(output_path)
    
    print(f"Convertido: {filename} -> {base_name}.png")

print(f"\nTotal de {len(all_files)} arquivos convertidos para PNG na pasta '{output_folder}'")


