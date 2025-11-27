import os
import glob

output_folder = "imgs"

images_to_keep = [
    "OAS2_0013_MR2_axl",
    "OAS2_0048_MR5_axl",
    "OAS2_0081_MR1_axl",
    "OAS2_0172_MR2_axl",
    "OAS2_0179_MR1_axl",
    "OAS2_0004_MR2_axl",
    "OAS2_0051_MR3_axl",
    "OAS2_0051_MR2_axl",
    "OAS2_0098_MR1_axl",
    "OAS2_0098_MR2_axl",
    "OAS2_0118_MR1_axl",
    "OAS2_0156_MR1_axl",
    "OAS2_0159_MR1_axl"
]

images_to_keep_set = set(f"{name}.png" for name in images_to_keep)

all_images = glob.glob(os.path.join(output_folder, "*.png"))

removed_count = 0

for image_path in all_images:
    image_filename = os.path.basename(image_path)
    if image_filename not in images_to_keep_set:
        os.remove(image_path)
        print(f"Removido: {image_filename}")
        removed_count += 1

print(f"\nTotal de {removed_count} imagens removidas da pasta '{output_folder}'")
print(f"Mantidas {len(images_to_keep)} imagens na pasta")

