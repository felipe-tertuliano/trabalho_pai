import os

input_folder = "axl_img"

images_to_remove = [
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

removed_count = 0

for image_name in images_to_remove:
    image_path = os.path.join(input_folder, f"{image_name}.nii")
    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"Removido: {image_name}.nii")
        removed_count += 1
    else:
        print(f"Não encontrado: {image_name}.nii")

print(f"\nTotal de {removed_count} arquivos .nii removidos da pasta '{input_folder}'")

