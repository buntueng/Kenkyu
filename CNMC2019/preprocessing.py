# create preprocessing files
# croping raw image to 350x350 pixel
import os
from PIL import Image

cnmc_dataset_path = os.path.join("D:","Research","CNMC2019","augmented_dataset")
preprocessing_path = os.path.join("D:","Research","CNMC2019","preprocessing")

if not os.path.exists(preprocessing_path):
    os.makedirs(preprocessing_path)

# read files and folders from the cnmc_dataset_path
file_paths = []
for root, dirs, files in os.walk(cnmc_dataset_path):
    for file in files:
        if file.endswith(".bmp"):
            file_paths.append([root,dirs,file])
            
# read and crop images, then save them to the preprocessing_path
for file_path in file_paths:
    folder = file_path[0].split("\\")[-1]
    image = Image.open(os.path.join(file_path[0],file_path[2]))
    # crop image using center as the anchor point
    width, height = image.size
    left = (width - 350)/2
    top = (height - 350)/2
    right = (width + 350)/2
    bottom = (height + 350)/2
    image = image.crop((left,top,right,bottom))
    save_path = os.path.join(preprocessing_path, folder)
    os.makedirs(save_path, exist_ok=True)
    image.save(os.path.join(save_path, file_path[2]))


            

