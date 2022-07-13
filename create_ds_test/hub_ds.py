import hub
from PIL import Image
import numpy as np
import os

# Dataset its created locally
ds = hub.empty('./animals_hub')

# We search the class names and list of files needed

dataset_folder = './animals'

class_names = os.listdir(dataset_folder)

files_list = []

for dirpath, dirnames, filenames in os.walk(dataset_folder):
    for filename in filenames:
        files_list.append(os.path.join(dirpath, filename))

with ds:
    #Create the tensors with names of your choice
    ds.create_tensor('images', htype = 'image', sample_compression = 'jpeg')
    ds.create_tensor('labels', htype = 'class_label', class_names = class_names)

    # Add arbitrary metadata 
    ds.info.update(description = 'My first hub dataset')
    ds.images.info.update(camera_type = 'SLR')


with ds:
    # Iterate through the files and append to hub dataset
    for file in files_list:
        label_text = os.path.basename(os.path.dirname(file))
        label_num = class_names.index(label_text)

        # Append data to the tensors
        ds.append({'images': hub.read(file), 'labels': np.uint32(label_num)})
        
ds.summary()

