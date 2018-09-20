import deepplantphenomics as dpp
import numpy as np
from PIL import Image
import os

import glob

os.chdir('C://Users//xxx//Desktop//smartfarm_data//panel1')

output_dir = 'C://Users//xxx//Desktop//smartfarm_data//segmented-images'

os.chdir('C://Users//xxx//Desktop//smartfarm_data//panel1')
panel1_h = glob.glob('*.jpg')
panel1_h.sort(key=os.path.getmtime)
my_files = ['C://Users//xxx//Desktop//smartfarm_data//panel1//' + s for s in panel1_h]


y = dpp.tools.segment_vegetation(my_files)

for i, img in enumerate(y):
    # Get original image dimensions
    org_filename = my_files[i]
    org_img = Image.open(org_filename)
    org_width, org_height = org_img.size
    org_array = np.array(org_img)

    # Resize mask
    mask_img = Image.fromarray((img * 255).astype(np.uint8))
    mask_array = np.array(mask_img.resize((org_width, org_height))) / 255

    # Apply mask
    img_seg = np.array([org_array[:,:,0] * mask_array, org_array[:,:,1] * mask_array, org_array[:,:,2] * mask_array]).transpose()

    # Write output file
    filename = my_files[i][0:len(my_files[i])-4]+'_seg.jpg'
    result = Image.fromarray(img_seg.astype(np.uint8))
    result.save(filename)
    