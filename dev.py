import numpy as np
from skimage import io, exposure, img_as_uint, img_as_float
import rawpy

im2 = io.imread('dataset/Sony/short_temp_down/00002_08_0.1s.png')
print(im2.dtype)

def downsample(in_path):

    with rawpy.imread(in_path) as raw:
        rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)

    print(rgb.shape)
    print(rgb.shape[0]/4, rgb.shape[1]/4)
    #rgb_downsample = resize(rgb, (1280 , 720), anti_aliasing=True)
    
    #rgb = Image.fromarray(rgb)
    #rgb_downsample = rgb.resize((1280 , 720), resample=Image.LANCZOS)

    #rgb_np = np.array(rgb)
    #print(rgb_np.shape)
    #plt.imshow(rgb)
    #plt.show()

    #return rgb_downsample

#in_path = 'dataset/Sony/short/00001_00_0.1s.ARW'
#downsample(in_path)
import os
d1 = sorted(os.listdir('dataset/Sony/short_temp_down/'))
d2 = sorted(os.listdir('dataset/Sony/long_temp_down/'))

print(d1[1].split('_'))