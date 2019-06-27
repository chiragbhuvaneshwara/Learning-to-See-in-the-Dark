import os
import numpy as np
import rawpy
from PIL import Image
from skimage import io
from skimage.transform import resize
import warnings
warnings.filterwarnings("ignore")

#in_path = 'dataset/Sony/short/00001_00_0.1s.ARW'
#in_path = 'dataset/Sony/short/00001_01_0.04s.ARW'
# in_path = 'dataset/Sony/long/00001_00_10s.ARW'

def downsample(in_path):

    with rawpy.imread(in_path) as raw:
        rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)

    #rgb_downsample = resize(rgb, (1280 , 720), anti_aliasing=True)
    reqd_shape = (rgb.shape[1]//4, rgb.shape[0]//4 )
    rgb = Image.fromarray(rgb)
    rgb_downsample = rgb.resize(reqd_shape, resample=Image.LANCZOS)

    #rgb_np = np.array(rgb)
    #print(rgb_np.shape)
    #plt.imshow(rgb)
    #plt.show()

    return rgb_downsample


def downsample_short(data_dir, save_dir, verbose=False):
    
    dl = sorted(os.listdir(data_dir))[1:]

    imageNameCount = {}
    i = 0
    for f in dl:
        i+=1
        #print('Downsampling Image ', i, 'out of', len(dl))
        
        imageName = f.split('_')[0]

        if imageName not in imageNameCount:
            imageNameCount[imageName] = 1
        else: 
            imageNameCount[imageName] += 1
        
        downsample_image = downsample(os.path.join(data_dir, f))
        
        parts = f.split('.')
        new_f = parts[0] + '.' + parts[1] + '.png'
        new_f = ''.join(c for c in new_f)
        # parts = new_f.split('_')
        # new_f = parts[0] +'_' + parts[1] +'_' + parts[2]

        #io.imsave(save_dir + new_f, downsample_image)
        downsample_image.save(save_dir + new_f)

        if verbose and i%10 == 0:
            print(i, 'images downsampled out of', len(dl))
    print('All images in',data_dir,'are downsampled')

    return imageNameCount

def downsample_long(data_dir, save_dir, imageCountFromShort, verbose = False):
    
    dl = sorted(os.listdir(data_dir))[1:]

    imageNameCount = {}
    i = 0
    for f in dl:
        i+=1
        #print('Downsampling Image ', i, 'out of', len(dl))
        
        imageName = f.split('_')[0]
        count = imageCountFromShort[imageName]

        downsample_image = downsample(os.path.join(data_dir, f))
        
        parts = f.split('_')

        for j in range(count):
            new_f = parts[0] +'_'+ str(int(parts[1])+j) +'_'+ parts[2][:-3] + 'png'
            new_f = ''.join(c for c in new_f)
            #io.imsave(save_dir + new_f, downsample_image)
            downsample_image.save(save_dir + new_f)

        if verbose and i%10 == 0:
            print(i, 'images downsampled out of', len(dl))
    
    print('All images in',data_dir,'are downsampled')

imageNameCount_short = downsample_short('dataset/Sony/short_temp/', 'dataset/Sony/short_temp_down/', verbose=True)
print('##################################')

downsample_long('dataset/Sony/long_temp/', 'dataset/Sony/long_temp_down/', imageNameCount_short, verbose = True)

d1 = sorted(os.listdir('dataset/Sony/short_temp_down/'))
d2 = sorted(os.listdir('dataset/Sony/long_temp_down/'))

print(d1[1].split('_'))
print(d2[1])


