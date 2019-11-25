import matplotlib.pyplot as plt
import numpy as np
import os

def plot(images, save_dir, step):
    target_dir = os.path.join(save_dir, str(step))
    n = 8
    pad = 2
    width = images[0].shape[0]
    image = np.zeros(((width + pad) * n - pad, (width + pad) * n - pad))
    for i in range(n):
        for j in range(n):
            image[(width + pad) * i:(width + pad) * i + width, (width + pad) * j:(width + pad) * j + width] = images[i * n + j][:, :, 0]
    plt.imsave(os.path.join(save_dir, f'{step}.jpg'), image, cmap = 'gray')

    '''
    os.system(f'mkdir -p {target_dir}')
    for i in range(images.shape[0]):
        images[i] = ((images[i] + 1) / 2).astype(np.float64)
        plt.imsave(os.path.join(target_dir, str(i).zfill(3) + '.jpg'), images[i])
    '''

        
        
