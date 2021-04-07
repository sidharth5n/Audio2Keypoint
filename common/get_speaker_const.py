import os
import numpy as np
import random

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk('./Gestures/human/keypoints_simple'):
    for file in f:
        files.append(os.path.join(r, file))
print(len(files))
random.shuffle(files)
random.shuffle(files)
shape = (len(files[:200000]), 2, 68)
array = np.zeros(shape)
for i, f in enumerate(files[:200000]):
    print(i)
    array[i] = np.loadtxt(f)
    #print(array[i])

array = array - array[:,:,27:28]
mean = np.mean(array, axis=0)
median = np.median(array, axis=0)
std = np.std(array, axis=0)
print(mean.shape)
print(median.shape)
print(std.shape)
import pdb
pdb.set_trace()
