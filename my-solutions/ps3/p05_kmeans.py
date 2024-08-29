from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

NUM_CENTROID = 16

A = imread('../data/peppers-small.tiff')
B = imread('../data/peppers-large.tiff')
SIZE = A.shape[0]

flat = A.reshape(-1, 3)

p_indices = np.random.choice(SIZE * SIZE, NUM_CENTROID, False)


pre_ct = None
centroid = flat[p_indices, :].astype('float64')
assignments = None

max_iter = 5
it = 0
while it < max_iter and (pre_ct is None or not np.array_equal(pre_ct, centroid)):
    assignments = np.argmin(np.linalg.norm(flat.reshape(-1, 1, 3) - centroid, axis=-1), axis=1)
    pre_ct = np.copy(centroid)
    for i in range(NUM_CENTROID):
        filter = flat[assignments == i, :]
        centroid[i] = np.mean(filter, axis=0) if filter.size != 0 else centroid[i] 
    
    
    it += 1

# last assignments
assignments = np.argmin(np.linalg.norm(B.reshape(-1, 1, 3) - centroid, axis=-1), axis=1)

output = centroid[assignments].reshape(512, 512, 3).astype(int)

print(output.shape)

plt.imshow(output)
plt.show()