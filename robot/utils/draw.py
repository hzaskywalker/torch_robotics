import numpy as np
from matplotlib import pyplot as plt 

def draw_vector(ld, value, min_value=-200, max_value=200):
    import cv2

    mi = ld.min(dim=0)[0]
    ma = ld.max(dim=0)[0]
    ld = (ld - mi)/(ma - mi) * 512
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    print(value.min(), value.max())

    value = value - value.min()
    value = value / (value.max() + 1e-9)
    for idx, (i, v) in enumerate(zip(ld, value)):
        c = 32 + int((255-32) * v)
        cv2.circle(img, (int(i[0]), int(i[1])), 3, (c, c, c), -1)
    #cv2.imwrite('x.jpg', img)
    return img

def draw_correlations(feature, sample_label):
    fig = plt.figure()

    fig.clf()
    t = 0
    num_dim_1 = feature.shape[1]
    num_dim_2 = sample_label.shape[1]
    for i in range(num_dim_1):
        for j in range(num_dim_2):
            t += 1

            fig.add_subplot(num_dim_1, num_dim_2, t)
            plt.scatter(feature[:, i], sample_label[:, j], s=0.5)

            #plts_object(fig, feature[:, i], sample_label[:, j], t,num_dim)
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = np.array(data, dtype=np.uint8)[None, :]
    return data