import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from tqdm import tqdm


def show_kpts(img, kpts, cmap=None):
    plt.imshow(img, cmap=cmap)
    plt.scatter(kpts[:, 0], kpts[:, 1], s=20, marker=".", color="m")


def normalize(sample, mean, std):
    image, key_pts = sample['image'], sample['keypoints']

    image_copy = np.copy(image)

    # convert image to grayscale
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

    # scale color range from [0, 255] to [0, 1]
    image_copy = image_copy/255.0

    if(key_pts is not None):
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = np.copy(key_pts)
        key_pts_copy = (key_pts_copy - mean)/std

    return {'image': image_copy, 'keypoints': key_pts_copy if key_pts is not None else None}


def rescale(sample, output_size):
    image, key_pts = sample['image'], sample['keypoints']

    h, w = image.shape[:2]
    if isinstance(output_size, int):
        if h > w:
            new_h, new_w = output_size * h / w, output_size
        else:
            new_h, new_w = output_size, output_size * w / h
    else:
        new_h, new_w = output_size

    new_h, new_w = int(new_h), int(new_w)

    img = cv2.resize(image, (new_w, new_h))

    if key_pts is not None:
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

    return {'image': img, 'keypoints': key_pts if key_pts is not None else None}


def randomCrop(sample, output_size):
    image, key_pts = sample['image'], sample['keypoints']

    h, w = image.shape[:2]
    new_h, new_w = output_size

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    image = image[top: top + new_h,
                  left: left + new_w]

    if key_pts is not None:
        key_pts = key_pts - [left, top]

    return {'image': image, 'keypoints': key_pts if key_pts is not None else None}


def create_data_set(df, grand_mean, grand_std, rescale_size, crop_size):
    X = []
    y = []
    for i in tqdm(range(len(df))):
        img_name = df.iloc[i, 0]
        kpts = df.iloc[i, 1:].values
        kpts = kpts.astype("float32").reshape(-1, 2)
        img = mpimg.imread(os.path.join('data/training/', img_name))
        sample = {"image": img, "keypoints": kpts}
        sample = rescale(sample, rescale_size)
        sample = randomCrop(sample, crop_size)
        if type(grand_mean) == np.ndarray:
            grand_mean = grand_mean.reshape(-1, 2)
            grand_std = grand_std.reshape(-1, 2)
        sample = normalize(sample, grand_mean, grand_std)
        X.append(sample['image'])
        y.append(sample['keypoints'].reshape(-1, 1))

    return np.array(X), np.array(y)
