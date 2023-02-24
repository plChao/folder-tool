import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from matplotlib_venn import venn2
from matplotlib.ticker import PercentFormatter

def dist_vis(numbers, binwidth, name="Unknown", font_size=30, std_line=False, describe=False, cumulative=False):
    font = {
        # 'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : font_size}
    plt.rc('font', **font)
    dc = pd.Series(numbers).describe()
    if describe:
        print(dc)
    numbers = pd.Series(numbers).dropna().to_list()

    plt.figure(figsize=(30, 10))
    plt.title(name)
    #         int(max(numbers)) + binwidth, binwidth))
    plt.hist(numbers, bins=np.arange(int(min(numbers)), \
            int(max(numbers)) + binwidth, binwidth), rwidth=0.8\
            , density=True)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    if std_line:
        m = dc['mean']
        std = dc['std']
        plt.axvline(m, color='k', linestyle="dashed")
        plt.axvline(m+std, color='y', linestyle="dashed")
        plt.axvline(m-std, color='y', linestyle="dashed")
        plt.axvline(m+2*std, color='g', linestyle="dashed")
        plt.axvline(m-2*std, color='g', linestyle="dashed")


    plt.show()
    if cumulative:
        plt.figure(figsize=(30, 10))
        plt.title(name)
        #         int(max(numbers)) + binwidth, binwidth))
        plt.hist(numbers, bins=np.arange(int(min(numbers)), \
                int(max(numbers)) + binwidth, binwidth), rwidth=0.8\
                , density=True , cumulative=True)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.05))
        plt.grid(axis='y')
        plt.show()
def visualize(imgs, name_list = [], width=3, figsize=(20, 10)):
    if name_list == []:
        name_list = ["" for _ in range(len(imgs))]
    count = 1
    hight = (len(imgs) + width - 1) // width
    plt.figure(figsize=figsize)
    for img, name in zip(imgs, name_list):
        plt.subplot(hight, width, count)
        plt.title(name)
        plt.imshow(img, cmap='gray')
        count += 1
    plt.show()
def draw_binary_on_image(base_image, binary_img, color = (255, 0, 0)):
    base_image = base_image.copy()
    if len(base_image.shape) == 2 or base_image.shape[2] == 1:
        base_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    assert base_image.shape[2] == 3, base_image.shape
    assert base_image.shape[:2] == binary_img.shape[:2], base_image.shape[:2]
    coords = cv2.findNonZero(binary_img.astype(np.float32))

    # iterate over the coordinates and replace the pixels in image_A with red color
    for coord in coords:
        x, y = coord[0]
        base_image[y, x] = color

    return base_image
def vis_venn2(num_list_a, num_list_b, name_a = "first_set", name_b = "second_set"):
    set_a = set(num_list_a)
    set_b = set(num_list_b)
    venn2([set_a, set_b], set_labels=(name_a, name_b))
    plt.show()