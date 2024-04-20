import cv2
import numpy as np
from scipy.signal import convolve


def image_blending(image1_path, image2_path, mask_path, levels):
    image1 = np.float32(cv2.imread(image1_path))
    image2 = np.float32(cv2.imread(image2_path))
    mask = np.round(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255)
    num_colors = 3

    # create gaussian pyramids
    Gb1, Gg1, Gr1 = gaussian_rgb_pyramid(image1, levels)
    Gb2, Gg2, Gr2 = gaussian_rgb_pyramid(image2, levels)
    mask = gaussian_pyramid(mask, levels)

    # create laplacian pyramids
    Lb1, Lg1, Lr1 = laplacian_rgb_pyramid(Gb1, Gg1, Gr1)
    Lb2, Lg2, Lr2 = laplacian_rgb_pyramid(Gb2, Gg2, Gr2)

    # add the two laplacian according to mask
    blend_laplacian = [[] for i in range(num_colors)]

    for i in range(levels, -1, -1):
        for j, L1, L2 in zip(range(num_colors), (Lb1, Lg1, Lr1), (Lb2, Lg2, Lr2)):
            blend_laplacian[j].append(blend(L1, L2, mask, i))

    L_colored = []
    for i in range(num_colors):
        L_colored.append(sum_laplacian(blend_laplacian[i][::-1]))

    output = np.stack(L_colored, axis=2)

    return output


def image_hybrid(image1_path, image2_path, levels):
    image1 = np.float32(cv2.cvtColor(cv2.imread(image1_path), cv2.COLOR_BGR2GRAY))
    image2 = np.float32(cv2.cvtColor(cv2.imread(image2_path), cv2.COLOR_BGR2GRAY))
    k = levels // 2 + 1

    # create gaussian pyramids
    G1 = gaussian_pyramid(image1, levels)
    G2 = gaussian_pyramid(image2, levels)

    expand1 = expand_pyramids(G1)
    expand2 = expand_pyramids(G2)

    # create laplacian pyramids
    L1 = laplacian_pyramid(G1, expand1)
    L2 = laplacian_pyramid(G2, expand2)

    composed_L = L1[:k - 1] + L2[k - 1:]
    composed_L = sum_laplacian(composed_L)

    return composed_L


def blend(L1, L2, mask, i):
    return mask[i] * L1[i] + (1 - mask[i]) * L2[i]


def kernel1d(image, kernel_type="exapnd"):
    if kernel_type == "blur3":
        kernel = [[1, 2, 1]]
        kernel = np.array(kernel) / sum(kernel[0])
    elif kernel_type == "blur5":
        kernel = [[1, 2, 4, 2, 1]]
        kernel = np.array(kernel) / sum(kernel[0])
    elif kernel_type == "expand":
        kernel = np.array([[1 / 2, 1, 1 / 2]])

    # use without FFT - small kernel
    p = convolve(image, kernel, mode='same', method='direct')
    return convolve(p, kernel.transpose(), mode='same', method='direct')


""" ########################## GAUSSIAN PYRAMID ########################## """


def gaussian_rgb_pyramid(image, levels):
    b_image, g_image, r_image = cv2.split(image)
    b = gaussian_pyramid(b_image, levels)
    g = gaussian_pyramid(g_image, levels)
    r = gaussian_pyramid(r_image, levels)
    return b, g, r


def gaussian_pyramid(image, levels):
    """
    @param image: G0 - original_image
    @param levels: number of levels of pyramids requested
    @return: [G0, ..., Gn], [Expand(G0).shape, Expand(G1).shape, ... , Expand(Gn).shape]
    """
    G_pyramids = [image]

    for level in range(levels):
        # Apply Gaussian blur to the image
        pyramid = kernel1d(image, "blur3")[::2, ::2]

        G_pyramids.append(pyramid)
        image = pyramid

    return G_pyramids


""" ########################## EXPAND PYRAMID ########################## """


def expand_pyramids(G_pyramids):
    """
    @param G_pyramids: G0, G1, G2, ..., Gn
    @return: the restore pyramids from G0, ..., Gn
    """
    restore = []
    shapes = np.array([G.shape for G in G_pyramids[:-1]])
    for i, shape, pyramid in zip(range(len(G_pyramids)), shapes, G_pyramids[1:]):
        res_pyramid = pyramid_up(pyramid, shape)
        restore.append(res_pyramid)

    return restore


def pyramid_up(image, og_shape):
    """
    @param image: image to rescale
    @param og_shape: shape to increase size to
    @return: the scaled up image
    """
    expand_image = np.zeros(og_shape, dtype=np.float32)
    for i in range(0, og_shape[0], 2):
        for j in range(og_shape[1]):
            if j % 2 == 0:
                expand_image[i, j] = image[i // 2, j // 2]

    return kernel1d(expand_image, "expand")


""" ########################## LAPLACIAN PYRAMID ########################## """


def laplacian_rgb_pyramid(Gb, Gg, Gr):
    expand_Gb = expand_pyramids(Gb)
    expand_Gg = expand_pyramids(Gg)
    expand_Gr = expand_pyramids(Gr)

    Lb = laplacian_pyramid(Gb, expand_Gb)
    Lg = laplacian_pyramid(Gg, expand_Gg)
    Lr = laplacian_pyramid(Gr, expand_Gr)
    return Lb, Lg, Lr


def laplacian_pyramid(G_pyramids, restore_pyramids):
    """
    @param G_pyramids: G0, G1, G2, ..., Gn
    @param restore_pyramids: expand(G1), expand(G1), ..., expand(Gn-1)
    @return: L0, L1, ... Ln
    """
    laplacian = []
    for i in range(len(restore_pyramids)):
        laplacian.append(np.array(G_pyramids[i]) - np.array(restore_pyramids[i]))

    laplacian.append(G_pyramids[-1])
    return laplacian


def sum_laplacian(L):
    """
    @param laplacians: sum laplacian
    @return: sum of laplacian
    """
    L_sum = L[-1]
    for i in range(len(L) - 2, -1, -1):
        L_sum = L[i] + pyramid_up(L_sum, L[i].shape)

    return L_sum
