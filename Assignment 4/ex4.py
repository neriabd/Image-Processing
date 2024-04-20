import cv2
import numpy as np


def blend_low_high_res_images(source_image_path, dest_image_path, ransac_threshold):
    """
    :param source_image_path: path to high resolution image
    :param dest_image_path: path to low resolution image
    :param ransac_threshold: maximum distance for points to be considered as inliers
    :return: The blended Image
    """
    source_image, dest_image, source_gray, dest_gray = open_images(source_image_path, dest_image_path)

    # get key-points and descriptors from both images using sift
    keypoints_source, descriptors_source, keypoints_dest, descriptors_dest = sift(source_gray, dest_gray)

    # Filter Descriptors based on (neighbor distance) / (neighbor2 distance) = 0.75
    filtered_descriptors = filter_descriptors(descriptors_source, descriptors_dest)

    keypoints_source_pos, keypoints_dest_pos = extract_kp_positions(filtered_descriptors, keypoints_source,
                                                                    keypoints_dest)

    # Perform RANSAC to estimate transformation
    transformation, indices = cv2.findHomography(keypoints_source_pos, keypoints_dest_pos, cv2.RANSAC,
                                                 ransacReprojThreshold=ransac_threshold)

    # width - dest_image.shape[1], height - dest_image.shape[0]
    transformed_image = cv2.warpPerspective(source_image, transformation, (dest_image.shape[1], dest_image.shape[0]))
    mask_indices = indexes_to_blend(transformed_image)

    # Replace black pixels with corresponding pixels from dest_image (the background)
    transformed_image[mask_indices] = dest_image[mask_indices]
    return transformed_image


def open_images(image1_path, image2_path):
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    return image1, image2, gray1, gray2


def filter_descriptors(kp_descriptors1, kp_descriptors2):
    """
    :param kp_descriptors1: descriptors - source image
    :param kp_descriptors2: descriptors - dest image
    :return: descriptors that are distinct in their observations
    """
    filtered_matches = []
    two_closest_descriptors = cv2.BFMatcher().knnMatch(kp_descriptors1, kp_descriptors2, k=2)
    for neighbor1, neighbor2 in two_closest_descriptors:
        if neighbor1.distance < 0.75 * neighbor2.distance:
            filtered_matches.append(neighbor1)

    return filtered_matches


def sift(gray1, gray2):
    # Initialize SIFT detector
    sift_object = cv2.SIFT_create()

    # Detect key-points and descriptors
    kp1, kp_descriptors1 = sift_object.detectAndCompute(gray1, None)
    kp2, kp_descriptors2 = sift_object.detectAndCompute(gray2, None)
    return kp1, kp_descriptors1, kp2, kp_descriptors2


def indexes_to_blend(transformed_image):
    gray_transformed = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)

    # if value == 0 -> 0 else 255
    _, binary_mask = cv2.threshold(gray_transformed, 0, 255, cv2.THRESH_BINARY)
    inverted_mask = cv2.bitwise_not(binary_mask) / 255

    kernel_size = (7, 7)
    inverted_mask = cv2.GaussianBlur(inverted_mask, kernel_size, 1)

    mask_indices = np.where(inverted_mask > 0)

    return mask_indices


def extract_kp_positions(filtered_descriptors, kp1, kp2):
    kp1_positions, kp2_positions = [], []
    for descriptor in filtered_descriptors:
        # Extract coordinates from key-points in the first image
        kp1_pos = kp1[descriptor.queryIdx].pt
        kp1_positions.append(kp1_pos)

        # Extract coordinates from key-points in the second image
        kp2_pos = kp2[descriptor.trainIdx].pt
        kp2_positions.append(kp2_pos)

    len_matched_kp = len(kp1_positions)
    kp1_positions = np.array(kp1_positions, dtype=np.float32).reshape(len_matched_kp, 1, 2)
    kp2_positions = np.array(kp2_positions, dtype=np.float32).reshape(len_matched_kp, 1, 2)
    return kp1_positions, kp2_positions
