import numpy as np
import matplotlib.pyplot as plt
import mediapy as media
from itertools import product
import pandas as pd


NORMALIZATION_VALUE = 255


def main(video_path, video_type):
    """
    Main entry point for ex1
    :param video_path: path to video file
    :param video_type: category of the video (either 1 or 2)
    :return: a tuple of integers representing the frame number for which the
    scene cut was detected (i.e. the last frame index of the first scene and
    the first frame index of the second scene)
    """
    grayscale_video_array = np.array(media.read_video(video_path, output_format='gray'))
    video_histograms = np.stack([np.histogram(frame.flatten(), bins=256, range=(0, 256))[0] for frame in grayscale_video_array])

    # generate S0, ..., Sn-1
    video_cum_histograms = video_histograms.cumsum(axis=1)

    # D[i] = S[i + 1] - S[i]
    frames_diff = np.diff(video_cum_histograms, axis=0)

    p_norm = 1

    # calc ||D[i]||p for a given p >= 1
    frame_norm = np.round(np.linalg.norm(frames_diff, ord=p_norm, axis=1), decimals=3)
    last_frame_before_cut = frame_norm.argmax()

    grayscale_video_normalized = grayscale_video_array / NORMALIZATION_VALUE
    create_plots_for_features(grayscale_video_normalized, video_type, frame_norm)
    create_plots_based_histogram(frame_norm, p_norm)

    # first scene = index of last frame on the first scene + 1
    return last_frame_before_cut, last_frame_before_cut + 1


def create_plots_based_histogram(frame_norm, norm_type):
    indexes = range(len(frame_norm))
    df = pd.DataFrame(frame_norm, index=indexes)
    ax = df.plot(figsize=(8, 8))
    plt.xlabel('Difference Between Cumulative Sum of Adjacent Frames \nX_Axis_Val[i] = S[i+1] - S[i]')
    plt.ylabel(f'Norm {norm_type}')
    plt.title(f'Dist Function: Norm {norm_type}, Between Adjacent Frames', fontsize=11, weight='bold', color='blue')
    ax.legend().set_visible(False)
    plt.savefig('diff_frames.png')


def create_plots_for_features(grayscale_video_array, video_type, frame_norm):
    grayscale_mean = grayscale_video_array.mean(axis=(1, 2))
    grayscale_sum = grayscale_video_array.sum(axis=(1, 2))
    grayscale_std = grayscale_video_array.std(axis=(1, 2))
    grayscale_norm = frame_norm
    features_names = ['Mean', 'Sum', 'Std', 'Norm']
    colors = ['red', 'green', 'blue', 'orange']
    features_num = len(features_names)

    features_grayscale = [grayscale_mean, grayscale_sum, grayscale_std, grayscale_norm]

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    plt.suptitle('Frames Features, Grayscale Values: [0, 1]', fontsize=20, weight='bold', color='blue')

    for i, coordinate, name, feature in zip(range(features_num), product([0, 1], repeat=2),
                                            features_names, features_grayscale):
        row, col = coordinate
        axs[row, col].plot(feature, label=f'Video Category: {video_type}', color=colors[i])
        axs[row, col].set_title(f'Video 3 From Category {video_type}: {name}')
        axs[row, col].legend(['Grayscale'], loc='best')
        axs[row, col].set_xlabel('Frame Number')
        axs[row, col].set_ylabel(f'Grayscale {name} Value')

    plt.savefig('features_data.png')
