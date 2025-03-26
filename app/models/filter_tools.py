import numpy as np

def smooth_keypoints(keypoints, window_size=5):
    smoothed = np.copy(keypoints)
    for i in range(keypoints.shape[1]):  # for each keypoint
        for j in range(3):  # x, y, z
            smoothed[:, i, j] = np.convolve(
                keypoints[:, i, j], 
                np.ones(window_size)/window_size, 
                mode='same'
            )
    return smoothed

def ema_keypoints(keypoints, alpha=0.2):
    smoothed = np.copy(keypoints)
    for i in range(1, keypoints.shape[0]):
        smoothed[i] = alpha * keypoints[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed