from typing import List

import numpy as np
import cv2

from misc.ev_eye_dataset_utils import ellipse_norm


class EllipseOutOfBoundException(BaseException):

    def __init__(self, *args):
        super().__init__(*args)


def shape_based_similarity(img: np.array, ellipse: List[float], gradient_ksize=5, ellipse_sample_points=48,
                           filter_min=False, filter_max=True,
                           single_abs=False, total_abs=False):
    """
    :param filter_max:
    :param filter_min:
    :param ellipse_sample_points:
    :param gradient_ksize:
    :param img:
    :param ellipse: [x,y,w,h,t]

    :return:
    """
    height, width = img.shape
    dx = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=gradient_ksize)
    dy = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=gradient_ksize)
    # sample 48 points
    sample_rads = np.arange(ellipse_sample_points) * np.pi * 2 / ellipse_sample_points
    orin_sample_rads, origin_sample_points, origin_norm, trans_points, trans_norm = ellipse_norm(ellipse, sample_rads)

    points_on_img = np.asarray(np.round(trans_points), np.int16)  # [x,y]
    if not (np.all(np.logical_and(0 <= points_on_img[:, 0], points_on_img[:, 0] < width))
            and np.all(np.logical_and(0 <= points_on_img[:, 1], points_on_img[:, 1] < height))):
        raise EllipseOutOfBoundException(
            f"ellipse out of img bound, get ellipse{ellipse},but with img shape {(height, width)}")

    sample_gradients = np.vstack([dx[points_on_img[:, 1], points_on_img[:, 0]],
                                  dy[points_on_img[:, 1], points_on_img[:, 0]]]).T  # [num,2]

    # normalized gradient
    sample_gradients_norm = np.linalg.norm(sample_gradients, axis=1, keepdims=True)
    valid_norm_mask = np.squeeze(sample_gradients_norm > 0)
    sample_gradients[valid_norm_mask, :] /= sample_gradients_norm[valid_norm_mask]
    sample_gradients[~valid_norm_mask, :] = 0

    gradient_filter_threshold = [sample_gradients_norm.mean(),
                                 sample_gradients_norm.mean() + 3.6 * sample_gradients_norm.std()]

    # filter gradients
    filter_mask = np.ones((ellipse_sample_points, 1), dtype=bool)
    if filter_min:
        filter_mask &= (sample_gradients_norm >= gradient_filter_threshold[0])
    if filter_max:
        filter_mask &= (sample_gradients_norm <= gradient_filter_threshold[1])
    filtered_norms = trans_norm[filter_mask[:, 0], :]
    filtered_gradients = sample_gradients[filter_mask[:, 0], :]

    if single_abs:
        similarity = np.mean(np.abs(np.sum(filtered_norms * filtered_gradients, axis=1, keepdims=True)))
    else:
        similarity = np.mean(np.sum(filtered_norms * filtered_gradients, axis=1, keepdims=True))

    if total_abs:
        return np.abs(similarity)
    else:
        return similarity
