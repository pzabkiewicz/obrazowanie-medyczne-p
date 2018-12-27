import cv2
import numpy as np
from skimage.morphology import disk, opening


def get_grayscale_img(rgb_img):
    ''' Creating gray-scale image according following formula:
        c_r*R + c_g * G + c_b * B,
        where c_r = 0.1, c_g = 0.7, c_b = 0.2 '''

    img_gray = np.empty(rgb_img.shape[:2], dtype=np.uint8)
    for i in range(rgb_img.shape[0]):
        for j in range(rgb_img.shape[1]):
            img_gray[i, j] = np.uint8(0.1 * rgb_img[i, j, 0]) + np.uint8(0.7 * rgb_img[i, j, 1]) + np.uint8(
                0.2 * rgb_img[i, j, 2])

    return img_gray

def remove_light_reflex_from_vessels(grayscale_img):
    ''' This operation should remove vessel light reflex '''
    opened_img = opening(grayscale_img, disk(3))
    return opened_img

def get_gabor_filter_response(img, orientations=24, base_angle=15, ksize=3):

    ''' Getting Gabor filter response as feature for classification.
    Here is used gabor filter in different orientations (e.g. every 'base_angle')
    and then it will be extracted and returned maximum response from all expositions

    :param
        img: grayscale image, i.e. 2D-matrix
    '''

    gabor_result_img = np.zeros_like(img)
    for i in range(orientations):
        theta = base_angle * (i + 1)
        gabor_kern = cv2.getGaborKernel(ksize=(ksize, ksize), sigma=2, theta=theta, lambd=10, gamma=0.5)
        gabor_img = cv2.filter2D(img, -1, gabor_kern)
        np.maximum(gabor_result_img, gabor_img, gabor_result_img)  # getting maximum response

    return gabor_result_img

def get_binary_image_according_kmeans_result(img):


    ''' K is set to 3 as pixels with high, medium and low probability
     of vessel occurance. Then binary image will be created from result
     of kmeans algorithm, where pixels with low probability will be treated as vessels

     :param
        img: grayscale image (i.e. 2D-matrix) prepared (treated by different filters,
        morphological operations etc. for better contrast enhancement) for kmeans algorithm
     '''

    # grayscale means 1 feature, i.e. column vector has to be created
    z = np.reshape(img, (img.shape[0] * img.shape[1], 1))
    z = np.float32(z)

    # computation ends after accuracy threshold will be achieved (in this case 1.0)
    criteria = (cv2.TERM_CRITERIA_EPS, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # Apply KMeans
    compactness, labels, centers = cv2.kmeans(z, 3, None, criteria, 10, flags)

    uniqueVals, counts = np.unique(labels, return_counts=True)
    whiteLabel = np.argmin(counts)
    blackLabel = np.argmax(counts)
    grayLabel = [i for i in range(3) if i != whiteLabel and i != blackLabel]

    # reconstructing binary image
    centers = np.uint8(centers)
    labels[labels == whiteLabel] = grayLabel[0]
    res = centers[labels.flatten()]
    result = res.reshape((img.shape))
    a, b = np.unique(result, return_counts=True)
    result[result == a[np.argmax(b)]] = 0
    result[result == a[np.argmin(b)]] = 255

    return result

def get_confusion_matrix_stats(img, groundtruth_img):

    ''' This function returns performance statistics, i.e. computes
     sensitivity, specificity and accuracy based on TP, TN, FP and FN.

     TP - vessel pixels correctly classified as vessels
     TN - background pixels correctly classified as background
     FP - background pixels incorrectly classified as vessels
     FN - vessel pixels incorrectly classified as background '''

    tp = 0; tn = 0; fp = 0; fn = 0

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            px = img[i][j]

            if px == 0:
                if groundtruth_img[i, j, 1] == 0:
                    tn = tn + 1
                elif groundtruth_img[i, j, 1] == 255:
                    fn = fn + 1
            elif px == 255:
                if groundtruth_img[i, j, 1] == 0:
                    fp = fp + 1
                elif groundtruth_img[i, j, 1] == 255:
                    tp = tp + 1

    sen = tp / (tp + fn) * 100  # sensitivity
    spec = tn / (tn + fp) * 100  # specificity
    acc = (tp + tn) / (tp + tn + fp + fn) * 100  # accuracy

    return {'sensitivity': sen, 'specificity':spec, 'accuracy':acc}