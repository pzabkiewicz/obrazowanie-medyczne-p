from os import listdir
from utils import *
import cv2

######################### K-MEANS SCRIPT #######################################

basepath_training = '../DRIVE/training/images/'
basepath_groundtruth = '../DRIVE/training/1st_manual/'

training_files = listdir(basepath_training)
groundtruth_files = listdir(basepath_groundtruth)
files_count = len(training_files)

sensitivity = 0
specificity = 0
accuracy = 0

it = 1
for img_paths in zip(training_files, groundtruth_files):

    # segmentation of original image
    img = cv2.imread(basepath_training + img_paths[0], 1)
    img_shape = img.shape[:2]

    img_gray = get_grayscale_img(img)
    opened_img = remove_light_reflex_from_vessels(img_gray)
    gabor_filter_img = get_gabor_filter_response(opened_img)
    tophat_img = cv2.morphologyEx(gabor_filter_img, cv2.MORPH_TOPHAT, disk(5))
    result = get_binary_image_according_kmeans_result(tophat_img)
    struct = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    result = cv2.dilate(result, struct, iterations=1)
    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, struct)

    # ground truth
    cap = cv2.VideoCapture(basepath_groundtruth + img_paths[1])
    ret, frame = cap.read()

    # getting stats
    stats = get_confusion_matrix_stats(result, frame)

    sensitivity = sensitivity + stats['sensitivity']
    specificity = specificity + stats['specificity']
    accuracy = accuracy + stats['accuracy']
    print('Progress: ' + (it * '|') + ((files_count-it) * '.') + ' %.2f' % (it / files_count * 100) + '%')
    it = it + 1

print('sen: %d, spec: %d, acc: %d' % (sensitivity / files_count, \
            specificity / files_count, \
            accuracy / files_count))
