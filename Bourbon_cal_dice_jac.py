import cv2
import numpy as np
import os

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

path = "dataset4//calculate_dice"
def rgb_to_grayscale(image):
    r, g, b = image[0], image[1], image[2]
    grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return grayscale
def calculate_dice(path):
    path = path
    vessels = os.listdir(path)
    nums = len(vessels)
    print(vessels)
    means = 0.
    dices = []
    for i, vessel in enumerate(vessels):
        img1 = cv2.imread(os.path.join(path, vessel, 'p', '1.1.png'))
        img2 = cv2.imread(os.path.join(path, vessel, 'l', 'predict_1.1.png'))
        _, img1 = cv2.threshold(
            cv2.cvtColor(img1.copy(), cv2.COLOR_BGR2GRAY),
            10, 255,
            cv2.THRESH_BINARY)
        _, img2 = cv2.threshold(
            cv2.cvtColor(img2.copy(), cv2.COLOR_BGR2GRAY),
            10, 255,
            cv2.THRESH_BINARY)
        img11 = np.where(img1 > 0, 1, 0)
        img22 = np.where(img2 > 0, 1, 0)
        dice = dice_coef(img11, img22)
        means += dice
        dices.append("{} dice: {}".format(vessel, round(dice, 4)))
    print("means dices: ", means / nums)
    print(dices)

def calculate_F1_score(path):
    path = path

    vessels = os.listdir(path)
    nums = len(vessels)
    means = 0.
    F1_scores = []
    for i, vessel in enumerate(vessels):
        img1 = cv2.imread(os.path.join(path, vessel, 'p', '1.1.png'),0 ,cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(os.path.join(path, vessel, 'l', 'predict_1.1.png'), 0,cv2.IMREAD_GRAYSCALE)
        img11 = np.where(img1 > 0, 1, 0)
        img22 = np.where(img2 > 0, 1, 0)

        TP = img11 * img22
        FP = img11 - TP
        FN = img22 - TP
        precision = TP.sum() / (TP.sum() + FP.sum())
        recall = TP.sum() / (TP.sum() + FN.sum())
        F1 = (2 * precision * recall) / (precision + recall)
        means += F1
        F1_scores.append("{} F1_score: {}".format(vessel, round(F1, 4)))

    print("means F1_score: ", means / nums)
    print(F1_scores)


def calculate_jaccard(path):
    path = path

    vessels = os.listdir(path)
    nums = len(vessels)

    means = 0.
    jaccard_scores = []
    for i, vessel in enumerate(vessels):
        img1 = cv2.imread(os.path.join(path, vessel, 'p', '1.1.png'))
        img2 = cv2.imread(os.path.join(path, vessel, 'l', 'predict_1.1.png'))
        _, img1 = cv2.threshold(
            cv2.cvtColor(img1.copy(), cv2.COLOR_BGR2GRAY),
            10, 255,
            cv2.THRESH_BINARY)
        _, img2 = cv2.threshold(
            cv2.cvtColor(img2.copy(), cv2.COLOR_BGR2GRAY),
            10, 255,
            cv2.THRESH_BINARY)

        img11 = np.where(img1 > 0, 1, 0)
        img22 = np.where(img2 > 0, 1, 0)

        y_true_f = img11.flatten()
        y_pred_f = img22.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        jaccard_score = (intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection + smooth)

        means += jaccard_score
        jaccard_scores.append("{} jaccard_score: {}".format(vessel, round(jaccard_score, 8)))

    print("means jaccard_score: ", means / nums)
    print(jaccard_scores)


if __name__ == '__main__':
    calculate_dice("dataset//red_accuracy")
    calculate_jaccard("dataset//red_accuracy")
    calculate_dice("dataset//green_accuracy")
    calculate_jaccard("dataset//green_accuracy")
