import matplotlib.pyplot as plt
import cv2

def get_sample_img(dl, b=0, i=0):
    img = next(iter(dl))['image'][b][i]
    return img