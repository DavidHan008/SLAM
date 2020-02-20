from PIL import Image as Image
import numpy as np
from solution3 import RGB2Bright


def SSD(inputDir, domainDir, outputDir, downsample_rate):

    target = np.array(Image.open(inputDir))
    domain = np.array(Image.open(domainDir))

    if target.ndim == 3:
        target = target[:, :, 0]

    '''
    target = RGB2Bright(target)
    target = Image.fromarray(target)
    target = np.array(
        target.resize(
            (int(
                target.size[0] /
                downsample_rate),
                int(
                target.size[1] /
                downsample_rate)),
            Image.BICUBIC))
    '''

    if domain.ndim == 3:
        domain = RGB2Bright(domain)
        domain = Image.fromarray(domain)
        domain = np.array(
            domain.resize(
                (int(
                    domain.size[0] /
                    downsample_rate),
                    int(
                    domain.size[1] /
                    downsample_rate)),
                Image.BICUBIC))

    ht, wt = target.shape
    hd, wd = domain.shape
    output = np.zeros_like(target)

    number = max(np.max(target), np.max(domain))

    domain = domain / number
    target = target / number

    xmin = 0
    ymin = 0
    reverse_ssd = 0

    for x in range(0, hd - ht):
        for y in range(0, wd - wt):
            tmp = np.linalg.norm(target - domain[x:x + ht, y:y + wt])
            if 1 / tmp >= reverse_ssd:
                xmin = x
                ymin = y
                reverse_ssd = 1 / tmp

    output[:, :] = domain[xmin:xmin + ht, ymin: ymin + wt] * number

    output = Image.fromarray(output)
    output.save(outputDir)


def ZNCC(inputDir, domainDir, outputDir, downsample_rate):

    target = np.array(Image.open(inputDir))
    domain = np.array(Image.open(domainDir))

    if target.ndim == 3:
        target = target[:, :, 0]

    '''
    target = RGB2Bright(target)
    target = Image.fromarray(target)
    target = np.array(
        target.resize(
            (int(
                target.size[0] /
                downsample_rate),
                int(
                target.size[1] /
                downsample_rate)),
            Image.BICUBIC))
    '''

    if domain.ndim == 3:
        domain = RGB2Bright(domain)
        domain = Image.fromarray(domain)
        domain = np.array(
            domain.resize(
                (int(
                    domain.size[0] /
                    downsample_rate),
                    int(
                    domain.size[1] /
                    downsample_rate)),
                Image.BICUBIC))

    ht, wt = target.shape
    hd, wd = domain.shape
    output = np.zeros_like(target)

    a = target.flatten()
    a = a - np.mean(a)
    a = a / np.linalg.norm(a)
    #domain1 = domain - np.mean(domain)

    xmin = 0
    ymin = 0
    zncc = -1   # zncc ranges in [-1,1]

    for x in range(0, hd - ht):
        for y in range(0, wd - wt):
            b = domain[x: x + ht, y: y + wt]
            b = b - np.mean(b)
            b = b.flatten()
            if np.linalg.norm(b) != 0:
                b = b / np.linalg.norm(b)
            tmp = np.sum(a * b)
            if tmp >= zncc:
                xmin = x
                ymin = y
                zncc = tmp

    output[:, :] = domain[xmin: xmin + ht, ymin: ymin + wt]

    output = Image.fromarray(output)
    output.save(outputDir)


if __name__ == "__main__":

    # Note that target can only be a gray picture, that is already downsampled!
    # I offered 3 usable gray target pictures in .//title//
    inputDir = './/source//middle2right_bright_title.jpg'
    domainDir = './/output//right_bright_downsampled.jpg'
    outputDir = './/latex//test2.jpg'
    downsample_rate = 8


    ZNCC(inputDir, domainDir, outputDir, downsample_rate)
