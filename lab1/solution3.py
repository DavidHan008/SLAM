from PIL import Image as Image
import numpy as np
import matplotlib.pyplot as plt


def RGB2Bright(input):
    # using YUV space to convert
    output = np.zeros_like(input[:, :, 0])
    output[:, :] = input[:, :, 0] * 0.299 + \
        input[:, :, 1] * 0.587 + input[:, :, 2] * 0.114
    output = output.astype(np.uint8)

    return output


def create_Cumulative(input):
    '''
    create cdf
    :param
    input: input pictures
    :return: cdf, dtype is int
    '''
    # using 8-bit
    cumulative = np.zeros([256])
    height, width = input.shape
    for x in range(height):
        for y in range(width):
            for gray in range(input[x, y], 256, 1):
                cumulative[gray] += 1

    #cumulative /= float(input.shape[0] * input.shape[1])

    return cumulative


def create_histogram(input):
    # using 8-bit
    histogram = np.zeros([256])
    height, width = input.shape
    for x in range(height):
        for y in range(width):
            histogram[input[x, y]] += 1

    #histogram /= float(input.shape[0] * input.shape[1])

    return histogram


def Histogram_specialization(sourceDir, targetDir, outputDir, downsample_rate):
    '''
    the main function
    input can be colored or gray pictures
    downsampling for original colored pictures only
    '''

    # images need to be downsampled, the original ones would take too much time
    source = np.array(Image.open(sourceDir))

    target = np.array(Image.open(targetDir))

    # mapping source into bright
    if source.ndim == 3:
        source = RGB2Bright(source)
        source = Image.fromarray(source)
        source = np.array(
            source.resize(
                (int(
                    source.size[0] /
                    downsample_rate),
                    int(
                    source.size[1] /
                    downsample_rate)),
                Image.BICUBIC))

    # mapping target into bright
    if target.ndim == 3:
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

    # get cumulative matrix
    sourceCumulative = create_Cumulative(source) / source.size
    targetCumulative = create_Cumulative(target) / target.size

    LookUpTable = np.zeros([256])

    j = 0
    for i in range(256):
        while (abs(sourceCumulative[i] - targetCumulative[j]) >=
               abs(sourceCumulative[i] - targetCumulative[j + 1])):
            j += 1
            if j == 255:
                break

        LookUpTable[i] = j
        if j == 255 or i == 255:
            for k in range(i, 256):
                LookUpTable[k] = j
            break

    output = np.zeros_like(source)
    for x in range(source.shape[0]):
        for y in range(source.shape[1]):
            output[x, y] = LookUpTable[source[x, y]]

    output = Image.fromarray(output)
    output.save(outputDir)

    '''
    # testing...
    downsampledDir = './/output//middle_downsample.jpg'
    downTargetDir = './/output//middle_target_downsample.jpg'
    source = Image.fromarray(source)
    target = Image.fromarray(target)
    source.save(downsampledDir)
    target.save(downTargetDir)
    '''


def display(sourceDir, targetDir, resultDir, downsample_rate):
    '''
    Display the images and their histograms.
    Works independently with Histogram_specialization.
    '''

    # images need to be downsampled, the original ones would take too much time
    source = np.array(Image.open(sourceDir))

    target = np.array(Image.open(targetDir))

    result = np.array(Image.open(resultDir))

    # mapping source into bright
    source = RGB2Bright(source)
    source = Image.fromarray(source)
    source = np.array(
        source.resize(
            (int(
                source.size[0] /
                downsample_rate),
             int(
                 source.size[1] /
                 downsample_rate)),
            Image.BICUBIC))

    # mapping target into bright
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

    # get (1,) array to apply to funtion 'hist'
    targetArr = target.flatten()
    sourceArr = source.flatten()
    resultArr = result.flatten()

    # plot all of them out
    fig = plt.figure(figsize=(19.2, 10.8))
    gs = plt.GridSpec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1], sharex=ax4, sharey=ax4)
    ax6 = fig.add_subplot(gs[1, 2], sharex=ax4, sharey=ax4)

    ax1.imshow(source, cmap=plt.cm.gray)
    ax1.set_title('Source')
    ax2.imshow(result, cmap=plt.cm.gray)
    ax2.set_title('Result')
    ax3.imshow(target, cmap=plt.cm.gray)
    ax3.set_title('Target')

    ax4.hist(
        sourceArr,
        bins=20,
        density=1,
        facecolor="blue",
        edgecolor="black",
        alpha=0.75)
    ax4.set_xlabel("gray")
    ax4.set_ylabel("frequency")
    ax4.set_title("source_histogram")

    ax5.hist(
        resultArr,
        bins=20,
        density=1,
        facecolor="blue",
        edgecolor="black",
        alpha=0.75)
    ax5.set_xlabel("gray")
    ax5.set_ylabel("frequency")
    ax5.set_title("result_histogram")

    ax6.hist(
        targetArr,
        bins=20,
        density=1,
        facecolor="blue",
        edgecolor="black",
        alpha=0.75)
    ax6.set_xlabel("gray")
    ax6.set_ylabel("frequency")
    ax6.set_title("target_histogram")

    plt.savefig('.//latex//3_6.jpg')
    plt.show()

    return

def exportDownsample(inputDir, outputDir, downsample_rate):
    input = np.array(Image.open(inputDir))

    input = RGB2Bright(input)
    input = Image.fromarray(input)
    output = np.array(
        input.resize(
            (int(
                input.size[0] /
                downsample_rate),
             int(
                 input.size[1] /
                 downsample_rate)),
            Image.BICUBIC))

    output = Image.fromarray(output)
    output.save(outputDir)

if __name__ == "__main__":
    sourceDir = './/source//left_bright.jpg'
    targetDir = './/source//left_bright.jpg'
    outputDir = './/output//left_bright_downsampled.jpg'
    downsample_rate = 8

    #exportDownsample(sourceDir, outputDir, downsample_rate)

    # the output is in outputDir
    Histogram_specialization(sourceDir, targetDir, outputDir, downsample_rate)

    # optional
    display(sourceDir, targetDir, outputDir, downsample_rate)


