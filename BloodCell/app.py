import numpy as np
import cv2
from matplotlib import pyplot as plt
#Python imaging library
from PIL import Image


class ImageProcessor:

    RGB_SCALE = 255
    CMYK_SCALE = 0
    WBC_MIN_AREA=1500
    PLATES_MIN_AREA=50

    def __init__(self):
        pass

    def  show_image(image):
        cv2.imshow("output", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_images_multiple(image1, image2, title1, title2):
        cv2.imshow(title1, image1)
        cv2.imshow(title2, image2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def k_means(image, k):
        Z = image.reshape((-1, 3))
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape(image.shape)
        return res2

    def rgb_to_grayscale(image):
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return img

    def lab(image):
        im = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        return im

    def yCrCb(image):
        im = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        return im

    def hsv(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        return hsv

    def hls(image):
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        return hls

    def luv(image):
        im = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        return im

    def yuv(image):
        im = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        return im

    def xyz(image):
        im = cv2.cvtColor(image, cv2.COLOR_RGB2XYZ)
        return im

    def bgr(self, image):
        im = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.show_image(im)

    def red_channel(self, image):
        im = image.copy()
        im[:, :, 0] = 0
        im[:, :, 2] = 0
        self.show_image(im)

    def max_rgb_filter(self, image):
        # split the image into its BGR components
        (B, G, R) = cv2.split(image)

        # find the maximum pixel intensity values for each
        # (x, y)-coordinate,, then set all pixel values less
        # than M to zero
        M = np.maximum(np.maximum(R, G), B)
        R[R < M] = 0
        G[G < M] = 0
        B[B < M] = 0

        # merge the channels back together and return the image
        im = cv2.merge([B, G, R])
        self.show_image(im)

    def red_intensity(self, image):
        b, g, r = cv2.split(image)
        numpy_vertical_concat = np.hstack(cv2.split(image))
        self.show_image(numpy_vertical_concat)

    def color_thresholding(self, image):
        im = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(im, (110, 25, 25), (180, 255, 255))

        imask = mask > 0
        green = np.zeros_like(img, np.uint8)
        green[imask] = img[imask]

        return green

    def adjust_gamma(image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        im_lut = cv2.LUT(image, table)
        return im_lut

    def rgb_to_cmyk_separate_channels(imPath):
        im = Image.open(imPath).convert('CMYK')
        np_image = np.array(im)
        c = np_image[:, :, 0]
        m = np_image[:, :, 1]
        y = np_image[:, :, 2]
        k = np_image[:, :, 3]
        return (c, m ,y, k)

    def rgb_to_cmyk_fromPath(imPath):
        im= Image.open(imPath).convert('CMYK')
        np_image = np.array(im)
        result=np_image[:, :, :]
        return result

    def rgb_to_cmyk(image):
        img = Image.fromarray(image, 'RGB')
        im= img.convert('CMYK')
        np_image = np.array(im)
        result=np_image[:, :, :]
        return result

    def dilate(image, kernelSize, iterations):
        kernel = np.ones((kernelSize, kernelSize), np.uint8)
        dilation = cv2.dilate(image, kernel, iterations)
        return dilation

    def all_channel_enhance(im, kernel):
        for channel in im:
            img_blur=cv2.medianBlur(channel,5)
            img_blackhat = cv2.morphologyEx(img_blur, cv2.MORPH_BLACKHAT, kernel)
            img_tophat = cv2.morphologyEx(img_blur, cv2.MORPH_TOPHAT, kernel)
            channel = img_blur + img_blackhat - img_tophat
        return im

    def color_histogram_eq_separate_channels(im):
        b, g, r = cv2.split(img)
        red = cv2.equalizeHist(r)
        green = cv2.equalizeHist(g)
        blue = cv2.equalizeHist(b)
        return cv2.merge((blue, green, red))

    def lut_transform(img):
        xp = [0, 64, 128, 192, 255]
        fp = [0, 16, 128, 240, 255]
        x = np.arange(256)
        table = np.interp(x, xp, fp).astype('uint8')
        lut_img = cv2.LUT(img, table)
        return lut_img

    def find_nucleus(img, channel, k):
        blur = cv2.GaussianBlur(img[:, :, channel],(9,9),1)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)

        kernel = np.zeros((11, 11), np.uint8)
        imMorph = th3.copy()
        cv2.dilate(imMorph,kernel,None,None,10)
        cv2.morphologyEx(imMorph, cv2.MORPH_OPEN, kernel,None,None,5)
        cv2.morphologyEx(imMorph, cv2.MORPH_CLOSE, kernel,None,None,5)
        return imMorph

    def plotHistogram(image):
        histg = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.plot(histg)
        plt.show()

    def contouring(imOriginal, thresholded):
        contours, hierarchy = cv2.findContours(thresholded,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contoursSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
        del contoursSorted[-1]

        wbclist = [item for item in contoursSorted if cv2.contourArea(item) > ImageProcessor.WBC_MIN_AREA]
        platelist = [item for item in contoursSorted if cv2.contourArea(item) > ImageProcessor.PLATES_MIN_AREA and cv2.contourArea(item) < ImageProcessor.WBC_MIN_AREA]
        approxed_wbcList=ImageProcessor.approxPoly(wbclist,0.015);
        approxed_plateList=ImageProcessor.approxPoly(platelist,0.01);

        cv2.drawContours(imOriginal, approxed_wbcList,-1, (0,0,255), 3,cv2.LINE_4)
        cv2.drawContours(imOriginal, approxed_plateList, -1, (0, 255, 0), 3, cv2.LINE_4)

        return len(approxed_wbcList),len(approxed_plateList)

    def approxPoly(contourList, threshold):
        approxed = []
        for item in contourList:
            epsilon = threshold * cv2.arcLength(item, True)
            approx = cv2.approxPolyDP(item, epsilon, True)
            approxed.append(approx)
        return approxed

    def drawText(img, text, x, y, color):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,5, color ,6,cv2.LINE_8)

cv2.namedWindow("output", cv2.WINDOW_KEEPRATIO)
resPath="resources/IMG_3643.jpg"
img = cv2.imread(resPath)

img_lut=ImageProcessor.lut_transform(img)
imCmyk=ImageProcessor.rgb_to_cmyk(img_lut)

imtresh=ImageProcessor.find_nucleus(imCmyk,1,2)
wbcCount,platesCount=ImageProcessor.contouring(img, imtresh)
ImageProcessor.drawText(img,'Platelet: '+str(platesCount),100,200,[0,255,0])
ImageProcessor.drawText(img,'WBC: '+str(wbcCount),100,400,[0,0,255])

ImageProcessor.show_image(img)

#########################################
# im1=ImageProcessor.lab(img)
# im2=ImageProcessor.xyz(img)
# im3=ImageProcessor.hsv(img) #[0]: wcb mag és rbc de nincs cyto; [1]: CCA; [2]: nincs wbc mag
# im4=ImageProcessor.hls(img) #[0]: wcb mag és rbc de nincs cyto; [1]: nincs wbc mag; [2]: CCA
# im5=ImageProcessor.ycb(img)
# im6=ImageProcessor.luv(img)
# im7=ImageProcessor.yuv(img)
#
# # WCB szempontjából jó csatornák:
# # [0] : hsv, hls, ycb, luv, yuv
# # [1] : lab, xyz, hsv, ycb, luv, cmyk
# # [2] : hsv, hls, yuv
#################################################

# normalize float versions
#norm_img1 = cv2.normalize(img[:, :, channel], None, alpha=0, beta=300, norm_type=cv2.NORM_MINMAX, dtype=-1)
# norm_img1 = cv2.normalize(img[:, :, channel], None, alpha=0, beta=1.2, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#norm_img1 = (255 * norm_img1).astype(np.uint8)


