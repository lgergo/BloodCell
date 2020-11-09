import numpy as np
import cv2
from matplotlib import pyplot as plt
#Python imaging library
from PIL import Image
from skimage.segmentation import clear_border
from skimage.feature import peak_local_max
from skimage import data, img_as_float
from scipy import ndimage as ndi
from scipy.ndimage.measurements import center_of_mass, label


class ImageProcessor:

    RGB_SCALE = 255
    CMYK_SCALE = 0
    WBC_MIN_AREA=1500
    PLATES_MIN_AREA=20

    def __init__(self):
        pass

    def  show_image(image, name):
        cv2.namedWindow(name,cv2.WINDOW_NORMAL)
        cv2.imshow(name, image)

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
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 10.0)
        ret, label, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # Now convert back into uint8, and make original image
        centers = np.uint8(centers)
        res = centers[label.flatten()]
        res2 = res.reshape(image.shape)
        #print('Dominant color is: bgr({})'.format(centers[0].astype(np.int32)))
        return res2, centers

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
        xp = [0, 64, 128, 192, 255]  #lut_in
        fp = [0, 16, 128, 240, 255]  #lut_out
        x = np.arange(256)
        table = np.interp(x, xp, fp).astype('uint8')
        lut_img = cv2.LUT(img, table)
        return lut_img

    def plotHistogram(image):
        im=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        plt.hist(im.ravel(),256,[0,256]);
        plt.show()

    def find_nucleus(img, channel):
        blur = cv2.GaussianBlur(img[:, :, channel],(9,9),1)
        ret3, th3 = cv2.threshold(blur, 151, 255, cv2.THRESH_BINARY)

        kernel = np.zeros((11, 11), np.uint8)
        imMorph = th3.copy()
        cv2.dilate(imMorph,kernel,None,None,10)
        cv2.morphologyEx(imMorph, cv2.MORPH_OPEN, kernel,None,None,5)
        cv2.morphologyEx(imMorph, cv2.MORPH_CLOSE, kernel,None,None,5)
        return imMorph

    def approxPoly(contourList, threshold):
        approxed = []
        for item in contourList:
            epsilon = threshold * cv2.arcLength(item, True)
            approx = cv2.approxPolyDP(item, epsilon, True)
            approxed.append(approx)
        return approxed

    def drawText(img, text, x, color):
        h,w,c=img.shape
        cv2.putText(img, text, (x, h-90), cv2.FONT_HERSHEY_SIMPLEX,3, color ,6,cv2.LINE_8)

    def extendImage(img):
        h,w,c=img.shape
        modified_height = 200
        black = np.zeros((modified_height, h, 3), np.uint8)
        final = np.vstack((img, black))
        return final

    def enhance_cyto(img):
        kernel = np.ones((9, 9), np.uint8)
        imBottomHat=cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel, iterations=1)
        imTopHat=cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel,iterations=1)
        imRes=img + imBottomHat-imTopHat
        return imRes

    def dm4_kmeans_rbc(img):
        blur = cv2.GaussianBlur(img, (9, 9), 1)
        img_lut = ImageProcessor.lut_transform(blur)
        kmeans, centers = ImageProcessor.k_means(img_lut, 4)
        centers_sorted = centers[centers[:, 2].argsort()]  # szín szerint sorba rendezem
        # 0 - sejtmag
        # 1 - wbc
        # 2 - kontúr
        # 3 - háttér
        wbcPlateletMask=cv2.inRange(kmeans,centers_sorted[0],centers_sorted[0])
        rbcMask=cv2.inRange(kmeans,centers_sorted[1], centers_sorted[2])

        (wbcContours,plateletContours) = ImageProcessor.getContourListOfWbcPlatelets(img,wbcPlateletMask)
        lists=np.concatenate((wbcContours,plateletContours))
        countRbc,rbcImage=ImageProcessor.contourRbc(kmeans, rbcMask, lists)
        img[rbcImage == 255] = (0, 255, 255)

        imExt=ImageProcessor.extendImage(img)
        cv2.drawContours(imExt, wbcContours, -1, (0, 0, 255), 3, cv2.LINE_4)
        cv2.drawContours(imExt, plateletContours, -1, (0, 255, 0), 3, cv2.LINE_4)
        ImageProcessor.drawText(imExt, 'WBC:'+str(len(wbcContours)), 100, [0, 0, 255])
        ImageProcessor.drawText(imExt, 'RBC: '+str(countRbc), 600, [0, 255, 255])
        ImageProcessor.drawText(imExt, 'Platelet:'+str(len(plateletContours)), 1300, [0, 255, 0])
        ImageProcessor.show_image(imExt,"result")

    def getContourListOfWbcPlatelets(img,imgThres):
        kernel=np.ones((9,9),np.uint8)
        cv2.dilate(imgThres, kernel, iterations=5)
        contours, hierarchy = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contoursSorted = sorted(contours, key=lambda x: cv2.contourArea(x))

        wbclist = [item for item in contoursSorted if cv2.contourArea(item) > ImageProcessor.WBC_MIN_AREA]
        platelist = [item for item in contoursSorted if
                     cv2.contourArea(item) > ImageProcessor.PLATES_MIN_AREA and cv2.contourArea(
                         item) < ImageProcessor.WBC_MIN_AREA]
        approxed_wbcList = ImageProcessor.approxPoly(wbclist, 0.015)
        approxed_plateList = ImageProcessor.approxPoly(platelist, 0.01)

        return approxed_wbcList, approxed_plateList

    def contourRbc(kmeansImage, rbcMask, wbcAndPlateletContours):
        contours, hierarchy = cv2.findContours(rbcMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = ImageProcessor.approxPoly(contours, 0.025)
        contoured = rbcMask
        for cnt in contours:
            cv2.drawContours(contoured, [cnt], 0, 255, -1)  # filling holes
        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        rbcMask = cv2.morphologyEx(rbcMask, cv2.MORPH_OPEN, kernel, iterations=2)
        rbcMask = cv2.morphologyEx(rbcMask, cv2.MORPH_CLOSE, kernel, iterations=2)

        sure_bg = cv2.dilate(rbcMask, kernel, iterations=3)
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(rbcMask, cv2.DIST_L2, cv2.DIST_MASK_3)

        # Comparison between image_max and im to find the coordinates of local maxima
        coordinates = peak_local_max(dist_transform, min_distance=35)

        # Find peaks and merge equal regions; results in two peaks
        # is_peak = peak_local_max(dist_transform, indices=False)  # outputs bool image
        # labels = label(is_peak)[0]
        # merged_peaks = center_of_mass(is_peak, labels, range(1, np.max(labels) + 1))
        # merged_peaks = np.array(merged_peaks)
        # merged_peaks=merged_peaks.astype(int)

        maximumPoints=np.zeros(dist_transform.shape)
        for point in coordinates:
            for contour in wbcAndPlateletContours:
                if cv2.pointPolygonTest(contour, (point[1],point[0]), True)<0:
                    cv2.circle(maximumPoints, (point[1],point[0]),1, (255, 255, 255),thickness=-1)

        #sure_fg = clear_border(sure_fg)
        sure_fg = np.uint8(maximumPoints)
        # Finding unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        markers = cv2.watershed(img, markers)
        # img[markers == -1] = [0, 0, 255]

        markers[markers == -1] = 0
        lbl = markers.astype(np.uint8)
        lbl2 = 255 - lbl
        lbl2[lbl2 != 255] = 0
        lbl2 = cv2.dilate(lbl2, None)
        #img[lbl2 == 255] = (255, 0, 0)

        return ret,lbl2

#---------------------------------------------------
resPath="resources/IMG_3643_rect.jpg"
img = cv2.imread(resPath)

ImageProcessor.dm4_kmeans_rbc(img)

#TODO bounding rect el kiszedni az adott WBC-t és beadni az AInak
#TODO https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html

cv2.waitKey(0)
cv2.destroyAllWindows()
