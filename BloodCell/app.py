import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max

class ImageProcessor:
    WBC_MIN_AREA=1000
    RBC_MIN_AREA=1000
    PLATES_MIN_AREA=50
    CATEGORIES = ["neutrophil", "lymphocyte", "monocyte", "eosinophil"]
    TRAINED_MODEL_PATH="resources/trainedmodel/wbcClassif_1"
    DEMO_IMAGE_PATH= "resources/IMG_3643.JPG"

    def __init__(self):
        pass

    # <editor-fold desc="Testing methods">
    def red_intensity(image):
        b, g, r = cv2.split(image)
        numpy_vertical_concat = np.hstack(cv2.split(image))
        return numpy_vertical_concat

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
    # </editor-fold>

    # <editor-fold desc="Display and ui">
    def show_image(image, name):
        cv2.namedWindow(name,cv2.WINDOW_NORMAL)
        cv2.imshow(name, image)

    def plotHistogram(image):
        im=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        plt.hist(im.ravel(),256,[0,256])
        plt.show()

    def drawText(img, text, x, color):
        h, w, c = img.shape
        cv2.putText(img, text, (x, h - 90), cv2.FONT_HERSHEY_DUPLEX, 3, color, 6, cv2.LINE_8)

    def drawPredictForRoi(img, x, y, text):
        cv2.putText(img, text, (x, y - 20), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2, cv2.LINE_8)
    # </editor-fold>

    def rgb_to_cmyk(image):
        img = Image.fromarray(image, 'RGB')
        im= img.convert('CMYK')
        np_image = np.array(im)
        result=np_image[:, :, :]
        return result

    def lut_transform(img):
        xp = [0, 64, 128, 192, 255]  #lut_in
        fp = [0, 16, 128, 240, 255]  #lut_out
        x = np.arange(256)
        table = np.interp(x, xp, fp).astype('uint8')
        lut_img = cv2.LUT(img, table)
        return lut_img

    def k_means(image, k):
        Z = image.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 10.0)
        ret, label, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        res = centers[label.flatten()]
        res2 = res.reshape(image.shape)

        return res2, centers

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

    def contourRbc(kmeansImage, rbcMask):
        contours, hierarchy = cv2.findContours(rbcMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = ImageProcessor.approxPoly(contours, 0.005)
        contoursOnlyRbc = [item for item in contours if cv2.contourArea(item) > ImageProcessor.RBC_MIN_AREA]
        filledRbc = np.zeros((rbcMask.shape),np.uint8)
        for cnt in contoursOnlyRbc:
            cv2.drawContours(filledRbc, [cnt], 0, 255, -1)

        kernel = np.ones((5, 5), np.uint8)
        filledRbc = cv2.morphologyEx(filledRbc, cv2.MORPH_OPEN, kernel, iterations=2)
        filledRbc = cv2.morphologyEx(filledRbc, cv2.MORPH_CLOSE, kernel, iterations=2)

        sure_bg = cv2.dilate(filledRbc, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(filledRbc, cv2.DIST_L2, cv2.DIST_MASK_3)
        coordinates = peak_local_max(dist_transform, min_distance=25)

        maximumPoints=np.zeros(dist_transform.shape)
        for point in coordinates:
            cv2.circle(maximumPoints, (point[1],point[0]),15, (255, 255, 255),thickness=-1)

        sure_fg = np.uint8(maximumPoints)
        unknown = cv2.subtract(sure_bg, sure_fg)

        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        markers = cv2.watershed(img, markers)
        markers[markers == -1] = 0
        lbl = markers.astype(np.uint8)
        lbl2 = 255 - lbl
        lbl2[lbl2 != 255] = 0
        kernel=np.ones((3, 3), np.uint8)
        lbl2 = cv2.dilate(lbl2, kernel)

        return ret,lbl2

    def getBoundingRectsOfWbc(imgOriginal,img, wbcContours):
        model = tf.keras.models.load_model(ImageProcessor.TRAINED_MODEL_PATH)
        for contour in wbcContours:
            x, y, w, h = cv2.boundingRect(contour)
            if(w>h):
                h+=w-h
            else:
                w+=h-w
            shift=50
            roi = imgOriginal[max(y-shift,0):min(y + h+shift,2100), max(x-shift,0):min(x + w+shift,2100)]
            roiSerized=cv2.resize(roi,(250,250))

            cv2.rectangle(img,(x-shift,y-shift),(x+w+shift,y+h+shift),(200,0,0),2)
            predictedCategory=ImageProcessor.recognise(roiSerized, model)
            ImageProcessor.drawPredictForRoi(img,x-shift,y-shift,predictedCategory)

    def recognise(roi, model):
        roiReshaped=roi.reshape(1,250,250,-1)
        predict=model.predict([roiReshaped])
        return ImageProcessor.CATEGORIES[np.argmax(predict)]

    def bloocCellSegmentation(img):
        imgOriginal = img.copy()
        blur = cv2.GaussianBlur(img, (9, 9), 1)
        img_lut = ImageProcessor.lut_transform(blur)
        kmeans, centers = ImageProcessor.k_means(img_lut, 4)
        centers_sorted_bycolor = centers[centers[:, 2].argsort()]
        #0-wbc  #1-rbc 1   #2-rcb 2 contour  #3- background

        wbcPlateletMask=cv2.inRange(kmeans,centers_sorted_bycolor[0],centers_sorted_bycolor[0])
        rbcMask=cv2.inRange(kmeans,centers_sorted_bycolor[1],centers_sorted_bycolor[1])+cv2.inRange(kmeans, centers_sorted_bycolor[2],centers_sorted_bycolor[2])

        (wbcContours,plateletContours) = ImageProcessor.getContourListOfWbcPlatelets(img,wbcPlateletMask)
        countRbc,rbcImage=ImageProcessor.contourRbc(kmeans, rbcMask)
        img[rbcImage == 255] = (0, 255, 255)

        ImageProcessor.getBoundingRectsOfWbc(imgOriginal,img,wbcContours)

        imExt=ImageProcessor.extendImage(img)
        cv2.drawContours(imExt, wbcContours, -1, (0, 0, 255), 2, cv2.LINE_4)
        cv2.drawContours(imExt, plateletContours, -1, (0, 255, 0), 3, cv2.LINE_4)
        ImageProcessor.drawText(imExt, 'WBC:'+str(len(wbcContours)), 100, [0, 0, 255])
        ImageProcessor.drawText(imExt, 'RBC: '+str(countRbc), 600, [0, 255, 255])
        ImageProcessor.drawText(imExt, 'Platelet:'+str(len(plateletContours)), 1300, [0, 255, 0])
        ImageProcessor.show_image(imExt,"result")

img = cv2.imread(ImageProcessor.DEMO_IMAGE_PATH)
ImageProcessor.bloocCellSegmentation(img)

cv2.waitKey(0)
cv2.destroyAllWindows()
