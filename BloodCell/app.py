import numpy as np
import cv2


flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print(flags)

class Test:

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

    def k_means(self, image, k):
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

    def rgb_to_grayscale(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.show_image(img)

    def lab(self, image):
        im = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        self.show_image(im)

    def ycb(self, image):
        im = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        self.show_image(im)

    def hsv_hls(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        self.show_images_multiple(hsv,hls, "HSV", "HLS")

    def luv(self, image):
        im = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        self.show_image(im)

    def yuv(self, image):
        im = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        self.show_image(im)

    def xyz(self, image):
        im = cv2.cvtColor(image, cv2.COLOR_RGB2XYZ)
        self.show_image(im)

    def bgr(self, image):
        im = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.show_image(im)

    def red_channel(self, image):
        im=image.copy()
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
        im= cv2.merge([B, G, R])
        self.show_image(im)

    def red_intensity(self, image):
        b,g,r=cv2.split(image)
        numpy_vertical_concat = np.hstack(cv2.split(image))
        self.show_image(numpy_vertical_concat)

    def color_thresholding(self, image):
        im=cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(im, (110, 25, 25), (180, 255, 255))

        imask = mask > 0
        green = np.zeros_like(img, np.uint8)
        green[imask] = img[imask]

        return green

    def adjust_gamma(self, image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        im_lut= cv2.LUT(image, table)
        self.show_image(im_lut)

    def main_test(self,image):
        im = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        #cv2.medianBlur(im,7,im)
        #cv2.equalizeHist(im,im)
        #im_th,th=cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.show_image(im)

cv2.namedWindow("output", cv2.WINDOW_NORMAL)
img = cv2.imread("resources/IMG_3643.jpg")


kmeans=Test.k_means(Test,img,5);
result=Test.color_thresholding(Test,kmeans)
Test.show_image(result)

#Test.show_images_multiple(kmeans,kmeans2,"output","output2")


