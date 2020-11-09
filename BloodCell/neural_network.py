# import tensorflow as tf
#
# Categories=["EOSINOPHIL","LYMPHOCYTE","MONOCYTE","NEUTROPHIL"]
#
# model=tf.keras.models.load_model("wbcClassif_1.model")
#
# image_array=cv2.imread("drive/My Drive/msc/eosinophil.jpeg")
# image_resized=cv2.resize(image_array,(320,240)).reshape(1,240,320,-1)
#
# prediction = model.predict([image_resized])
#
# print(prediction)
# print(Categories)