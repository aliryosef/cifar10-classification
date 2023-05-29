#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\Users\aliry\Desktop\gui 1 window\Image Classification.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets 

import numpy as np

import cv2 
import tensorflow as tf
from tensorflow.keras.models import load_model
class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(665, 760)
        self.imagelabel = QtWidgets.QLabel(Dialog)
        self.imagelabel.setGeometry(QtCore.QRect(100, 190, 461, 371))
        self.imagelabel.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.imagelabel.setText("")
        self.imagelabel.setObjectName("imagelabel")
        self.imagebutton = QtWidgets.QPushButton(Dialog)
        self.imagebutton.setGeometry(QtCore.QRect(10, 70, 131, 41))
        self.imagebutton.setObjectName("imagebutton")
        self.path_label = QtWidgets.QLabel(Dialog)
        self.path_label.setGeometry(QtCore.QRect(160, 70, 421, 41))
        self.path_label.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.path_label.setText("")
        self.path_label.setObjectName("path_label")
        self.predictbutton = QtWidgets.QPushButton(Dialog)
        self.predictbutton.setGeometry(QtCore.QRect(242, 607, 171, 41))
        self.predictbutton.setObjectName("predictbutton")
        self.resultlabel = QtWidgets.QLabel(Dialog)
        self.resultlabel.setGeometry(QtCore.QRect(242, 660, 171, 41))
        self.resultlabel.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.resultlabel.setText("")
        self.resultlabel.setObjectName("resultlabel")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(270, 150, 151, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
     
        #load input image when browse button clicked
        self.imagebutton.clicked.connect(self.load_image)
        #predict the correct class when predict button clicked
        self.predictbutton.clicked.connect(self.prediction)
    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Image Classification Using Deep Neural Networks"))
        self.imagebutton.setText(_translate("Dialog", "Browse Image"))
        self.predictbutton.setText(_translate("Dialog", "Predict"))
        self.label.setText(_translate("Dialog", "Input Image"))
        self.resultlabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
    #function "load_image"  we will use it to load the image
    def load_image(self):
        global filename
        #open file dialogue to choose the image
        filename,_ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image" , " " ,
                                                           "Image Files (*.jpg *.jpeg *.png *.bmp)")
        #loading image file in pixmap format
        pixmap=QtGui.QPixmap(filename)
        #scaling the image so it can fit in the label
        pixmap=pixmap.scaled(self.imagelabel.width(), self.imagelabel.height(), QtCore.Qt.KeepAspectRatio)
        #setting the image at the center of the label
        self.imagelabel.setPixmap(pixmap)
        self.imagelabel.setAlignment(QtCore.Qt.AlignCenter)
        #showing image path at the path label
        self.path_label.setText(filename)
        
    # function "prediction" we will use it to predict image class
    def prediction(self):
        #reading the image
        input_image = cv2.imread(filename)
        #reshape the image to size (32x32) so it can fit in our CNN
        sized_image = cv2.resize(input_image, (32, 32))
        #extend the image demenssisons so it can fit in our CNN
        extended_image = np.expand_dims(sized_image, axis=0)
        extended_image = extended_image.astype('float32')
        extended_image /= 255
        #loading the pre-trained CNN
        cif10=load_model('cifar10_classifier.h5')
        #predicting the class using the CNN
        predict=cif10.predict_classes(extended_image)
        #defining class names
        classes = ['Airplane','Automobile','Bird','Cat', 'Deer','Dog','Frog','Horse','Ship','Truck']
        #getting the correct name of the predicted class
        class_index = predict[0]
        predicted_class = classes[class_index]
        #showing prediction result
        self.resultlabel.setText(predicted_class)
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

