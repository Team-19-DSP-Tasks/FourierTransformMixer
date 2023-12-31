# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'click_and_drag.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
from matplotlib.image import imread
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
import numpy as np

class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")

        # image 01
        self.horizontalLayout01 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout01.setObjectName("HorizCanv01")
        self.figure01 = plt.figure()
        self.canvas = FigureCanvas(self.figure01)
        self.canvas.mpl_connect('button_press_event', lambda event: self.on_canvas_click(event, self.canvas, interval=500))
        self.canvas.mpl_connect('button_press_event', lambda event: self.on_canvas_press(event, self.canvas))
        self.canvas.mpl_connect('motion_notify_event', lambda event: self.on_canvas_drag(event, self.canvas))
        self.canvas.mpl_connect('button_release_event', lambda event: self.on_canvas_release(event, self.canvas))
        self.horizontalLayout01.addWidget(self.canvas)

        self.verticalLayout.addWidget(self.frame)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.image_loaded = False

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def on_canvas_press(self, event, canvas):
        if self.image_loaded:
            self.press_x = event.x
            self.press_y = event.y
            self.contrast_factor = 1.0
            self.brightness_factor = 0.0
            # Set the cursor to a hand when dragging starts
            QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

    def on_canvas_drag(self, event, canvas):
        if self.image_loaded and event.button == 1:  # Left button
            # Calculate the vertical movement
            delta_y = event.y - self.press_y

            # Adjust contrast based on vertical movement
            self.contrast_factor = 1.0 + delta_y / 100.0

            # Calculate the horizontal movement
            delta_x = event.x - self.press_x

            # Adjust brightness based on horizontal movement
            self.brightness_factor = delta_x / 100.0

            # Apply contrast and brightness adjustments to the displayed image
            adjusted_image = self.original_image * self.contrast_factor + self.brightness_factor

            # Update the displayed image on the canvas
            canvas.figure.clf()
            ax = canvas.figure.add_subplot(111)
            ax.imshow(adjusted_image)
            ax.axis('off')
            canvas.draw()

    def on_canvas_release(self, event, canvas):
        if self.image_loaded and event.button == 3:     # Right button
            # Reset brightness and contrast
            self.contrast_factor = 1.0
            self.brightness_factor = 0.0
            self.reset_image()
            # Set the cursor back to the default arrow
            QtWidgets.QApplication.restoreOverrideCursor()

    def reset_image(self):
        # Clear previous plots
        self.canvas.figure.clf()

        # Plot the original image
        ax = self.canvas.figure.add_subplot(111)
        ax.imshow(self.original_image)
        ax.axis('off')

        # Redraw the canvas
        self.canvas.draw()

    def on_canvas_click(self, event, canvas, interval=300):
        if event.dblclick:
            self.on_canvas_double_click(event, canvas)
        else:
            self.on_canvas_single_click(event, canvas)

    def on_canvas_single_click(self, event, canvas):
        # Your code for single-click event goes here
        print("Single-click event")

    def on_canvas_double_click(self, event, canvas):
        print("Double-click event")
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setNameFilter("Image Files (*.png *.jpg *.bmp)")
        file_dialog.setWindowTitle("Open Image File")

        if file_dialog.exec_() == QtWidgets.QFileDialog.Accepted:
            selected_file = file_dialog.selectedFiles()[0]
            self.original_image = imread(selected_file)
            self.displayed_image = np.copy(self.original_image)
            self.image_loaded = True

            self.canvas.figure.clf()
            ax = self.canvas.figure.add_subplot(111)
            ax.imshow(self.displayed_image)
            ax.axis('off')
            self.canvas.draw()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())