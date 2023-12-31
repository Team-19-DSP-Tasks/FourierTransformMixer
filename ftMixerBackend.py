import logging
import sys
import time

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QFileDialog, QMessageBox

# Suppress Matplotlib font finding debug messages
logging.getLogger("matplotlib.font_manager").disabled = True

logging.basicConfig(
    level=logging.DEBUG,
    filename="log.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Add an empty line to the log file
with open("log.log", "a") as log_file:
    log_file.write("\n")


class MyThread(QThread):
    update_progress = pyqtSignal(int)

    def run(self):
        for step in range(101):
            step += 1
            time.sleep(0.0005)
            self.update_progress.emit(step)


class Images:
    def __init__(self):
        """
        Initializes an instance of the Images class with
        various attributes to store image data and related components.
        """
        self.image_instances = {}  # It holds all the instances of the class.
        # This dictionary should always contain only 4 instances or less.
        # That depends on what figures the user chose to plot in.
        self.path = ""
        self.gray_img = None  # holds the gray-scaled image data, encapsuled accessed only by Image class.
        self.shape = (
            None  # A tuple to store the width and height of the gray-scaled image
        )
        self.fourier_transform = None
        self.fourier_transform_shifted = None
        self.magnitude_spectrum = None
        self.phase_spectrum = None
        self.real_component = None
        self.imaginary_component = None
        self.shifted_components = []
        self.typed_shifted_components = None
        self.brightness = None
        self.contrast = None
        self.fft_components_plot = []

    def calc_img_gray_scale(self, path):
        """
        Loads an image from the specified path, converts it to grayscale, and stores the grayscale image data.

        Parameters:
        - 'path' (str): The path to the image file.
        """
        try:
            self.original_gray_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.gray_img = np.copy(
                self.original_gray_img
            )  # Create a copy for processing
            self.shape = self.gray_img.shape
            logging.info(f"The original shape of img is {self.shape}")
        except Exception as e:
            logging.error(f"Error loading image at {self.path}: {e}")
            raise

    def calculate_fft_components(self):
        """
        Calculate the 2D Fourier Transform and related components of the image.
        """

        # Compute 2D Fourier transform
        fourier_transform = np.fft.fft2(self.gray_img)

        # Shift the zero-frequency component "The DC component" to the center
        self.fourier_transform_shifted = np.fft.fftshift(fourier_transform)

        # Compute magnitude, phase, real, and imaginary components
        self.magnitude_spectrum = np.abs(fourier_transform)
        self.phase_spectrum = np.angle(fourier_transform)
        self.real_component = (fourier_transform).real
        self.imaginary_component = fourier_transform.imag

        # Compute the components of the shifted Fourier Transform
        self.shifted_components = [
            np.log(np.abs(self.fourier_transform_shifted) + 1),
            np.angle(self.fourier_transform_shifted),
            np.log(np.maximum(self.fourier_transform_shifted.real, 0) + 1),
            np.log(np.maximum(self.fourier_transform_shifted.imag, 0) + 1),
        ]

        # Map each component to its type in a dictionary for the ease of accesss
        self.typed_shifted_components = dict(
            zip(["Magnitude", "Phase", "Real", "Imaginary"], self.shifted_components)
        )

    @classmethod
    def resize_all_images(cls, image_instances):
        """
        Resize all Image instances to the smallest size among them.

        Parameters:
        - image_instances (list): List of Image instances.

        Returns:
        - None
        """

        # Check if the dictionary image_instacnes is empty
        if not image_instances:
            return

        # Find the smallest size among all instances
        min_width = min(image.shape[1] for image in image_instances.values())
        min_height = min(image.shape[0] for image in image_instances.values())

        logging.info(f"min_width: {min_width}, min_height: {min_height}")

        # Resize all instances to the smallest size
        for image in image_instances.values():
            image.resize_img(min_width, min_height)
            image.calculate_fft_components()

        # Add logging statement to print the shapes after resizing
        for canvas_id, image in image_instances.items():
            logging.info(
                f"After resizing, Canvas ID: {canvas_id}, Image shape: {image.shape}"
            )

    def resize_img(self, new_width, new_height):
        """
        Resize the image to the specified width and height.

        Parameters:
        - new_width (int): The new width of the image.
        - new_height (int): The new height of the image.

        Returns:
        - None
        """

        try:
            self.gray_img = cv2.resize(self.gray_img, (new_width, new_height))

            # log the size of image: checking if the functions are working correctly
            logging.info(f"Resized image shape: {self.shape}")

        except Exception as e:
            logging.error(f"Error resizing image: {e}")
            raise

    def adjust_brightness(self, brightness):
        # Apply only brightness adjustment to the image
        self.gray_img = np.clip(self.original_gray_img + brightness, 0, 255).astype(
            np.uint8
        )

    def adjust_contrast(self, contrast):
        # Apply only contrast adjustment to the image
        alpha = (contrast + 127) / 127
        beta = 0  # No brightness adjustment in contrast adjustment
        self.gray_img = np.clip(alpha * self.original_gray_img + beta, 0, 255).astype(
            np.uint8
        )

    def reset_brightness_contrast(self):
        self.gray_img = np.copy(self.original_gray_img)


class Image_Processing_App(Images):
    def __init__(self, ui):
        self.ui = ui
        # Inherited Attributes I will operate on.
        super().__init__()

        # UI Class Attributes
        self.mag_phase_inv_fourier = None
        self.real_imaginary_inv_fourier = None

        # the new array components for the output image
        self.magnitude_sum = None
        self.phase_sum = None
        self.real_sum = None
        self.imaginary_sum = None

        self.fft_plotted_components = {}
        self.sliced_components_dict = {}
        self.inner_outer_dict = {}

        # Dictionary to store fft_component for each image
        self.image_fft_components = {}

        # Dictionary to map canvas IDs to canvas objects
        self.canvas_mapping = {
            "input_1": self.ui.imgcanvas01,
            "input_2": self.ui.imgcanvas02,
            "input_3": self.ui.imgcanvas03,
            "input_4": self.ui.imgcanvas04,
        }

        # Dictionary to map output_canvas IDs to canvas objects
        self.output_canvas_mapping = {
            "output_1": self.ui.outCanvas01,
            "output_2": self.ui.outCanvas02,
        }

        # Dictionary to map FFT canvas IDs to canvas object
        self.fftCanvas_mapping = {
            "fft_1": self.ui.fftcanvas01,
            "fft_2": self.ui.fftcanvas02,
            "fft_3": self.ui.fftcanvas03,
            "fft_4": self.ui.fftcanvas04,
        }

        # Dictionary to store the sliders values (weights)
        self.slider_values = {
            "img01atr": 0,
            "img02atr": 0,
            "img03atr": 0,
            "img04atr": 0,
        }

        self.weights = []

        # Connections
        for i in range(1, 5):
            canvas_name = f"imgcanvas{i:02d}"
            canvas = getattr(self.ui, canvas_name)
            canvas.mpl_connect("button_press_event", self.store_and_display_image)

        self.ui.imgComboBoxToolbar.currentIndexChanged.connect(
            self.update_fft_combobx_on_change_image_combobox
        )
        self.ui.ftCombonentComboBox.currentTextChanged.connect(
            self.update_image_component_on_change_fft_combobox
        )
        self.ui.outputPortSelector.currentTextChanged.connect(self.select_output_port)
        self.ui.selectMode.currentIndexChanged.connect(self.select_mode)
        self.ui.actionRectangle_Selection.triggered.connect(self.region_selection)
        self.ui.selectRegion.currentTextChanged.connect(self.handle_combobox_change)
        self.ui.mixComponents.clicked.connect(self.mix)

        # Connect sliders to the update_slider_value function
        for i in range(1, 5):
            slider_name = getattr(self.ui, f"img{i:02d}atr")
            slider_name.valueChanged.connect(self.update_slider_value)

        # Mouse event handler for brightness/contrast adjustment
        for canvas in self.canvas_mapping.values():
            canvas.mpl_connect("button_press_event", self.brightness_contrast)

    # Functions
    def assign_image_path(self):
        options = QFileDialog.Options()
        path, _ = QFileDialog.getOpenFileName(
            None,
            "Open Image File",
            "test-images",
            "Image Files (*.png *.jpg *.bmp);;All Files (*)",
            options=options,
        )
        return path

    def store_and_display_image(self, event):
        if event.dblclick:
            path = self.assign_image_path()
            if path:
                canvasID = self.get_id_from_canvas_mapping(event, self.canvas_mapping)
                image_instance = Images()  # Create a new instance for this canvas
                image_instance.calc_img_gray_scale(path)
                image_instance.calculate_fft_components()

                self.image_instances[canvasID] = image_instance
                image_instance.resize_all_images(self.image_instances)

                # Initialize the new array components with respective shapes
                self.magnitude_sum = np.zeros_like(
                    image_instance.typed_shifted_components["Magnitude"]
                )
                self.phase_sum = np.zeros_like(
                    image_instance.typed_shifted_components["Phase"]
                )
                self.real_sum = np.zeros_like(
                    image_instance.typed_shifted_components["Real"]
                )
                self.imaginary_sum = np.zeros_like(
                    image_instance.typed_shifted_components["Imaginary"]
                )

                # Loop through all image instances
                for canvas_id, img_instance in self.image_instances.items():
                    # Open each image
                    self.plot_image_on_canvas(
                        img_instance.gray_img, canvas_id, self.canvas_mapping
                    )
                    # Plot FFT for each image
                    fft_canvas_id = self.get_fft_id_from_img_id(canvas_id)
                    # Initialize the image_fft_components dictionary with the default fft_component
                    fft_component = self.ui.ftCombonentComboBox.currentText()
                    self.image_fft_components[canvasID] = fft_component
                    self.plot_image_on_canvas(
                        image_instance.typed_shifted_components[fft_component],
                        fft_canvas_id,
                        self.fftCanvas_mapping,
                    )
                    # Create the dictionary entry
                    key = f"{fft_canvas_id}_{fft_component}"
                    value = img_instance.typed_shifted_components[fft_component]
                    self.fft_plotted_components[key] = value

                self.update_input_combobox()
                self.handle_combobox_change()
                self.update_labels()

    def plot_image_on_canvas(self, data, canvas_id, canvas_mapping):
        """
        Plots an image on the specified canvas.

        Parameters:
        - 'data': The image data to be displayed.
        - 'canvas_id' (str): The ID of the canvas where the image should be displayed.
        - 'canvas_mapping': The mapping of canvas IDs to canvas objects.
        """
        canvas = canvas_mapping.get(canvas_id)
        if canvas is not None:
            canvas.figure.clf()
            ax = canvas.figure.add_subplot(111)
            ax.imshow(data, cmap="gray")
            ax.axis("off")
            canvas.draw()
        else:
            logging.error(f"Canvas not found for ID: {canvas_id}")

    def get_id_from_canvas_mapping(self, event, canvas_mapping):
        """
        Gets the canvas ID based on the event object.

        Parameters:
        - 'event': The event object.
        - 'canvas_mapping': The mapping of canvas IDs to canvas objects.

        Returns:
        - 'canvas_id' (str): The ID of the canvas.
        """
        canvas_id = None
        for key, canvas in canvas_mapping.items():
            if canvas == event.canvas:
                canvas_id = key
                break
        return canvas_id

    def get_fft_id_from_img_id(self, canvas_id):
        """
        Gets the FFT canvas ID based on the image canvas ID.

        Parameters:
        - 'canvas_id' (str): The ID of the image canvas.

        Returns:
        - 'fft_canvas_id' (str): The ID of the FFT canvas corresponding to the given image canvas.
        """
        # make the two dictionary keys as lists that are corresponding to each other (order wise) and take the key from the fft_canvas corresponding to the same order of the key in the img_canvas
        canvas_list = list(self.canvas_mapping.keys())
        try:
            fft_canvasID = list(self.fftCanvas_mapping.keys())[
                canvas_list.index(canvas_id)
            ]
            return fft_canvasID
        except ValueError:
            logging.error("Canvas not found in the mapping.")
            return None

    def update_input_combobox(self):
        """
        Updates the content of the image selection combobox (imgComboBoxToolbar)
        based on the available image instances.
        """
        self.ui.imgComboBoxToolbar.clear()

        for item in self.image_instances.keys():
            self.ui.imgComboBoxToolbar.addItem(f"{item}")

        logging.info(f"Canvas Used: {self.canvas_mapping.keys()}")
        logging.info(f"Canvas Used: {self.fftCanvas_mapping.keys()}")

    def update_fft_combobx_on_change_image_combobox(self):
        """
        Updates the FFT component selection combobox (ftCombonentComboBox)
        when the selected image changes.
        """
        canvas_id = self.get_the_key_for_image_fft_components()
        selected_image_fft_component = self.image_fft_components.get(canvas_id)
        self.ui.ftCombonentComboBox.setCurrentText(selected_image_fft_component)

    def update_image_component_on_change_fft_combobox(self):
        """
        Updates the selected FFT component for a specific
        image when the FFT component selection changes.
        """
        self.update_fft_combobox_when_select_mode_change()
        selected_fft = self.ui.ftCombonentComboBox.currentText()
        canvas_id = self.get_the_key_for_image_fft_components()
        self.image_fft_components[canvas_id] = selected_fft
        self.ui.ftCombonentComboBox.setCurrentText(selected_fft)
        self.update_labels()

        self.update_fft_plot()
        pass

    def update_labels(self):
        for i, (key, value) in enumerate(self.image_fft_components.items()):
            label_name = f"label{i:02d}"
            label = getattr(self.ui, label_name)
            label.setText((value))

    def update_fft_combobox_when_select_mode_change(self):
        """
        Updates the selected FFT component for all images based on the selected processing mode.

        Returns:
        - None
        """
        selected_item = self.ui.selectMode.currentText()
        component_mapping = {
            "Magnitude and Phase": "Magnitude",
            "Real and Imaginary": "Real",
            # Add more mappings as needed
        }

        if not self.ui.selectMode.signalsBlocked():
            for key in self.image_fft_components.keys():
                self.image_fft_components[key] = component_mapping.get(
                    selected_item, self.image_fft_components[key]
                )

    def get_the_key_for_image_fft_components(self):
        """
        Retrieves the canvas ID associated with the selected image in the image combobox.

        Returns:
        - canvas_id (str): The ID of the canvas associated with the selected image.
        """
        return self.ui.imgComboBoxToolbar.currentText()

    def update_fft_plot(self):
        """
        Description: Updates FFT components plotting after resizing and
        plots the selected FFT component from comboboxes.
        """
        print(self.image_fft_components)

        # update fft_components after resizing
        canvas_id = self.get_the_key_for_image_fft_components()
        if canvas_id in self.image_instances:
            image_instance = self.image_instances[canvas_id]
            fft_canvasID = self.get_fft_id_from_img_id(canvas_id)
            fft_component = self.ui.ftCombonentComboBox.currentText()
            self.plot_image_on_canvas(
                image_instance.typed_shifted_components[fft_component],
                fft_canvasID,
                self.fftCanvas_mapping,
            )

    def select_mode(self):
        """
        Description: Handles the selection of processing modes
        (Magnitude and Phase or Real and Imaginary),
        updates the FFT component combobox accordingly,
        and replots all FFT components based on the selected mode.
        """
        selected_item = self.ui.selectMode.currentText()

        fft_component = self.change_combobox_mode(selected_item)

        # When mode is Switched replot all of the fft_components according to the mode
        for fft_canvas_id in self.fftCanvas_mapping.keys():
            image_instance = self.image_instances.get(
                fft_canvas_id.replace("fft", "input")
            )
            if image_instance:
                self.plot_image_on_canvas(
                    image_instance.typed_shifted_components[fft_component],
                    fft_canvas_id,
                    self.fftCanvas_mapping,
                )

    def change_combobox_mode(self, selected_mode):
        combonents = selected_mode.split()
        print(combonents)
        self.ui.ftCombonentComboBox.clear()
        self.ui.ftCombonentComboBox.addItems([combonents[0], combonents[2]])
        fft_component = combonents[0]
        return fft_component

    def brightness_contrast(self, event):
        """
        Description: Applies Brightness/Contrast adjustment
        on the clicked canvas image. Restores default settings on right-click.

        Parameters:
        - 'event': Mouse event containing information about
                the button pressed and mouse position.
        """
        if event.button == 1:  # Left mouse button
            if event.xdata is not None and event.ydata is not None:
                canvas_id = self.get_id_from_canvas_mapping(event, self.canvas_mapping)
                if canvas_id and canvas_id.startswith("input"):
                    self.dragging = True  # Flag to indicate dragging
                    self.start_x = event.xdata
                    self.start_y = event.ydata
                    canvas = self.canvas_mapping.get(canvas_id)
                    if canvas:
                        canvas.mpl_connect(
                            "motion_notify_event", self.motion_notify_event
                        )
                        canvas.mpl_connect("button_release_event", self.release_event)
                    else:
                        logging.error(f"Canvas not found for ID: {canvas_id}")
        elif event.button == 3:  # Right mouse button
            canvas_id = self.get_id_from_canvas_mapping(event, self.canvas_mapping)
            if canvas_id and canvas_id.startswith("input"):
                image_instance = self.image_instances.get(canvas_id)
                if image_instance:
                    image_instance.reset_brightness_contrast()
                    # Update the displayed image, and before plotting resize as before.
                    image_instance.resize_all_images(self.image_instances)
                    self.plot_image_on_canvas(
                        image_instance.gray_img, canvas_id, self.canvas_mapping
                    )
                    self.calculate_and_plot_adjusted_fft(canvas_id)

    def motion_notify_event(self, event):
        """
        Description: Enables continuous Brightness/Contrast adjustment
        on the clicked canvas image while dragging the mouse.

        Parameters:
        - 'event': Mouse event containing information about the mouse position.
        """
        if hasattr(self, "dragging") and self.dragging:
            if event.xdata is not None and event.ydata is not None:
                dx = event.xdata - self.start_x
                dy = event.ydata - self.start_y

                # Calculate brightness and contrast adjustments
                brightness = int(dy)
                contrast = int(dx)

                canvas_id = self.get_id_from_canvas_mapping(event, self.canvas_mapping)
                if canvas_id and canvas_id.startswith("input"):
                    image_instance = self.image_instances.get(canvas_id)
                    if image_instance:
                        image_instance.adjust_brightness(brightness)
                        image_instance.adjust_contrast(contrast)
                        # Update the displayed image
                        image_instance.resize_all_images(self.image_instances)
                        self.plot_image_on_canvas(
                            image_instance.gray_img, canvas_id, self.canvas_mapping
                        )
                        self.calculate_and_plot_adjusted_fft(canvas_id)

    def release_event(self, event):
        """
        Description: Stops Brightness/Contrast adjustment on mouse release.

        Parameters:
        - 'event': Mouse event containing information about the button released.
        """
        if hasattr(self, "dragging"):
            self.dragging = False

    def calculate_and_plot_adjusted_fft(self, canvas_id):
        """
        Description: Calculates and plots the FFT after brightness/contrast
        adjustments on the specified canvas.

        Parameters:
        - 'canvas_id': ID of the canvas on which adjustments were made.
        """
        image_instance = self.image_instances.get(canvas_id)
        if image_instance:
            # image_instance.calculate_fft_components()
            fft_canvasID = self.get_fft_id_from_img_id(canvas_id)
            if fft_canvasID and fft_canvasID.startswith("fft"):
                canvas_ID = fft_canvasID.replace("fft", "input")
            # Get the current fft_component from the dictionary
            fft_component = self.image_fft_components.get(canvas_ID)
            self.plot_image_on_canvas(
                image_instance.typed_shifted_components[fft_component],
                fft_canvasID,
                self.fftCanvas_mapping,
            )

    def region_selection(self):
        """
        Description: Enables selection of a region on the FFT canvas.
        """
        # Removed an unused parameter
        for _, canvas in self.fftCanvas_mapping.items():
            canvas.mpl_disconnect(canvas._idPress)
            canvas._idPress = canvas.mpl_connect("button_press_event", self.on_press)

    def on_press(self, event):
        """
        Description: Stores starting indices for the region selection on the FFT canvas.

        Parameters:
        - 'event': Mouse event containing information about
                the button pressed and mouse position.
        """
        if event.button == 1:  # Left mouse button
            fft_canvas_id = self.get_id_from_canvas_mapping(
                event, self.fftCanvas_mapping
            )
            print("Canvas ID:", fft_canvas_id)  # Add this line to print the canvas ID
            if fft_canvas_id and fft_canvas_id.startswith("fft"):
                self.dragging = True
                self.start_x = int(event.xdata) if event.xdata else None
                self.start_y = int(event.ydata) if event.ydata else None
                self.fft_canvas_id = fft_canvas_id
                canvas = self.fftCanvas_mapping.get(fft_canvas_id)
                if canvas:
                    canvas.mpl_disconnect(canvas._idRelease)
                    canvas._idRelease = canvas.mpl_connect(
                        "button_release_event", self.on_release
                    )
            else:
                logging.error("Invalid canvas for region selection")
        elif event.button == 3:
            canvas_id = self.get_the_key_for_image_fft_components()
            # Get the current fft_component from the dictionary
            fft_component = self.image_fft_components.get(canvas_id)
            image_instance = self.image_instances.get(
                self.fft_canvas_id.replace("fft", "input")
            )
            self.plot_image_on_canvas(
                image_instance.typed_shifted_components[fft_component],
                self.fft_canvas_id,
                self.fftCanvas_mapping,
            )

    def handle_combobox_change(self):
        selected_text = self.ui.selectRegion.currentText()
        if selected_text == "Full Image":
            self.ui.actionRectangle_Selection.setEnabled(False)
        elif selected_text == "Inner" or selected_text == "Outer":
            self.ui.actionRectangle_Selection.setEnabled(True)
        return selected_text

    def on_release(self, event):
        """
        Store ending indices and slice the typed_shifted_components dictionary.
        """
        if event.button == 1:  # Left mouse button
            if hasattr(self, "dragging") and self.dragging:
                self.dragging = False
                end_x = int(event.xdata) if event.xdata else None
                end_y = int(event.ydata) if event.ydata else None

                fft_canvas_id = self.get_id_from_canvas_mapping(
                    event, self.fftCanvas_mapping
                )
                # Get the current fft_component from the dictionary
                canvas_id = self.get_the_key_for_image_fft_components()
                fft_component = self.image_fft_components.get(canvas_id)
                print(self.image_fft_components)
                image_instance = self.image_instances.get(
                    fft_canvas_id.replace("fft", "input")
                )

                if (
                    image_instance
                    and fft_canvas_id
                    and fft_component
                    and self.start_x is not None
                    and self.start_y is not None
                    and end_x is not None
                    and end_y is not None
                ):
                    x_indices = sorted([self.start_x, end_x])
                    y_indices = sorted([self.start_y, end_y])

                    # Access the typed_shifted_components dictionary based on fft_component and slice the region
                    selected_fft_component = (
                        image_instance.typed_shifted_components.get(fft_component)
                    )
                    if selected_fft_component is not None:
                        # Print the shape of selected_fft_component before slicing
                        print(
                            "Shape of selected_fft_component before slicing:",
                            selected_fft_component.shape,
                        )
                        # Create an array of zeros or ones with the same shape as the selected component
                        sliced_comp = (
                            np.zeros_like(selected_fft_component)
                            if self.handle_combobox_change() == "Inner"
                            else np.ones_like(selected_fft_component)
                        )

                        sliced_comp[
                            y_indices[0] : y_indices[1], x_indices[0] : x_indices[1]
                        ] = (1 if self.handle_combobox_change() == "Inner" else 0)

                        # Create or update the dictionary with the sliced component for all components
                        for canvas_id, _ in self.image_instances.items():
                            fft_canvas_id = self.get_fft_id_from_img_id(canvas_id)
                            key = f"{fft_canvas_id}_{fft_component}"
                            self.sliced_components_dict[key] = sliced_comp

                            # Add a transparent rectangle overlay on all components
                            self.plot_transparent_rectangle(
                                fft_canvas_id,
                                x_indices[0],
                                y_indices[0],
                                x_indices[1],
                                y_indices[1],
                                alpha=0.5,  # Adjust the alpha value for transparency
                            )

    def plot_transparent_rectangle(
        self, canvas_id, x_start, y_start, x_end, y_end, alpha=0.5
    ):
        """
        Plot a transparent rectangle overlay on the specified canvas.

        Parameters:
        - canvas_id: ID of the canvas where the rectangle will be plotted.
        - x_start, y_start: Coordinates of the top-left corner of the rectangle.
        - x_end, y_end: Coordinates of the bottom-right corner of the rectangle.
        - alpha: Transparency level for the rectangle (0.0 to 1.0).
        """
        canvas = self.fftCanvas_mapping[canvas_id].figure.gca()

        # Remove any existing rectangles on the canvas
        existing_rectangles = [
            p for p in canvas.patches if isinstance(p, patches.Rectangle)
        ]
        for rectangle in existing_rectangles:
            rectangle.remove()

        # Add the new rectangle
        new_rectangle = patches.Rectangle(
            (x_start, y_start),
            x_end - x_start,
            y_end - y_start,
            linewidth=2,
            edgecolor="r",  # Adjust the color of the rectangle as needed
            facecolor="none",
            alpha=alpha,
        )
        canvas.add_patch(new_rectangle)
        canvas.figure.canvas.draw()

    def update_slider_value(self):
        """
        Description: Updates the weights based on the current slider values.
        """
        slider_list = [
            self.ui.img01atr,
            self.ui.img02atr,
            self.ui.img03atr,
            self.ui.img04atr,
        ]

        # Update the weights based on the current slider values
        allValues = [slider.value() for slider in slider_list]
        sumOfWeights = sum(allValues)

        if sumOfWeights > 100:
            self.weights = [(value / sumOfWeights) * 1 for value in allValues]
        else:
            self.weights = [value / 100 for value in allValues]
        # self.weights = [slider.value() / 100 for slider in slider_list]
        print(f"Weights: f{self.weights}")

        # Set the range for all sliders
        num_images = int((1 / len(self.image_instances)) * 100)
        for slider in slider_list:
            slider.setRange(0, 100)

        print(self.weights)

    def choose_mode(self):
        """
        Description: Checks the processing mode
        Magnitude and Phase or Real and Imaginary)
        based on the current images. Returns 1 for Magnitude and Phase,
        2 for Real and Imaginary.

        Returns:
        - Integer (1 or 2)
        """
        print(self.weights)

        # Check which mode we are in
        selected_item = self.ui.selectMode.currentText()
        magnitude_phase = 0
        real_imaginary = 0
        for _ in self.image_instances.items():
            if selected_item == "Magnitude and Phase":
                magnitude_phase += 1
            elif selected_item == "Real and Imaginary":
                real_imaginary += 1
        if magnitude_phase == 0:
            return 1
        elif real_imaginary == 0:
            return 2
        else:
            raise ValueError("Invalid types")

    def apply_mixer(
        self,
        mags_or_real,
        phase_or_imag,
        selected_text,
        key_mags_or_real,
        key_phase_or_imag,
    ):
        result_mags_or_real = np.zeros_like(mags_or_real)
        result_phase_or_imag = np.zeros_like(phase_or_imag)

        for i, ((key, value), (_, value_2)) in enumerate(
            zip(self.image_instances.items(), self.inner_outer_dict.items())
        ):
            image_inst = value
            if selected_text == "Full Image":
                value_2 = 1  # Array of ones with the same shape as image_inst
            else:
                value_2 = value_2  # Use the existing value from the dictionary

            type = self.image_fft_components[key]
            if type == key_mags_or_real:
                result_mags_or_real += (
                    2
                    * self.weights[i]
                    * image_inst.typed_shifted_components[key_mags_or_real]
                    * value_2
                )

            elif type == key_phase_or_imag:
                result_phase_or_imag += (
                    2
                    * self.weights[i]
                    * image_inst.typed_shifted_components[key_phase_or_imag]
                    * value_2
                )

        result_mags_or_real = np.fft.ifftshift(np.exp(result_mags_or_real))
        result_phase_or_imag = np.fft.ifftshift(result_phase_or_imag)
        reconstructed = result_mags_or_real * np.exp(1j * result_phase_or_imag)
        return np.clip(np.abs(np.fft.ifft2(reconstructed)), 0, 255)

    def inverse_fft(self):
        """
        Description: Performs inverse FFT based on the chosen mode,
        selected region, and image weights. Returns the resulting image.

        Returns:
        - Resulting image array.
        """
        mode = self.choose_mode()
        selected_text = self.handle_combobox_change()

        if selected_text == "Full Image":
            self.inner_outer_dict = self.image_instances
        else:
            self.inner_outer_dict = self.sliced_components_dict

        if mode == 2:
            reconstructed_image = self.apply_mixer(
                self.magnitude_sum, self.phase_sum, selected_text, "Magnitude", "Phase"
            )
        elif mode == 1:
            reconstructed_image = self.apply_mixer(
                self.real_sum, self.imaginary_sum, selected_text, "Real", "Imaginary"
            )

        return reconstructed_image

    def mix(self):
        """
        Description: Mixes and plots the resulting image based on inverse FFT.
        """
        self.startProgressBar()
        result_img = self.inverse_fft()
        self.plot_image_on_canvas(
            result_img, self.select_output_port(), self.output_canvas_mapping
        )

    def select_output_port(self):
        return self.ui.outputPortSelector.currentText()

    def startProgressBar(self):
        self.thread = MyThread()
        self.thread.start()
        self.thread.update_progress.connect(self.updateProgressValue)

    def updateProgressValue(self, value):
        self.ui.progressBar.setValue(value)

        if value == 100:
            self.show_info_dialog()
            self.ui.progressBar.setValue(0)

    def show_info_dialog(self):
        """
        Show a dialog message when the progress bar reahces 100% ->
            that means the end of Ifft Calculation and plotting.
        """
        info_dialog = QMessageBox(None)
        info_dialog.resize(400, 200)
        # Add the following line to set the dialog icon
        icon = QIcon("icons/image-processing.png")
        info_dialog.setWindowIcon(icon)

        info_dialog.setIcon(QMessageBox.Information)
        info_dialog.setWindowTitle("Info")
        info_dialog.setText("IFFT Calculation Completed 100%!")
        info_dialog.exec_()
