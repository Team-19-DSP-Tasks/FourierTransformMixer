# Fourier Transform Image Mixer

## Table of Contents:
- [Description](#description)
- [Project Features](#project-features)
- [Executing program](#executing-program)
- [Help](#help)
- [Contributors](#contributors)
- [License](#license)

## Description

The Fourier Transform Mixer is a desktop program designed to illustrate the relative importance of magnitude and phase components while emphasizing frequencies' contributions to a 2D signal (image). The software provides image viewers, output ports, brightness/contrast adjustment, components mixer, regions mixer, and real-time mixing features to offer a comprehensive understanding of Fourier transformations.

## Project Features

:white_check_mark: **Images Viewers**:
  - Open and view four grayscale images in separate viewports.
  - Colored images are coverted automatically to grayscale.
  - Unified sizes of opened images based on the smallest size.
  - Display Fourier Transform (FT) components (Magnitude, Phase, Real, Imaginary) for each image.
  - Easy browse: Change images by double-clicking on their viewer.

:white_check_mark: **Two Output Ports**:
  - Display mixer results in two output viewports similar to input image viewports.
  - User control over which viewport shows the mixer result.

:white_check_mark: **Brightness/Contrast**:
  - Change brightness/contrast (window/level) via mouse dragging.
  - Applicable to any of the four components.

:white_check_mark: **Components Mixer**:
  - Output image is the Inverse Fourier Transform (ifft) of a weighted average of the FT of the input four images.
  - Customize weights of each image FT via sliders.

:white_check_mark: **Regions Mixer**:
  - Pick regions (inner or outer) for each FT component.
  - Draw rectangles on each FT for region selection.
  - Customize size/percentage of the region rectangle via sliders or resize handles.

:white_check_mark: **Realtime Mixing**:
  - Display progress bar during the ifft operation.
  - Cancel previous operation if a new mixing request is made while the previous one is still running (Check Threads!).

## Executing program

To be able to use our app, you can simply follow these steps:
1. Install Python3 on your device. You can download it from <a href="https://www.python.org/downloads/">Here</a>.
2. Install the required packages by the following command.
```
pip install -r requirements.txt
```
3. Run the file with the name "ftMixerUI.py"

## Help

If you encounter any issues or have questions, feel free to reach out.

## Contributors

Gratitude goes out to all team members for their valuable contributions to this project.

<div align="left">
  <a href="https://github.com/cln-Kafka">
    <img src="https://avatars.githubusercontent.com/u/100665578?v=4" width="100px" alt="@Kareem Noureddine">
  </a>
  <a href="https://github.com/1MuhammadSami1">
    <img src="https://avatars.githubusercontent.com/u/139786587?v=4" width="100px" alt="@M.Sami">
  </a>
  <a href="https://github.com/MohamedSayedDiab">
    <img src="https://avatars.githubusercontent.com/u/90231744?v=4" width="100px" alt="@M.Sayed">
  </a>
</div>

## License

All rights reserved © 2023 to Team 19 - Systems & Biomedical Engineering, Cairo University (Class 2025)
