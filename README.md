# Face-Recognition

## Description

This GitHub repository contains code that implements a Face recognition application to identify and store user's photos. 

The project aims to simplify the process of organizing and managing large amounts of image data by automating the identification and categorization of photos using pre-trained Machine Learning models.

## How It Works
1. <b>Face Capture</b>: The code accesses the user's webcam to capture their face in real-time.
2. <b>Face Detection</b>: Using the pre-trained machine learning models, the code detects and extracts the face from the captured image.
3. <b>Face Recognition</b>: The extracted face is compared against a collection of photos to identify matching faces.
4. <b>Image Sorting</b>: All the photos that contain the user's face are identified and moved to a designated folder for easy access.

## Benefits
1. <b>Efficient Photo Retrieval</b>: By automatically identifying and organizing photos that feature the user's face, the code makes it easy to locate and access specific images.
2. <b>Streamlined Organization</b>: The code automatically sorts the photos into a separate folder, reducing the need for manual categorization.
3. <b>Time Savings</b>: The automated process eliminates the need for manual searching and sorting, saving users valuable time.


## How to Run the Project
To run the project, follow the instructions below:

1. Clone this repository by running the following command in terminal:

        $ git clone https://github.com/AkankshCaimi/Face-Recognition.git
        
2. Open the project directory:

        $ cd Face-Recognition
        
3. Run the following command on terminal to install the used libraries:

        $ pip install opencv-python
        $ pip install os
        $ pip install dlib
        $ pip install numpy
        $ pip install imutils

4. Store all the photos in `./Dataset` directory.
5. Run the program:

        $ python Face_Recognition.py

    This will initiate the webcam capture and face recognition process.

6. The identified photos containing the user's face will be moved to a separate directory (`./similar`) for easy access.


## Author
Me :)

