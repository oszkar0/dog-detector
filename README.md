# Dog Detector Project

## Description

The Dog Detector project utilizes a pretrained neural network, VGG16, to which additional layers have been added. The final neural network has two outputs: the first classifier determines if a dog is present in the input, and the second identifies two points, creating a bounding box around the detected dog. The network was trained on images of my dog, and additional images were augmented using the Albumentations library. The project was built using TensorFlow. The detector may not work fully effectively for other dogs. Additional training data is needed to improve accuracy.

## Demo

Check out the project in action: [Dog Detector Demo](https://youtu.be/Bok2TSt484g)

## Installation

To install the project, follow these steps:

1. Clone the repository: ```git clone https://github.com/oszkar0/dog-detector```

2. Navigate to the 'app' folder, where you will find the `requirements.txt` file containing the required packages.

3. Create and activate virtual environment.

4. Install the required packages using: ```pip install -r requirements.txt```

Now, the project is ready for use.

## Usage

To run the program, navigate to the folder containing the `detector.py` file and run the following command:

```python detector.py [weight file name] [source (0 for camera, video file name e.g. "test.mp4")]```

The program will read the neural network weights (you can find sample weights in app/weights), initiate the detector, and start processing frames from the selected source.


I found the following video very helpful in developing the project: [Build a Deep Face Detection Model with Python and Tensorflow by Nicholas Renotte](https://www.youtube.com/watch?v=N_W4EYtsa10)