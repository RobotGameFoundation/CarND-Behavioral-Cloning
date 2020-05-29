# Behavioral Cloning Project

* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cnn_architecture.png "Model Visualization"
[image2]: ./examples/Figure.png "figure1"
[image3]: ./examples/Figure_1.png "figure2"
[image3]: ./examples/center.jpg "center"
[image3]: ./examples/left.jpg "left"
[image3]: ./examples/right.jpg "right"


## Rubric Points
---
### Files Submitted & Code Quality
- `model.py`
- `model.h5`
- `model1.h5`
- `video.mp4`
- `video1.mp4`
- `README.md`
- `drive.py`


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of  convolution neural networks with 3x3 and 5x5 filter sizes 
The model includes RELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer.

```python
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(160, 320, 3)))

model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Conv2D(24,(5,5),subsample=(2,2),activation="relu"))
model.add(Conv2D(36,(5,5),subsample=(2,2),activation="relu"))
model.add(Conv2D(48,(5,5),subsample=(2,2),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
```
#### 2. Attempts to reduce overfitting in the model
To prevent overfitting, I used several data augmentation techniques like flipping images 
as well as using left and right images to help the model generalize.
The model was trained and validated on different data sets to ensure that the model was not overfitting.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data
I run the simulator and record the data ont the two different tracks and train them seperatel

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a good model was to use the [Nvidia](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) architecture since it has been proven to be very successful
in self-driving car tasks. 

#### 2. Final Model Architecture
The final model architecture code is shown above.
Here is a visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process
To create the training data, I used the Udacity sample data as a base. For each image, normalization
would be applied before the image was fed into the network. In my case, a training sample consisted
of six images:
1. Center camera image
2. Center camera image flipped horizontally
3. Left camera image
4. Left camera image flipped horizontally
5. Right camera image
6. Right camera image flipped horizontally

Here are some training images:
![alt text][image4]
![alt text][image5]
![alt text][image6]

The model was then tested on the track to ensure that the model was performing as expected.

Below are 2 images of the history of the training and validation loss of first and second track.
We can see the second track is more hard since the loss is higher than the first one.
![alt text][image2]
![alt text][image3]