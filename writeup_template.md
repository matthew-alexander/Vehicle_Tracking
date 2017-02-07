

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/car_and_hog.png
[image3]: ./examples/sliding_windows.png
[image4]: ./examples/heatmap_boxes.png
[image5]: ./examples/test_frame3.jpg


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  



### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This document serves as the readme addressing the rubric points and providing out put images and discussion of my process.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first 8 code cells of the IPython notebook `P5.py`. The main HOG related functions are there, specifically the `get_hog_features()` function which returns a vector of Hog features used for training. 

I started by reading in all the `vehicle` and `non-vehicle` images (cell 4 in the notebook).  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

My process for choosing the HOG parameters was mainly trial and error. I used a test image to check to see if detections were being made and I played with HOG orientations, pixels per cell and cells_per_block so to reduce the number of features in the feature vector that is constructed for training the Linear Support Vector Machine. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The model I choose to discriminate between cars and non-cars was a linear support vector machine (SVC). The model was trained on 8792 and 9666 car and non-car images, respectfully. 

Features were extracted from each image and formed into a vector for training. The main processes for feature extraction used on each image were: 

1. spatial binning -- each image was processed to reduce its size to 32x32 pixels. 
2. color histogram -- a histogram was calculated for each color channel of each image. These three histograms were then concatenated together to form color features.
3. HOG features -- a histogram of oriented gradients (HOG) was made for each image. This was done for all channels. 

The three feature vectors above were combined into a large single vector vor each image and then scaled using sklearn's `StandardScaler()` in order to reduce the impact of large values in certain features over others. The resultant feature vector length for each image was 6108.

The scaled feature vectors were then split into training and testing sets -- with 20% of the data reserved for testing. The training of the SVC took around 12 seconds to train and achieved a test accuracy of 99%. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The the model can discriminate between cars and non-cars. In order to find cars in video frames, I implemented a sliding window algorithm that subsampled smaller images from the larger video frame. Three different sizes were used to scan the image. Each smaller image was then resized to a 64x64 pixel image and features were extracted to form a feature vector for classificaton by the SVC. 

Three different sizes of ome_filewindows were used along with varying overlaps for the windows: 

1. small windows -- 64x64, with an overlap of 0.5 in both x and y directions. These smaller windows were limited to a y axis range from 400-500px. 
2. medium windows -- 96x96, with an overlap of 0.8 in both x and y directions. These windows were limited to a y axis range from 400-550 px. 
3. large windows -- 256x256, with an overlap of 0.8 in both x and y directions. These windows were limited to a y axis range from 500-700 px.

All the windows were run through the model and for each image that the model predicted a car, the window was drawn on the image. This usually resulted in a series of classifications and windows that overlaped on a car in the videoframe: 

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to try to minimize false positives and reliably detect cars?

In order to improve the detections and remove false positives, a method was used to create a heatmap of classified pixels and create a bounding box of the heatmap to create one single box around the extent of the car. 

![alt text][image4]
This process is extended in the pipeline for the video in a way that collects heatmaps of 10 consecutive frames, sums them together and then thresholds the result to create more stable bounding boxes. A function from the `collections` library called `label` which looks at the heatmap and collects the contiguous pixels into groups. A box is created around the max and min of the groups. The result is a single box around each group of heat pixels, which represent detecteted cars in the video frame. An image of this taken from the video is included below. 


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/IsMhs4fMBuE/0.jpg)](http://www.youtube.com/watch?v=IsMhs4fMBuE)

Here's a [link to my video file](./full_output.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The main filter I used in the video pipeline process was to add "heat" to the heatmaps, label the images, and then draw bounding boxes for each labeled group, which is a car. 

I added a condition in the sliding windows that rejects the smallest boxes. I also accumulated the heatmaps for 10 frames with a `collections` data structure called `deque`. These 10 heatmaps were then summed and a threshold of max=5 was set to develop more stable bounding boxes for cars. The effect is to have a moving average like effect over a 10 frame window. 

Here's an example result showing the final bounding boxes overlaid on a frame of video:

![alt text][image5]
---
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main challenge is reducing the number of false positives. In my video there was a final very quick false positive that still managed to regestire despite the checks I put in and the buffering nature of the heatmap function. I could have removed it by limiting the scope of the sliding windows in that area, but I was hesitant to do that because I didn't want to have a one-off fix -- I am interesting in the algorithm working in general cases, not just this video example. 

In the tuning of the algoritm there were also cases where the cars on the opposite side of the road were detected. While these were eliminated by narrowing the scope of the windows on the x-axis (a bit) this same algorithm would likely pick up these detections on a two-lane road. And this may be just as well, since the car would need to be able to interpret the scenarios in which there is oncoming traffic.

There is some probably some work that could be done in terms of re-trainig the model using these false positives as labeled cases of "non-car" images. Another thing that can be improved is to clear the heatmap queue before running the video stream, as the video starts the first few frames with some leftover boxes from when the video was processed last. 

