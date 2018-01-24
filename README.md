# THIS CODE DEMONSTRATES 
1. HOW TO CREATE A NEURAL NETWORK WITH ONLY NUMPY ARRAY 
2. HOW TO SAVE AND SERVE YOUR TRAINED MODEL USING FLASK

## Requirements
 - Python >=2.7 or >=3.4
 - Flask
 ## Visuals
- dataset.png is the visualization of the dataset
- We train NN to find the decision boundary of this dataset 
- After training (prediction.png) shows the NN learned decision boundary

![Image](dataset.png)

## After training
![Image](prediction.png)

## Train and save
USAGE:
1. To train NN and save(pickle) run
 - $ python nn_script.py
> The NN will be trained and saved in static folder

2. To serve the model run
- $ python app.py
 > visit localhost:5000

## NOTE
  - THIS IMPLEMENTATION WAS MADE FOR LEARNING PURPOSES, PLEASE DO NOT SUE ME FOR
    INEFFICIENT IMPLEMENTATION.
