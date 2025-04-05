# Histopathologic-Cancer-Detection
Unsupervised Learning Week 3 Project

# CNN Cancer Detection Kaggle Mini-Project

The problem at hand is to detect at least one pixel containing tumor tissue in a section of a digital pathology scan. Given a folder of 220,025 such images will train a convolutional neural network to determine which images contain cancerous cells. Using our model we will predict whether the 57,458 test images contain at least one tumor cells or not.

## EDA

For our EDA we will check the counts of elements with tumor (1) and without tumor (0). We will displayed 10 images from each category to examine any obvious visual differences.

From this analysis we find a small class imbalance with about 60% of our data having label 0 and 40% label 1. Given that we are using a convolutional neural network and that neural networks are relatively robust to class imablances relative to other learning methods, this seems to be a reasonable split.

Some of the images with tumor cells appear more purple/violet but with viewing only 10 samples of each it's impossible to say if this is significant or not.

![image](https://github.com/user-attachments/assets/24671389-aaf3-483f-a1ce-b9a1ba515bd2)

![image](https://github.com/user-attachments/assets/837894f3-8101-4492-999d-86254a2b204d)

![image](https://github.com/user-attachments/assets/288000d7-1ee4-47d0-b7fe-e452b2027635)

![image](https://github.com/user-attachments/assets/4daa5af2-64e7-46e3-bc3b-659c7f04d392)

## Preprocessing

To prepare our dataset for training, we began by splitting the original dataframe into training and validation sets using an 80/20 split. From inspection of the images, they are 96 pixels by 96 pixels.

We created data generators for our training and validation datasets in order to flow the data into our model during training. In both cases we normalized the pixel values by rescaling the RGB values to a range of 0 to 1. We experimented with other data augmentation techniques, but with trial and error couldn't find a configuration that significantly impacted the model training and performance and so this was abandoned in our final model. In our generators we used a batch size of 128 both to speed up training and allow the model to generalize adequately to unseen data.

Finally, because of the class imbalance identified in our EDA, we set class weights to be used when training our model.

## CNN Model

### Compiling model

For our binary image classification task we used a convolutional neural network. We experimented with much larger layer sizes, but those seemed more prone to overfitting as the validation accuracy reached its peaked at a relatively low epoch. The final network strikes a balance between model capacity to learn complex features and regularization techniques to prevent overfitting.

The model has two convolutional blocks that take an input of RGB 96x96x3 matrices. The first convolutional block has two layers of 32 filters of size 3x3. This is to extract lower level features from our images. The second block has two layers of 64 filters to extract more abstract and higher level features. Both levels contain a max pooling layers in an attempt to focus the network on the most significant features. We also included a dropout rate of 0.4 at each layer in an attempt to prevent overfitting.

After flattening the convolutional output, we fed it into a dense layer with 512 neurons and a ReLu activation function. This stage also includes a dropout to prevent the network overreliance on specific features. Our output uses a sigmoid function that allows for a binary classification.

The model uses Adam as an optimizer and cross entropy as its loss function. During training we also monitor accuracy and AUC.

### Training model

The model was fit using the data generators defined in our Preprocessing section with the number of steps per epoch defined so that all images are seen once by the model in each epoch. Because of the class imbalance, class weights are taken into account. The model was allowed to run for 100 epochs.

We set up severall callbacks during training. The checkpoint callback was included in order to easily retrieve the best epoch of our model for later predictions. The learning rate was reduced at a rate of 0.9 when the validation loss failed to improve for 2 epochs in a row.

