# Histopathologic-Cancer-Detection
Unsupervised Learning Week 3 Project

# CNN Cancer Detection Kaggle Mini-Project

The problem at hand is to detect at least one pixel containing tumor tissue in a section of a digital pathology scan. Given a folder of 220,025 such images will train a convolutional neural network to determine which images contain cancerous cells. Using our model we will predict whether the 57,458 test images contain at least one tumor cells or not.

## EDA

For our EDA we checked the counts of elements with tumor (1) and without tumor (0). We display 10 images from each category to examine any obvious visual differences. We also check the disribution of intensity for each of the RGB channels to see if any obvious patterns emerge.

Our analysis revealed a small class imbalance, with approximately 60% of samples labeled as 0 (no tumor) and 40% as 1 (tumor present). The class distribution is visualized below using both bar and pie charts. This imbalance is not extreme and should be manageable.

To explore visual patterns associated with each class, we randomly sampled and displayed 10 images from each class.
Images labeled as containing tumor cells often show a more violet or purplish hue, although this trend was not consistent enough across enough samples to draw statistical conclusions.

To explore the potential differences in color between the two classes, we computed the mean intensity of each RGB channel for 500 randomly sampled images from each class. We then plotted histograms comparing the distributions of red, green, and blue intensities for tumor vs. non-tumor images. Tumor images tend to have slightly higher red and blue channel intensities compared to green, which may explain the common purple tint noted in visual inspection. However, the overlap between the two classes remains substantial across all channels. This suggests that color alone is likely insufficient for high accuracy classification. However, a CNN may be able to extract spatial and more subtle color patterns that could correlate with tumor presence.

![image](https://github.com/user-attachments/assets/24671389-aaf3-483f-a1ce-b9a1ba515bd2)

![image](https://github.com/user-attachments/assets/837894f3-8101-4492-999d-86254a2b204d)

![image](https://github.com/user-attachments/assets/288000d7-1ee4-47d0-b7fe-e452b2027635)

![image](https://github.com/user-attachments/assets/4daa5af2-64e7-46e3-bc3b-659c7f04d392)

![image](https://github.com/user-attachments/assets/e6b9259b-b88d-429f-b08a-5bd2f2932daa)

## Preprocessing

To prepare our dataset for training, we began by splitting the original dataframe into training and validation sets using an 80/20 split. All images are uniformly sized at 96×96 pixels, so we used this as the target resolution for all model input.

To efficiently feed data into the model during training, we used Keras’ ImageDataGenerator to create data generators for both the training and validation sets. In both cases, we rescaled the RGB pixel values to the range [0, 1] by dividing by 255, a common normalization step that helps neural networks converge faster. Although we initially experimented with various data augmentation techniques (e.g., rotation, flipping, zoom), we did not observe consistent improvements in accuracy and it came at a cost of efficiency. As a result, we opted to use only rescaling in the final model pipeline to reduce training complexity and runtime. Both generators were configured with a batch size of 128, striking a balance between training speed and generalization performance.

Finally, to account for the class imbalance identified during EDA (~60% label 0, ~40% label 1), we computed class weights using scikit-learn’s compute_class_weight function. These weights were applied during training to help mitigate bias toward the majority class.

## CNN Model

### Compiling

For our binary image classification task, we used a convolutional neural network. We initially experimented with larger architectures, but found they were more prone to overfitting, with validation accuracy plateauing early. The final model strikes a balance between the capacity to learn complex features and regularization techniques to prevent overfitting.

The model consists of two convolutional blocks and takes an RGB image input of size 96×96×3. The first block contains three layers with 32 filters of size 3×3 to extract low-level features. The second block expands to 64 filters per layer to capture more abstract, higher-level features. Both blocks include max pooling layers to help the network focus on the most informative patterns. A dropout rate of 0.2 is applied after each block to reduce overfitting.

After flattening the output from the convolutional layers, we include a dense layer with 512 neurons and ReLU activation. This layer captures global patterns across the image. A higher dropout rate is applied here to prevent the network from over-relying on any particular features. The final output layer uses a sigmoid activation function to support binary classification.

The model is optimized with Adam and trained using binary cross-entropy loss. During training, we monitor both accuracy and AUC to evaluate model performance.

### Training

The model was trained using the data generators described in the Preprocessing section, with the number of steps per epoch set to ensure that all training images are seen once per epoch. Due to class imbalance, class weights were applied to help the model focus more on underrepresented examples. While training was set for a maximum of 50 epochs, we included early stopping to prevent overfitting and unnecessary computation.

### Performance Analysis

#### Training

To evaluate the performance of the CNN model, we tracked accuracy, loss, and AUC (area under the ROC curve) across both training and validation sets during training. Accuracy provides a general sense of how many predictions the model got right, while loss reflects the model’s error in prediction and optimization. AUC is particularly useful in binary classification problems like this one, as it measures the model’s ability to distinguish between classes across all thresholds.

The training accuracy steadily improved over the epochs and reached above 94%, indicating that the model was able to learn effectively from the training data. Validation accuracy showed more fluctuation but followed a generally upward trend, ending around 93%, suggesting good generalization despite some volatility. The training loss decreased consistently, showing that the model continued to improve its fit to the data. While the validation loss was more erratic, it did not show a consistent upward trend, which suggests that overfitting was largely controlled. Finally, the AUC scores remained high for both training and validation sets throughout training, with the training AUC reaching nearly 1.0 and the validation AUC stabilizing around 0.97. These results indicate that the model maintained strong class separation and overall performance on unseen data, even with some variation across epochs.

#### Best epoch performance

The final model was evaluated using its best saved weights, based on validation performance. Key metrics included the confusion matrix, ROC curve, and precision-recall curve, offering a comprehensive view of its classification ability. The confusion matrix shows that the model correctly identified 24,893 negative samples and 16,485 positive samples, with only 1,037 false positives and 1,621 false negatives—indicating strong performance.

The ROC curve illustrates the model’s excellent ability to distinguish between the two classes, with an area under the curve (AUC) of 0.98. Similarly, the precision-recall curve yielded an average precision score of 0.98, confirming the model's high precision across a range of recall values. Together, these metrics suggest that the final model generalizes well and maintains strong predictive performance, making it well-suited for our binary classification of images on whether they contain tumors cells.

# Conclusion

In this project, we developed a convolutional neural network to classify histopathologic images for the presence of tumor tissue. Through careful preprocessing, class imbalance handling, and model tuning, we trained a binary classifier capable of distinguishing between tumor and non-tumor patches. We incorporated early stopping, model checkpointing, and class weighting to improve performance and avoid overfitting.

The model’s performance was evaluated using accuracy, AUC, and loss curves over time, as well as post-training metrics such as a confusion matrix, ROC curve, and precision-recall analysis. These visualizations provided some positive insights into the model's predictive behavior and its ability to balance class identifications.

While the model shows promising results, further improvements could be explored by experimenting with deeper architectures, ensembling methods, or advanced augmentation techniques. As with all neural networks, computation power is a severe limitation, so while larger networks were too computationally costly for this study, they may yeild even better predictive power. Another option for reducing computation time might be to emply a random forest or other classifier to the CNN output, rather than a dense neural layer.

Overall, this project demonstrates the potential of deep learning in medical image classification tasks and highlights the potential for deep learning methods in health diagnoses.
