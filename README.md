# Code_Alpha_Handwritten_Character_Recognition
This project aims to recognize handwritten alphabet characters using convolutional neural networks (CNNs). It utilizes the A-Z Handwritten Data dataset and implements a CNN model to classify the alphabet characters.

#Dataset : 
The dataset used in this project is the A-Z Handwritten Data, which contains 26 classes of handwritten alphabet characters (A-Z). Each image in the dataset is represented as a 28x28 grayscale image.

#Requirements : 
Python 3
Libraries: matplotlib, cv2, numpy, pandas, scikit-learn, tqdm, Keras (TensorFlow backend)

#Model Training : 
The CNN model architecture consists of multiple convolutional layers followed by max-pooling layers.
The model is trained using the Adam optimizer with a learning rate of 0.001 and categorical cross-entropy loss.
Early stopping and learning rate reduction on plateau are employed as callbacks during model training to prevent overfitting.

#Saved Model : 
The trained CNN model is saved as model_v5 in the Google Drive directory. This saved model can be loaded and used for making predictions on new data.

#Evaluation : 
The trained model is evaluated on a test set to measure accuracy and loss.
The final model is saved for future use.

