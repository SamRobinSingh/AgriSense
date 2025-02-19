### **Plant Disease Detection Using CNN**

This project uses Convolutional Neural Networks (CNNs) to identify plant diseases from images and provide detailed descriptions and solutions. We use two primary frameworks: TensorFlow (for image prediction and description) and PyTorch (for CNN model implementation).


Overview:

The purpose of this project is to help farmers and agricultural experts identify plant diseases from leaf images. By analyzing an uploaded image, the system predicts the disease class, provides a description, and suggests solutions.

Technologies Used:

Python: Programming language.
TensorFlow & Keras: For loading and running predictions with a pre-trained model.
PyTorch: For implementing a custom CNN architecture.
NumPy, Pandas: Data manipulation and preprocessing.
Image Processing: tensorflow.keras.preprocessing for image loading and preprocessing.
CNN Model Architecture: Developed in PyTorch for customized training and evaluation.

Setup and Dependencies:
Install the required libraries:

```bash
pip install numpy pandas tensorflow torch torchvision
```

Download the trained model files and dataset:

cnn_model1.h5: The pre-trained Keras model file for disease prediction.
Disease_data_final.csv: A CSV file containing disease names, descriptions, and solutions.

Deployment:
TensorFlow Deployment:

Place the cnn_model1.h5 and Disease_data_final.csv files in the same directory as main.py.
Update file paths in the code if necessary.
Running the TensorFlow Code:

Open a terminal and navigate to the project directory.
To deploy this project run
```bash
cd <path-to-your-project>
```
Then, Clone the repository using the command line,
```bash
git clone https://github.com/SamRobinSingh/AgriSense
```

Run the main.py file:
```bash
python Main.py
```
This script:
Loads and preprocesses an input image.
Predicts the disease class using the model.
Retrieves the disease name, description, and suggested solutions from the CSV file.

PyTorch Model Deployment:

Save your PyTorch code in a separate file, e.g., cnn_model_pytorch.py.
Define the CNN architecture and load pre-trained weights if available.

Training and Inference in PyTorch:

If training from scratch, run the script to train the CNN with your dataset.
Save the trained model weights for future deployment.
