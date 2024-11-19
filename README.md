# Traffic Sign Recognition

This project implements a traffic sign recognition system using a deep learning model trained on the "Chinese Traffic Sign" dataset. The model uses MobileNetV2 as a baseline, enhanced with additional layers for improved accuracy. It also provides a Streamlit-based user interface for easy testing and deployment.

---

## Features

- **Pretrained MobileNetV2**:
  - Uses a pretrained model as the backbone for feature extraction.
- **Enhanced Architecture**:
  - Adds BatchNormalization and Dropout for better regularization.
- **Interactive Deployment**:
  - Streamlit-based user interface for uploading images and predicting traffic signs.
- **Metrics and Validation**:
  - Supports metrics such as accuracy, precision, recall, and F1-score.
- **Handles Class Imbalance**:
  - Incorporates class weights and data augmentation to improve performance.

---

## Dataset Details

- **Source**: "Chinese Traffic Sign" dataset.
- **Contents**:
  - `annotations.csv`: Contains image metadata, bounding box coordinates, and traffic sign categories.
  - `images/`: Contains 5,998 traffic sign images categorized into 54 classes.

---

## Requirements

- **Python Version**: 3.8 or later
- **Packages**:
  - TensorFlow
  - Streamlit
  - Pandas
  - Numpy
  - Matplotlib
  - Seaborn
  - scikit-learn

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd traffic-sign-recognition
2. Create and activate a virtual environment (optional but recommended).

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install required dependencies.

    ```bash
    pip install -r requirements.txt
    ```

4. Download the pretrained models  and place them in the  directory. These models are used for classification.

5. Run the Flask application.

    ```bash
    python app.py
    ```

6. Open your browser and  interact with the app.

## Usage

1. Start to run the app.
2. Upload a `.jpg` image using the file upload form.
3. After the image is uploaded, the app will display the classification results from both the baseline and enhanced models.
4. The app predicts the document category.

## File Structure
.
app.py                   # Streamlit deployment script
train_model.py           # Model training script
enhanced_model.h5        # Trained model file
/datasets/
    annotations.csv      # Dataset annotations
    images/              # Traffic sign images
requirements.txt         # Python dependencies
README.md                # Project documentation


## Models

- **Baseline Model**: A standard MobileNetV2 model trained on document images.
- **Enhanced Model**: A modified MobileNetV2 model with additional layers like Batch Normalization and Dropout for improved performance.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
