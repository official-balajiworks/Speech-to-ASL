# Speech-to-ASL Alphabet Recognition System

This project is implemented in a single Jupyter Notebook:

```text
Speech to ASL.ipynb
```

The notebook recognizes spoken English, converts the recognized text into
individual alphabet letters, and displays corresponding American Sign Language
(ASL) alphabet images.

## Features

- Downloads the ASL Alphabet dataset from Kaggle.
- Loads and preprocesses ASL alphabet images.
- Trains an image-classification model using transfer learning.
- Uses MobileNetV2 pretrained on ImageNet.
- Classifies images into 26 alphabet classes, from `A` to `Z`.
- Converts text into individual letters.
- Validates ASL images using the trained CNN.
- Uses browser-based speech recognition.
- Converts speech into an ASL alphabet image sequence.

## Notebook

```text
Speech to ASL.ipynb
```

The notebook contains the complete workflow:

1. Install required packages.
2. Download and extract the Kaggle ASL dataset.
3. Configure image size, batch size, random seed, and class names.
4. Load the image dataset.
5. Split the images into training and validation datasets.
6. Build the MobileNetV2-based CNN model.
7. Train the model.
8. Optionally fine-tune the model.
9. Save the trained model.
10. Predict individual ASL alphabet images.
11. Convert text into alphabet sequences.
12. Convert speech into text.
13. Display the corresponding ASL image sequence.

## Dataset

The notebook uses the Kaggle ASL Alphabet dataset:

```text
grassknoted/asl-alphabet
```

The dataset contains 26 alphabet classes:

```text
A, B, C, D, E, F, G, H, I, J, K, L, M,
N, O, P, Q, R, S, T, U, V, W, X, Y, Z
```

The notebook loads approximately 78,000 images:

- 62,400 training images.
- 15,600 validation images.
- 26 classes.

The dataset is downloaded and extracted inside Google Colab.

## Requirements

- Python 3.9 or later.
- Google Colab or Jupyter Notebook.
- TensorFlow.
- NumPy.
- Matplotlib.
- NLTK.
- Kaggle API.
- A browser with microphone access for speech recognition.

Install the required packages:

```python
!pip install -q kaggle
!pip install -q nltk
!pip install -q SpeechRecognition
```

## Kaggle Setup

Configure Kaggle authentication before downloading the dataset.

Upload your `kaggle.json` file to Google Colab and run:

```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

Download the dataset:

```python
!mkdir -p /content/kaggle_data

!kaggle datasets download \
    -d grassknoted/asl-alphabet \
    -p /content/kaggle_data
```

## Model Configuration

The notebook uses the following configuration:

```python
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
```

## Model Architecture

The classifier uses MobileNetV2 as a pretrained feature extractor.

```text
Input Image
    |
Random Flip
    |
Random Rotation
    |
Random Zoom
    |
MobileNetV2 Preprocessing
    |
MobileNetV2 Backbone
    |
Global Average Pooling
    |
Dropout
    |
Dense Layer: 128 Units
    |
Dropout
    |
Softmax Output: 26 Classes
```

The final layer contains 26 output neurons, one for each English alphabet
letter.

## Training

The model is initially trained with the MobileNetV2 base model frozen:

```python
base_model.trainable = False
```

The optimizer is Adam with a learning rate of `0.0001`, and the loss function is
sparse categorical cross-entropy.

The notebook trains the model for five epochs:

```python
EPOCHS = 5

history = cnn_model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)
```

## Fine-Tuning

The notebook also includes an optional fine-tuning stage. The final 30 layers of
MobileNetV2 can be trained while earlier layers remain frozen:

```python
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False
```

A smaller learning rate should be used during fine-tuning:

```python
learning_rate = 1e-5
```

## Image Prediction

The notebook provides a function for predicting the letter in an image:

```python
letter, confidence = predict_asl_image(image_path)
```

The function returns:

- The predicted alphabet letter.
- The model confidence score.

Example:

```python
print("Predicted letter:", letter)
print("Confidence:", confidence)
```

## Text-to-ASL Conversion

The text-processing pipeline:

1. Converts text to lowercase.
2. Removes punctuation.
3. Tokenizes the text.
4. Converts each word into uppercase letters.
5. Displays the corresponding ASL images.

Example:

```text
Input:
I LOVE YOU

Alphabet sequence:
I → L → O → V → E → Y → O → U
```

The project performs alphabet fingerspelling. It does not translate complete
English sentences into ASL grammar.

## Speech-to-ASL Conversion

Run the following function to start the speech-to-ASL workflow:

```python
speech_to_asl()
```

The workflow is:

```text
User Speech
    |
Browser Speech Recognition
    |
Recognized English Text
    |
Text Preprocessing
    |
Alphabet Conversion
    |
ASL Image Retrieval
    |
ASL Image Display
```

The browser must support speech recognition and must be granted microphone
permission.

## ASL Image Validation

For each letter, the notebook searches the relevant dataset folder and predicts
candidate images using the CNN.

An image is accepted only when:

```python
predicted_letter == requested_letter
```

and the confidence is at least:

```python
0.80
```

This helps ensure that the displayed image matches the requested alphabet
letter.

## Important Code Fixes

Make sure the dataset prefetch and model creation statements are separated:

```python
validation_dataset = validation_dataset.prefetch(AUTOTUNE)

base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
```

Define the class names before prediction:

```python
class_names = train_dataset.class_names
```

The model must be trained before using prediction functions:

```python
history = cnn_model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=5
)
```

## Limitations

- The project recognizes static alphabet signs only.
- It does not recognize complete dynamic ASL signs.
- It does not model ASL grammar.
- Speech recognition depends on browser support and microphone quality.
- Prediction performance may vary with lighting, background, camera angle, and
  hand position.
- Horizontal image flipping may not be suitable for every sign-language class.
- The current system spells words letter by letter.

## Future Improvements

- Add webcam-based real-time recognition.
- Add hand detection using MediaPipe.
- Support dynamic signs using video models.
- Add a test dataset and confusion matrix.
- Calculate precision, recall, and F1-score.
- Add early stopping and model checkpoints.
- Create a Streamlit or Gradio interface.
- Add support for complete ASL words and grammar.
- Optimize the model for mobile or edge-device deployment.

## How to Run

1. Open `Untitled20.ipynb` in Google Colab.
2. Configure Kaggle authentication.
3. Run the notebook cells in order.
4. Download and extract the dataset.
5. Train the CNN model.
6. Save the trained model if required.
7. Run image prediction or speech-to-ASL conversion.

To start speech-to-ASL conversion:

```python
speech_to_asl()
```

## Author

**Balaji Arulmani**

## License

The notebook code may be released under the MIT License.

The ASL dataset remains subject to the license and usage conditions specified by
its Kaggle publisher.
