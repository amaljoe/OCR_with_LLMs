# Handwritten Text Full-Page OCR

This project focuses on improving OCR for handwritten texts, extending TrOCR to handle full-page structured text (paragraphs and essays). It includes scripts and methods used throughout the development process, from initial experimentation to the final optimized methodology.

---

## Inference Example
<img width="1728" alt="Screenshot 2024-11-30 at 2 28 11 AM" src="https://github.com/user-attachments/assets/60670aaf-b1dc-44e1-803d-fa1c835ca392">

---

## **How to Run the Project**

1. **Run the Application with UI**  
   Launch the application with a graphical interface using the command:
   ```bash
   python ui.py


2. **Generate Synthetic Data**  
   Generate labeled synthetic data using the provided notebook:
    - Open and run `synthetic_data_generation_final.ipynb` in your Jupyter Notebook or preferred environment.

---

## **Methodology Summary**
The project used a systematic approach to train a model for detecting bounding boxes of high-quality text patches:

1. **Dataset Creation**:
    - Divided images into overlapping patches of fixed height (~2× font size) and full width.
    - Applied TrOCR to detect lines in patches.
    - Filtered good patches based on confidence scores and removed duplicates using BLEU scores.

2. **Model Training**:
    - Trained a YOLO model (initialized with YOLOv11 weights) on approximately 1,100 labeled examples.
    - Trained for 100 epochs, achieving a final validation loss of <1.

---

## **Key Features**

- **Enhanced OCR for Handwritten Text**: Focuses on structured layouts while avoiding complex graphs or unstructured data.
- **Synthetic Data Generation**: Automated label generation using a brute-force patching approach and TrOCR.
- **Efficient Text Detection**: Optimized bounding box detection using a YOLO model to streamline the pipeline.
- **Full-Page OCR Pipeline**: Handles full-page text detection and recognition for structured text.

---

## **Limitations**

- **Out-of-Vocabulary Characters**: Some characters, like the Greek letter sigma, are mapped to visually or semantically similar known characters due to model limitations.
