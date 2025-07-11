# Khmer OCR Tool - README

This project implements an end-to-end OCR (Optical Character Recognition) system for Khmer and English languages. It uses a combination of CRAFT for text detection and TrOCR for text recognition, with a focus on Khmer script support.

## Project Overview

The Khmer OCR Tool addresses the challenge of recognizing Khmer text, which is traditionally difficult for standard OCR systems. By combining state-of-the-art models and techniques, this project achieves improved accuracy for both Khmer and English text recognition.

## Features

- **Text Detection**: Uses CRAFT (Character-Region Awareness For Text detection) to identify text regions in images
- **Text Recognition**: Employs TrOCR (Transformer-based OCR) to recognize text from detected regions
- **Dual Language Support**: Optimized for both Khmer and English text
- **Synthetic Data Generation**: Creates training data from text using various fonts
- **Data Augmentation**: Applies transformations to improve model robustness
- **End-to-End Pipeline**: Combines detection and recognition in a seamless workflow
- **Streamlit Interface**: Provides an easy-to-use web interface for OCR

## Dataset Sources

The project uses multiple data sources:
1. Khmer dictionary (44k entries): https://huggingface.co/datasets/seanghay/khmer-dictionary-44k
2. Khmer fonts info previews: https://huggingface.co/datasets/seanghay/khmerfonts-info-previews
3. Khmer Hanuman dataset (100k entries): https://huggingface.co/datasets/seanghay/khmer-hanuman-100k

## Models

- **Text Detection**: CRAFT (https://github.com/clovaai/CRAFT-pytorch)
- **Text Recognition**: TrOCR (https://huggingface.co/microsoft/trocr-large-printed)

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- Transformers
- OpenCV
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Pillow

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/khmer-ocr-tool.git
   cd khmer-ocr-tool
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Clone CRAFT repository:
   ```
   git clone https://github.com/clovaai/CRAFT-pytorch.git
   ```

4. Download CRAFT pre-trained model:
   ```
   wget -q -O CRAFT-pytorch/craft_mlt_25k.pth https://drive.google.com/uc?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ
   ```

### Running the Jupyter Notebook

Open and run the `khmer_ocr_notebook.ipynb` notebook to follow the complete workflow from data preparation to model evaluation.

### Running the Streamlit App

Launch the Streamlit web interface:
```
streamlit run streamlit_app.py
```

## Project Structure

- `khmer_ocr_notebook.ipynb`: Main Jupyter notebook with the complete OCR pipeline
- `streamlit_app.py`: Streamlit web interface for the OCR system
- `models/`: Directory containing model files
- `dataset/`: Directory for storing and processing datasets
- `CRAFT-pytorch/`: CRAFT text detection model repository

## Future Improvements

- Fine-tune TrOCR specifically for Khmer language
- Expand the synthetic data generation with more fonts and styles
- Implement post-processing for text correction
- Add support for more Southeast Asian languages
- Optimize for mobile deployment

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The CRAFT text detection model from CLOVA AI Research
- Microsoft's TrOCR model
- Hugging Face for dataset hosting
- Contributors to the Khmer language datasets
