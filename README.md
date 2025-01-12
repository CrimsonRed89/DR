# FIVES Fundus Dataset - Classification and Segmentation

## Overview
This project focuses on analyzing the FIVES Fundus dataset through **classification** and **segmentation** tasks using various deep learning models. The aim is to develop efficient algorithms for ocular disease detection and retinal image analysis.

---

## Tasks Performed

### 1. **Classification**
Applied multiple state-of-the-art models to classify the fundus images into their respective categories. Models used include:

- **VGG16**
- **VGG19**
- **ResNet50**
- **MobileNet**
- **Inception**
- **EfficientNet**
- **Xception**
- **Self-Designed Model** (achieved the best performance)

### 2. **Segmentation**
Performed segmentation tasks to identify and delineate specific regions in fundus images. Models utilized include:

- **ResUNet**
- **Attention-based Models**
- **DeepLab**
- **UNet**
- **CascadeNet**
- **SegNet**
- **PSPNet**

---

## Highlights

### Classification
- Evaluated standard models alongside a custom-designed CNN for superior performance on classification tasks.
- Optimized architectures to improve accuracy, precision, recall, and F1 scores.

### Segmentation
- Implemented advanced architectures to achieve precise segmentation of retinal regions.
- Compared results across popular segmentation frameworks to identify the most effective solution.

---

## Installation and Usage

### Requirements
- Python 3.8+
- TensorFlow 2.x
- Keras
- NumPy
- OpenCV
- Matplotlib
- Other dependencies listed in `requirements.txt`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FIVES-Fundus-Analysis.git](https://github.com/CrimsonRed89/Diabetic-Retinopathy-Segmentation-Classification-FIVES-DATASET.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Notebooks
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the scripts for classification or segmentation tasks:
   ```bash
   python classification.py
   python <choose segmentation model>.py
   ```

---

## Results
The project showcases the potential of deep learning models in the domain of medical imaging. By utilizing both classification and segmentation approaches, the analysis provides a comprehensive understanding of the FIVES dataset and highlights its applicability in diagnosing ocular diseases.

![WhatsApp Image 2024-12-26 at 12 46 13_13fbe18b](https://github.com/user-attachments/assets/1c0ca5ea-8b73-484a-b2b8-014ae153bcb9)
<img width="556" alt="image" src="https://github.com/user-attachments/assets/a9bd8ad8-cb4f-4ef0-9dc3-6256770ea86b" />

---

## Contributions
- **Custom Model Development:** A self-designed classification model surpassing standard architectures.
- **Multi-Model Comparison:** Extensive evaluation of diverse architectures for both tasks.
- **Segmentation Innovation:** Integration of attention mechanisms and advanced frameworks for precise retinal segmentation.

---

## Future Scope
- Extend analysis to include ensemble learning techniques for enhanced performance.
- Explore real-time classification and segmentation applications.
- Integrate explainable AI (XAI) techniques for better interpretability of results.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Acknowledgments
Special thanks to the creators of the FIVES dataset and the open-source community for their invaluable tools and resources.

---

## Contact
For queries or collaborations, feel free to reach out:
- **Email:** aayusharora0304@gmail.com
