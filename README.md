# Depth Regression Model for RGBd Image Dataset

This project focuses on designing and evaluating a neural network for monocular depth estimation. The model predicts the distance of objects in an image from the camera using a *single* RGB image. Depth estimation is a vital task in computer vision, enabling machines to infer the 3D structure of a scene from 2D data.

## **Applications**
Depth estimation plays a crucial role in:
- **Autonomous Driving**: For obstacle detection and path planning.
- **Augmented Reality (AR)**: For spatial understanding and object placement.
- **Robotics**: For navigation, object manipulation, and spatial awareness.

---

## **Objective**
The primary objective is to develop a model capable of accurately predicting depth maps from RGB images. The performance is evaluated across key metrics, including:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**

---

## **Model Overview**
The project is implemented as an **ensemble model** combining two networks:
1. **GlobalDepthNetwork**:
   - Focuses on capturing global depth information from the entire image.
   - Extracts high-level contextual features to understand the overall scene structure.

2. **LocalGradientNetwork**:
   - Focuses on capturing local gradient and edge information.
   - Enhances the model's ability to predict fine-grained depth details and spatial relationships.

The ensemble approach ensures both global context and local details are effectively captured for robust depth prediction.

---

## **Dataset**
The dataset used in this project includes RGB and depth images, enabling supervised learning for depth regression tasks. You can explore the dataset here:
[Dataset Link](https://www.kaggle.com/code/kmader/showing-the-rgbd-images/notebook)

---

## **Project Features**
1. **Model Design**:
   - Implementation of an ensemble model combining GlobalDepthNetwork and LocalGradientNetwork.
   - Use of convolutional layers for feature extraction and depth prediction.

2. **Evaluation Metrics**:
   - Performance is measured using standard regression metrics like MSE, RMSE, and MAE.

3. **Visualization**:
   - Depth maps are visualized for qualitative analysis.

4. **Code Structure**:
   - Well-documented code for easy reproduction and further experimentation.

---

## **Technologies Used**
- **Python**
- **TensorFlow/Keras**: For building and training the neural networks.
- **NumPy**: For numerical computations.
- **Matplotlib**: For visualization of results.
- **Jupyter Notebook**: For interactive development.

---

## **How to Run the Project**
1. **Clone the Repository**:
   ```bash
   git clone <https://github.com/Anees1774/Depth-Regression-RGBd.git>
   cd <Depth-Regression-RGBd>
   ```
2. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
   Open the `depth_regression_final.ipynb` file and follow the cells step-by-step.

3. **Dataset Preparation**:
   - Download the dataset from the [link](https://www.kaggle.com/code/kmader/showing-the-rgbd-images/notebook).
   - Place the dataset in the appropriate directory.



