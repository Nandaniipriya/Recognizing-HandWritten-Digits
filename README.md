# Handwritten Digit Recognition with Multiple Models. 🔢

<p align="center">
  <img src="https://github.com/Nandaniipriya/Recognizing-HandWritten-Digits/raw/main/assets/overview.png" alt="Banner Image" width="80%">
</p>

## 📋 Overview
This project implements various machine learning algorithms to recognize handwritten digits using the Scikit-learn digits dataset. We compare multiple classification approaches including Multi-Layer Perceptron (MLP), Decision Trees, K-Nearest Neighbors (KNN), Random Forest, and Support Vector Machines (SVM).

## 🎯 Model Performance

<p align="center">
  <img src="https://github.com/Nandaniipriya/Recognizing-HandWritten-Digits/raw/main/assets/graph.png" alt="Graph" width="100%">
</p>

| Algorithm | Accuracy |
|-----------|----------|
| MLP Classifier | 91.47% |
| Decision Tree | 86.11% |
| KNN | 98.06% |
| Random Forest | 97.78% |
| SVM | 99.17% |



## 🛠️ Dependencies
- Python 3.12
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn

## 💻 Model Implementations

### 1. Multi-Layer Perceptron (MLP)
```python
MLPClassifier(
    hidden_layer_sizes=(15,),
    activation='logistic',
    alpha=1e-4,
    solver='sgd',
    learning_rate_init=0.1,
    random_state=1
)
```
<p align="center">
  <img src="https://github.com/Nandaniipriya/Recognizing-HandWritten-Digits/raw/main/assets/MLPClassifier.png" alt="MLPClassifier" width="100%">
</p>

- Input Layer: 64 nodes (8x8 pixel images flattened)
- Hidden Layer: 15 nodes with logistic activation
- Output Layer: 10 nodes (one for each digit 0-9)
- Training iterations: ~189 epochs
- Final training loss: ~0.0113

### 2. Decision Tree Classification
- Basic implementation using `DecisionTreeClassifier`
- Includes confusion matrix visualization

### 3. K-Nearest Neighbors (KNN)
- Implemented with `KNeighborsClassifier`
- Uses Euclidean distance metric
- Optimal K-value determined using elbow method

### 4. Random Forest Classification
- Implemented using `RandomForestClassifier`
- Uses 250 estimators
- Includes confusion matrix heatmap

### 5. Support Vector Machine (SVM)
- Implemented using `SVC`
- Parameters: C=100, gamma=0.001
- Highest accuracy among all models

## 📈 Data Processing
- Dataset: Scikit-learn digits dataset (8x8 pixel images)
- Total samples: 1797
- Features: 64 pixels per image (8x8 flattened)
- Labels: Digits 0-9
- Train-test split: 80-20

## 🚀 Usage

```python
# Install required packages
pip install scikit-learn numpy pandas matplotlib seaborn

# Basic implementation example
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load data
digits = datasets.load_digits()
X = digits.images.reshape((len(digits.images), -1))
y = digits.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model (example with MLP)
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(15,))
mlp.fit(X_train, y_train)

# Make predictions
predictions = mlp.predict(X_test)
```

## 📊 Visualizations
The project includes:
- Sample digits visualization
- Training loss curves for MLP
- Confusion matrix heatmaps
- Error rate plots for KNN
- Performance comparison plots across models

<p align="center">
  <img src="https://github.com/Nandaniipriya/Recognizing-HandWritten-Digits/raw/main/assets/confusion-matrix-mnist.png" alt="confusion-matrix-mnist" width="100%">
</p>

## 🔍 Future Improvements
1. Implement cross-validation
2. Add data augmentation techniques
3. Experiment with different network architectures
4. Implement hyperparameter tuning
5. Test on external handwritten digit datasets
6. Add real-time prediction capability
7. Optimize model performance
8. Add ensemble methods

## 👥 Contributors
- Nandani Priya

## 📬 Contact
For any queries or suggestions, please reach out to:
- 📧 Email: nandani15p@gmail.com
- 💻 GitHub: https://github.com/Nandaniipriya

---
⭐ Don't forget to star this repo if you found it helpful!
