<div align="center">

# 🌸 Machine Learning Model Comparison
### K-Nearest Neighbors • Decision Tree • Gaussian Naïve Bayes

A comprehensive comparison of three classical supervised machine learning algorithms on the **Iris Dataset**, including hyperparameter tuning, performance evaluation, data visualization, and an interactive desktop application.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C)

</div>

---

# 📖 Overview

Choosing the right machine learning model requires more than comparing accuracy.

This project evaluates three widely used supervised learning algorithms on the Iris dataset using a consistent preprocessing pipeline, hyperparameter optimization, and multiple evaluation metrics.

The goal is to understand how each algorithm behaves, where it performs well, and the trade-offs between accuracy, interpretability, and computational efficiency.

---

# ✨ Features

✅ Data preprocessing & exploratory analysis

✅ Hyperparameter tuning using GridSearchCV

✅ Cross-validation

✅ Feature importance analysis

✅ Learning curves

✅ Confusion matrices

✅ PCA visualization

✅ Interactive Tkinter application

✅ Performance comparison dashboard

---

# 🤖 Algorithms Compared

| Algorithm | Learning Type | Strength |
|------------|--------------|-----------|
| 🌿 K-Nearest Neighbors | Instance-Based | High accuracy |
| 🌳 Decision Tree | Rule-Based | Easy to interpret |
| 📊 Gaussian Naïve Bayes | Probabilistic | Fast training |

---

# 📊 Dataset

**Dataset:** Iris Dataset (UCI)

| Property | Value |
|----------|------:|
| Samples | 150 |
| Features | 4 |
| Classes | 3 |
| Target | Iris Species |

Features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Classes:

- Setosa
- Versicolor
- Virginica

---

# 📂 Project Structure

```text
ml-assignment
│
├── data/
│   └── preprocessing.py
│
├── models/
│   └── train_evaluate.py
│
├── ui/
│   └── app.py
│
├── results/
│
├── main.py
├── run_all.py
├── requirements.txt
└── README.md
```

---

# 🔄 Machine Learning Workflow

```text
             Iris Dataset
                  │
                  ▼
          Data Preprocessing
                  │
                  ▼
        Exploratory Data Analysis
                  │
                  ▼
         Train/Test Split (80/20)
                  │
                  ▼
      Hyperparameter Optimization
            (GridSearchCV)
                  │
                  ▼
     Train Three ML Algorithms
                  │
                  ▼
     Model Evaluation & Comparison
                  │
                  ▼
      Interactive Prediction GUI
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Quratulaien/comparision-of-three-models.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the complete pipeline

```bash
python run_all.py
```

Launch the GUI

```bash
python ui/app.py
```

---

# 📈 Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- 5-Fold Cross Validation
- Learning Curves

---

# 📊 Visualizations

The project automatically generates:

- 📈 Metrics Comparison
- 📉 Cross Validation Results
- 🌳 Decision Tree Visualization
- 📊 Feature Importance
- 🔥 Correlation Heatmap
- 📌 Pair Plot
- 📦 Box Plots
- 🎯 PCA Projection
- 📈 Learning Curves
- 📉 Hyperparameter Tuning Results

---

# 🖥 Desktop Application

The project includes a Tkinter-based desktop interface that allows users to classify Iris flowers interactively.

### Features

- Live feature sliders
- Algorithm selector
- Confidence scores
- Prediction probabilities
- One-click sample inputs
- Compare predictions from all three models

---

# 📊 Results

| Model | Accuracy | F1 Score | Speed | Interpretability |
|------|----------:|----------:|------:|:---------------:|
| KNN | ~97% | ~0.97 | ⭐⭐⭐ | ⭐ |
| Decision Tree | ~97% | ~0.97 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Naïve Bayes | ~93% | ~0.93 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

# 💡 Key Insights

- Decision Trees provide strong performance while remaining easy to interpret.
- KNN achieved comparable accuracy but requires storing the training data.
- Gaussian Naïve Bayes trains extremely quickly but is affected by correlated features.
- Petal Length and Petal Width contribute the most to classification performance.

---

# 🚀 Future Improvements

- Support Vector Machine
- Random Forest
- XGBoost
- Neural Networks
- Model Deployment with FastAPI
- Streamlit Web Application

---

# 📚 Learning Outcomes

This project strengthened my understanding of:

- Supervised Machine Learning
- Feature Engineering
- Model Evaluation
- Hyperparameter Optimization
- Data Visualization
- Building Interactive ML Applications

---

# 👩‍💻 Author

## Qurat ul Aien

**Aspiring AI/ML Engineer**

Interested in

- Machine Learning
- Deep Learning
- Large Language Models
- Computer Vision
- Explainable AI

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a star!

</div>
