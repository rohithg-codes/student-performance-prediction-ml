#  Student Performance Prediction using Linear Regression (From Scratch)

##  Overview

This project implements **Multiple Linear Regression from scratch using Python**, without using machine learning libraries like Scikit-learn.

The model predicts **student marks** based on real-world behavioral features such as:

* Study hours
* Sleep hours
* Phone usage

---

##  Problem Statement

To predict student performance based on lifestyle and study-related factors using a mathematical machine learning model.

---

##  Model

The model follows:

y = w1*x1 + w2*x2 + w3*x3 + b

Where:

* x1 = study hours
* x2 = sleep hours
* x3 = phone usage
* w = weights
* b = bias

---

##  Training Process

The model is trained using **Gradient Descent**:

1. Initialize weights and bias
2. Predict output
3. Compute error
4. Calculate gradients
5. Update parameters
6. Repeat until convergence

---

##  Dataset

| Study | Sleep | Phone | Marks |
| ----- | ----- | ----- | ----- |
| 2     | 6     | 5     | 40    |
| 3     | 7     | 4     | 50    |
| 4     | 6     | 6     | 55    |
| 5     | 8     | 3     | 65    |
| 6     | 7     | 2     | 70    |
| 7     | 6     | 2     | 75    |
| 8     | 8     | 1     | 85    |
| 9     | 7     | 1     | 90    |

---

##  Tech Stack

* Python
* NumPy

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python main.py
```

---

##  Features

* Built ML model from scratch
* No external ML libraries used
* Multi-variable prediction
* Interactive user input

---

##  Learning Outcomes

* Understanding of linear regression
* Implementation of gradient descent
* Multi-feature modeling
* Mathematical foundation of ML

---

##  Author

**Rohith G**

* GitHub: https://github.com/rohithg-codes
