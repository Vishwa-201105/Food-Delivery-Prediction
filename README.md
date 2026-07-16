# Food Delivery Time Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas" />
  <img src="https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?style=for-the-badge&logo=numpy" />
</p>

## 📌 Overview

Food delivery platforms rely on accurate delivery time estimation to improve customer satisfaction and operational efficiency. This project uses **Machine Learning** to predict food delivery times based on various order and delivery-related factors.

The application is deployed through an interactive **Streamlit dashboard**, allowing users to input delivery details and receive real-time delivery time predictions.

---

## 🚀 Features

- 📈 Predicts estimated food delivery time using a trained Machine Learning model
- 🎯 Real-time predictions through an interactive Streamlit interface
- 📍 Accepts multiple delivery-related inputs
- ⚡ Fast inference using a serialized ML model
- 💻 Clean and user-friendly dashboard
- 📊 Built using Python's Data Science ecosystem

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python 3.9+ |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Model Serialization | Joblib |
| Web Application | Streamlit |

---

## 📊 Input Features

The prediction model considers several parameters including:

- 📏 Delivery Distance (km)
- 🍽️ Food Preparation Time (minutes)
- 👤 Delivery Person Age
- ⭐ Delivery Person Rating
- 🛍️ Order Type
- 🛵 Vehicle Type

These features are processed and passed to the trained model to estimate the delivery time.

---

## 🤖 Machine Learning Workflow

```
Dataset
   │
   ▼
Data Cleaning
   │
   ▼
Feature Engineering
   │
   ▼
Model Training
   │
   ▼
Model Evaluation
   │
   ▼
Save Trained Model (.pkl)
   │
   ▼
Streamlit Application
   │
   ▼
Real-Time Prediction
```

---

## 📂 Project Structure

```
Food-Delivery-Prediction/
│
├── app.py                 # Streamlit application
├── model.pkl              # Trained ML model
├── data.csv               # Dataset (if included)
├── requirements.txt       # Project dependencies
├── README.md              # Documentation
└── assets/                # Images/Screenshots (optional)
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/Vishwa-201105/Food-Delivery-Prediction.git

cd Food-Delivery-Prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## 💻 Application Preview

> **Add screenshots of your Streamlit dashboard here.**

Example:

```
assets/homepage.png
assets/prediction.png
```

Then display them using:

```markdown
![Home](assets/homepage.png)

![Prediction](assets/prediction.png)
```

---

## 📈 Model Output

The application predicts the **Estimated Delivery Time** based on user-provided inputs.

Example:

| Input | Value |
|--------|------:|
| Distance | 7 km |
| Preparation Time | 18 mins |
| Rating | 4.7 |
| Vehicle | Bike |

**Predicted Delivery Time:** **28 Minutes**

---

## 📦 Dependencies

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🎯 Future Improvements

- Improve prediction accuracy using ensemble models
- Add live traffic and weather information
- Integrate map-based distance estimation
- Deploy the application on Streamlit Cloud
- Add model explainability using SHAP
- Support API-based predictions
- Containerize the application using Docker

---

## 🤝 Contributing

Contributions are welcome!

If you'd like to improve this project:

1. Fork the repository
2. Create a new feature branch
3. Commit your changes
4. Open a Pull Request

---

## 👨‍💻 Author

**Vishwa S**

- GitHub: https://github.com/Vishwa-201105

---

## ⭐ Support

If you found this project useful, consider giving it a **⭐ Star** on GitHub. It helps support the project and motivates further development.

---

## 📄 License

This project is intended for educational and portfolio purposes.

Feel free to use and modify it with proper attribution.
