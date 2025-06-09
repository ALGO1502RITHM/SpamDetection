# SpamDetection
To detect email Spam
# 📧 Anti-Spam Email Detector

A simple and effective web application that detects whether an email is spam or not using machine learning (XGBoost). Users can paste email text into the form, and the system will predict if it's spam based on key features.

## 🚀 Features

- Paste raw email content and detect spam in real time.
- Extracts key textual features like:
  - Number of links
  - Word count
  - Presence of common spam keywords (e.g. "offer", "free", "click")
  - Capitalization patterns
- Scaled and processed features using `StandardScaler`.
- Prediction powered by an XGBoost model.
- Simple and intuitive frontend built with HTML, CSS, and JavaScript.
- REST API built using Flask.

---

## 🖥️ Tech Stack

- **Frontend:** HTML, CSS, JavaScript (Fetch API)
- **Backend:** Python (Flask)
- **Machine Learning:** XGBoost, Scikit-learn
- **Other Tools:** Joblib, Regex, Flask-CORS
