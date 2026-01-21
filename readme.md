# Loan Approval Predictor - [Oneloan](https://oneloan.streamlit.app/)

## Why This Project

Loan approval decisions are critical in banking and finance. This project predicts whether a loan will be approved or rejected based on applicant financial and personal information. It demonstrates the use of machine learning for real-world decision-making and provides a simple web interface for users to interact with the model.

---

## How It Works

1. **User Input:**
   The user enters applicant details such as income, loan amount, dependents, education, employment status, loan term, CIBIL score, and asset values in the web interface.

2. **Data Preprocessing:**
   Categorical inputs like `education` and `self_employed` are encoded into numerical values to match the model requirements.

3. **Model Prediction:**
   The trained Logistic Regression model takes the input data and predicts whether the loan is approved or rejected. It also calculates the probability of approval.

4. **Result Display:**
   The prediction and approval probability are displayed in the UI. A bar chart visualizes the approval vs rejection probability for better understanding.
<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/fb5dd73f-8d84-43e3-9cb6-6cde9ea0fedd" />

---

## Folder Structure

```
loan-approval/
│
├── loan.csv                  # Dataset containing applicant and loan details
├── train.py                  # Script to train the machine learning model
├── loan_model.pkl            # Trained machine learning model
├── app.py                    # Streamlit web application
└── requirements.txt          # Required Python packages
```
[Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
---

## Technologies Used

| Technology   | Purpose                                      |
| ------------ | -------------------------------------------- |
| Python       | Programming language for model and web app   |
| Pandas       | Data manipulation and preprocessing          |
| scikit-learn | Machine learning model (Logistic Regression) |
| Streamlit    | Web app framework to build interactive UI    |
| NumPy        | Handling numerical input data                |
| Altair       | Visualizing approval probability in charts   |
| Joblib       | Saving and loading trained models            |

---

## Why These Technologies Were Used

* **Python**: Popular language for data science and machine learning.
* **Pandas**: Simplifies data loading, cleaning, and preprocessing.
* **scikit-learn**: Provides an easy and efficient way to build predictive models.
* **Streamlit**: Fast and simple way to create web apps for ML projects.
* **Altair**: Creates interactive charts to visualize model predictions.
* **Joblib**: Efficient serialization of the trained model for reuse.

---

## Why Logistic Regression

* The problem is a **binary classification**: approved or rejected.
* Logistic Regression provides **probability-based predictions**, which align well with real-world loan decision-making.
* It is **simple, interpretable, and fast**, making it ideal for small datasets and educational projects.
* Outputs are easy to display in a web interface along with **probabilities**, which helps users understand the confidence of the prediction.

---

## How to Run

1. Clone or download the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model (if `loan_model.pkl` is not available):

```bash
python train.py
```

4. Run the Streamlit web app:

```bash
streamlit run app.py
```

5. Open the URL shown in the terminal to access the app.

---

## Deployed Web Application

The application is deployed and accessible online at:
[https://oneloan.streamlit.app/](https://oneloan.streamlit.app/)

---
