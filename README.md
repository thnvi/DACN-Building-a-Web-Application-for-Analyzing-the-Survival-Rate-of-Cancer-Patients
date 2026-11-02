# Mortality Analysis 

Building a Web Application for Analyzing the Survival Rate of Cancer Patients is a specialized project in Information Technology Engineering at Nguyen Tat Thanh University, completed by Bui Thi Thanh Vi under the supervision of MSc. Bui Duy Tan.

The project develops a web application that helps analyze cancer patient data. Users can upload a CSV/Excel file or manually enter records directly in the interface. The system automatically preprocesses the data and visualizes the results with pie and bar charts, allowing users to quickly see the proportion of Benign versus Malignant cases.

Beyond visualization, the application integrates machine learning models (Random Forest and Logistic Regression) for prediction, evaluated with accuracy, ROC-AUC, and confusion matrix, along with Kaplan–Meier survival analysis (using the lifelines library). The project also includes a FastAPI backend and Docker scaffold for future extension.

Tech stack: Python, Streamlit, Pandas, Matplotlib/Seaborn, scikit-learn, lifelines, FastAPI, Docker.
Dataset: a cancer diagnostic dataset of 569 samples with M/B labels and features such as radius, texture, perimeter, and area.
