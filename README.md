# **Predicting Focus Score from Digital Behavior**

![UTA-DataScience-Logo](https://github.com/user-attachments/assets/fec1b411-bda5-437a-9eb8-08a018eb84ae)

## **One-Sentence Summary**
This project investigates whether a person's digital habits like screen time, app switching, and sleep, can be used to predict their daily focus level using machine learning, based on data from the [Mental Health and Digital Behavior 2020–2024 Kaggle dataset](https://www.kaggle.com/datasets/atharvasoundankar/mental-health-and-digital-behavior-20202024).

## **Overview**

The idea of this project is simple: can we estimate how focused a person feels on a given day just by looking at how they use their phone?

We live in an era where digital devices collect vast amounts of behavioral data passively, without requiring users to fill out forms or check boxes. If we could use this passively collected data to estimate meaningful cognitive states like focus, it could revolutionize how we build tools for mental health and productivity.

This project approaches the task as a supervised regression problem, where the goal is to predict the focus_score (a self-reported daily score) based on behavioral signals such as screen time, app switching frequency, notification count and sleep. We intentionally avoided using mood or anxiety scores as predictors to simulate a realistic, passive-sensing setup that could work in practice.

Three models were developed and compared:

- Linear Regression as a baseline

- Random Forest Regressor to capture non-linearities

- XGBoost Regressor for performance-focused modeling

The results show that even without deep models or complex tuning, we can predict focus with low error, opening a path for more scalable and privacy-conscious wellbeing applications.

--- 

## **Summary of Work Done**
We started with data exploration, cleaned the dataset, visualized distributions and relationships, scaled features, trained three regression models, evaluated them using standard metrics (R², RMSE, MAE), and visualized the results to understand both performance and prediction behavior.

### **Data**
The dataset was obtained from Kaggle and includes daily snapshots of digital and psychological data from users between 2020 and 2024. Each row represents one person’s activity on one day.

Input Features (used for modeling):
- daily_screen_time_min: Total screen time in minutes

- num_app_switches: Number of times the user switched between apps

- sleep_hours: Hours of sleep

- notification_count: Number of notifications received

- social_media_time_min: Minutes spent on social media

Output Variable:
- focus_score: A continuous number representing self-reported focus that day

Other Variables (not used in model):
- mood_score, anxiety_level: Avoided to prevent label leakage

Size & Format:
- CSV file with ~500 rows and 8 columns

- We performed an 80/20 train-test split for evaluation

---
### **Preprocessing**

Before modeling, the following steps were taken:

- Checked missing values: none where found

- Selected only behavioral features as inputs

- Standardized all input features using StandardScaler, ensuring models treated them equally

---
### **Data Visualization**

Understanding the data before modeling is crucial, especially in behavioral science, where features often behave in non-obvious ways. Below are key visualizations and interpretations that shaped how we approached the problem.

<img width="592" height="775" alt="Image" src="https://github.com/user-attachments/assets/9ca6814a-e980-4556-baa2-7e4f845e26d7" />

<img width="588" height="787" alt="Image" src="https://github.com/user-attachments/assets/867fa796-fbeb-43db-abf4-6883f37b7ec6" />

<img width="592" height="396" alt="Image" src="https://github.com/user-attachments/assets/8a69cf9a-0a88-49b4-9876-d9dce0be462e" />

**Feature Distributions**
We examined the distribution of each input feature to detect outliers and understand shape characteristics:

1. Screen Time: 
- The daily_screen_time_min distribution is right-skewed, with most values clustered between -1 and 1 (i.e., near the mean), but with several high outliers. This confirms that while most users fall in a narrow usage band, a few exhibit extremely high screen time. These outliers could represent binge behavior or atypical days.

2. App Switching:
- The num_app_switches distribution is fairly multimodal and jagged. This reflects real behavioral patterns: some users frequently switch between apps (perhaps due to distraction), while others use fewer, more consistently.

3. Sleep:
- sleep_hours is nearly normal but with slightly heavier tails, indicating some outliers in sleep behavior, likely due to late nights or sleep tracking inconsistencies.

4. Notifications:
- notification_count shows a strong central peak, with values mostly centered around the mean. A few high-notification days stretch the upper end, possibly reflecting days of high digital activity or interruptions.

5. Social Media Time:
- This distribution is notably skewed right. Most people spend a relatively modest amount of time on social media, but a small minority logs significantly more time, suggesting potential compulsive usage or reliance on social platforms.


<img width="783" height="672" alt="Image" src="https://github.com/user-attachments/assets/e8c03635-1b55-4009-b5da-6812b2ea6084" />

**Correlation Heatmap**
This heatmap shows the Pearson correlation coefficients between all behavioral features and the target focus_score. Key insights:

- Focus score has a negative correlation with most behavioral features.

- Highest negative correlation is with notification_count (-0.34)

- daily_screen_time_min (-0.31) and num_app_switches (-0.28) also show meaningful inverse relationships

- Surprisingly, sleep hours have nearly no correlation with focus in this dataset : a finding that may be due to either low variance or noise in self-reported sleep.

These weak-to-moderate correlations suggest the signal is subtle, which is typical in mental health data and makes predictive modeling both challenging and meaningful.

---
### **Problem Formulation**

The projectt was a supervised regression problem:

- Input: A 5-dimensional vector of behavioral features

- Output: A continuous focus_score

Models used:

- Linear Regression (baseline, interpretable)

- Random Forest Regressor (non-linear, robust)

- XGBoost Regressor (optimized gradient boosting)

---
### **Training Setup**

- Software: Python 3.11, scikit-learn, XGBoost, matplotlib, pandas

- Hardware: CPU only; training took seconds per model

- Hyperparameters: Defaults for all models 

- Training time: Instantaneous due to dataset size

We used RMSE, MAE, and R² Score to evaluate the models.

---

### **Performance & Comparison**

To model the relationship between digital behaviors and a user's self-reported focus score, we formulated this as a supervised regression task. Our goal was to predict a continuous focus_score variable from behavioral features using several standard machine learning models. We selected three models that balance interpretability, flexibility, and performance:

<img width="592" height="577" alt="Image" src="https://github.com/user-attachments/assets/71f280e0-e962-4272-9422-f0118a41e964" />

**Linear Regression**
- We began with Linear Regression as a baseline model. Its simplicity makes it easy to interpret and useful for understanding which features have linear associations with the target variable. While it assumes linearity and independence among features.

- R² Score: 0.411

- RMSE: 0.484

- MAE: 0.397

- Visualization:
The plot of actual vs. predicted focus scores shows a moderate positive trend, although predictions cluster and fail to fully capture variability at higher focus levels. This indicates that while the model captures some signal, it's limited in flexibility.

<img width="588" height="580" alt="Image" src="https://github.com/user-attachments/assets/9c56cfb7-c7fc-4243-b551-00f2277965db" />

**Random Forest**
To capture nonlinear relationships and interactions between features, we introduced a Random Forest model. This ensemble method aggregates predictions from multiple decision trees, which allows it to model complex patterns in the data without strong assumptions about linearity.

- R² Score: 0.442

- RMSE: 0.471

- MAE: 0.390

- Visualization:
The predicted vs. actual scatterplot improved in tightness around the diagonal compared to linear regression, suggesting the model was better at adapting to subtle trends in the data. However, variance in predictions remains, especially at higher actual focus scores.

<img width="590" height="582" alt="Image" src="https://github.com/user-attachments/assets/74bfa97e-2650-4864-aa27-08d1c2680279" />

**XGBoost Regressor**
Finally, we experimented with XGBoost, a powerful gradient boosting algorithm known for its predictive strength in structured data tasks. It builds trees sequentially, correcting errors at each step and often outperforming bagged methods like random forests.

- R² Score: 0.345

- RMSE: 0.511

- MAE: 0.411

- Visualization:
Despite XGBoost's reputation, it underperformed here. The predicted values show more scatter, and the regression line diverges more frequently from the true scores. One possible reason is overfitting to noise in the small dataset or insufficient tuning.


<img width="890" height="591" alt="Image" src="https://github.com/user-attachments/assets/63bc5696-0fb1-4b21-97d2-e204cc789d72" />

- Best Overall: Random Forest achieved the highest R² and lowest error metrics, balancing flexibility with generalization.

- Most Interpretable: Linear Regression offered the clearest insight into feature influence, despite slightly lower performance.

- Underperformer: XGBoost surprisingly lagged, likely due to overfitting or data limitations.


---

### **Conclusions**

This project demonstrates that:

- Digital behavior alone can predict focus levels with decent accuracy

- Even without advanced models or tuning, traditional regression methods are effective

- It is feasible to imagine tools that assess wellbeing or productivity using only passively collected data

- Random Forest delivered the best trade-off between interpretability and performance, while XGBoost showed promise but needs tuning.

---

### **Future Work**

- Apply hyperparameter tuning (especially for XGBoost)

- Use cross-validation to validate results more robustly

- Explore residual analysis to identify biases or errors

- Test ensemble models that combine the strengths of multiple approaches
  
--- 

## **How to reproduce**

To fully reproduce the results:

- Clone this repo and install the following: pip install pandas numpy scikit-learn matplotlib xgboost

- Download the dataset from Kaggle: https://www.kaggle.com/datasets/atharvasoundankar/mental-health-and-digital-behavior-20202024

- Place mental_health_digital_behavior_data.csv in your working directory

- Open and run Digital_Wellbeing_Focus_Prediction.ipynb

---

## **Repository Structure**

- Digital_Wellbeing_Focus_Prediction.ipynb	:Main notebook with data prep, training, evaluation
- README.md	(This file): full narrative explanation

---

## **Citations**
Dataset: [Mental Health and Digital Behavior 2020–2024, Kaggle](https://www.kaggle.com/datasets/atharvasoundankar/mental-health-and-digital-behavior-20202024)






