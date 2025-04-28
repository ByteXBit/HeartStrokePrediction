<div style="background-color: #f5f5f5; padding: 20px; border-radius: 10px;">

# <span style="text-decoration: underline;">HeartStrokePrediction</span>

## <span style="text-decoration: underline;">Introduction</span>
This project aims to analyze and predict the likelihood of a stroke based on various health factors using Machine Learning (ML) and Deep Learning (DL) models.  
We performed detailed **data cleaning**, **visualization**, **feature engineering**, **model building**, and **evaluation**.  
This was done as part of an academic project to understand end-to-end Machine Learning pipelines and Deep Learning modeling.

---

## <span style="text-decoration: underline;">1. Data Preprocessing</span>

### Loading and Exploring Data
- The dataset `healthcare-dataset-stroke-data.csv` was loaded using **pandas**.
- Basic information such as **shape** and **statistical description** was obtained to understand the dataset.

```python
data.shape
data.describe()
Handling Missing Values
The bmi column had missing values.
Missing bmi values were filled with the mean value of BMI, grouped by gender, marital status, and age group to ensure contextual relevance.
The id column was dropped as it does not provide any predictive value.
Feature Engineering
An age_group column was created to categorize individuals as Infant, Child, Adolescent, Young Adult, Adult, or Old Aged based on their age.
Rows with ambiguous gender ("Other") were removed to maintain binary classification.
<span style="text-decoration: underline;">2. Data Visualization and Analysis</span>
We used Plotly Express, Seaborn, and Matplotlib to visualize various feature relationships with stroke:

Gender vs Stroke
<img src="https://github.com/user-attachments/assets/7ba46767-37fb-4ed4-9fc8-3013fc0d1d9d" alt="Gender vs Stroke" style="box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2); border-radius: 10px;">

Age vs Stroke
<img src="https://github.com/user-attachments/assets/98995bf4-b5fa-4447-b505-74faea007988" alt="Age vs Stroke" style="box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2); border-radius: 10px;">

Hypertension vs Stroke
<img src="https://github.com/user-attachments/assets/da44c94d-ac76-4291-a30c-c66e98e6f757" alt="Hypertension vs Stroke" style="box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2); border-radius: 10px;">

<span style="text-decoration: underline;">3. Handling Class Imbalance</span>
The dataset was highly imbalanced (very few people had strokes).

We used upsampling and SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset:

Upsampling involved duplicating minority class samples.
SMOTE generated synthetic samples to balance the classes.
<span style="text-decoration: underline;">4. Data Scaling</span>
Important continuous features like age, avg_glucose_level, and bmi were normalized using MinMaxScaler or StandardScaler for consistent scaling across ML models.
<span style="text-decoration: underline;">5. Machine Learning Models</span>
We implemented and trained several ML models:

Model	Key Details
Extra Trees Classifier	Ensemble method
Random Forest Classifier	Ensemble method, tuned with hyperparameters
XGBoost Classifier	Advanced boosting technique
Gradient Boosting Classifier	Boosted ensemble method
Each model was:

Trained on the training set.
Evaluated on both training and testing sets.
Assessed using Accuracy Scores, Classification Reports, and Confusion Matrices.
Example model training:

Python
etc_model = ExtraTreesClassifier()
rfc_model = RandomForestClassifier(n_estimators=29, max_leaf_nodes=900, max_features=0.8, criterion='entropy')
xgb_model = XGBClassifier(objective="binary:logistic", eval_metric="auc")
gbc_model = GradientBoostingClassifier(max_depth=29, min_samples_leaf=4, min_samples_split=13, subsample=0.8)

models = [etc_model, rfc_model, xgb_model, gbc_model]

for model in models:
    model.fit(x_train, y_train)
<span style="text-decoration: underline;">6. Deep Learning Models (ANN)</span>
To enhance prediction accuracy, Deep Learning models were built using TensorFlow and Keras.

Basic Neural Network
A simple ANN model with two hidden layers.
Compiled with adam optimizer and binary_crossentropy loss.
Trained for 50 epochs.
Python
model = Sequential([
    Dense(16, input_dim=X.shape[1], activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
Improved Neural Network
Handled class imbalance using SMOTE.
Added Dropout layers to prevent overfitting.
Used EarlyStopping callback to terminate training early if no improvement was observed.
Python
model = Sequential([
    Dense(32, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
<span style="text-decoration: underline;">7. Evaluation</span>
Classification reports showed precision, recall, f1-score for both classes (Stroke / No Stroke).
Confusion matrices visualized True Positives, True Negatives, False Positives, and False Negatives.
Test Accuracy was printed at the end for both ML and DL models.
<span style="text-decoration: underline;">8. Conclusion</span>
Through this project, we learned how to:

Preprocess messy real-world healthcare datasets.
Handle missing values thoughtfully.
Balance imbalanced datasets using resampling and SMOTE.
Build, train, and evaluate multiple machine learning models.
Build deep learning models using Keras and TensorFlow.
Evaluate models using statistical and visual metrics.
This end-to-end project helped us understand the critical steps in building a reliable prediction system for sensitive applications like healthcare.

<span style="text-decoration: underline;">📊 Technologies Used</span>
Python
Pandas, Numpy
Matplotlib, Seaborn, Plotly Express
Scikit-learn
XGBoost
TensorFlow, Keras
imbalanced-learn (SMOTE)
<span style="text-decoration: underline;">📁 Dataset</span>
Healthcare Dataset for Stroke Prediction (available on Kaggle).
</div> ```
