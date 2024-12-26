#import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

#add your file path -> can be in csv or excel
file_path = 'name_of_your_dataset' #either in csv or excel
data = pd.read_csv(r"your_file_path") #in ether csv or excel
data = data.drop(columns=["Patient_Name"]).dropna(subset=["Disease"])

#if more information is given add them as well
data["BMI"] = data["Weight_kg"] / ((data["Height_cm"] / 100) ** 2)

label_encoder = LabelEncoder()
data["Disease"] = label_encoder.fit_transform(data["Disease"])

X = data.drop(columns=["Disease"])
y = data["Disease"]

#training the dataset into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

#XGBoostClassifier is used
model = XGBClassifier(scale_pos_weight=class_weights_dict, use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

#prediction of the diseases along with its accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
confusion_mat = confusion_matrix(y_test, y_pred)

#printing the results
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", confusion_mat)
