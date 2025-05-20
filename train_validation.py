import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Load datasets
train = pd.read_csv("laptops_train.csv")
test = pd.read_csv("laptops_test.csv")

# Combine for unified preprocessing
data = pd.concat([train, test], ignore_index=True)

# Clean column names
data.columns = data.columns.str.strip()
print("Cleaned columns:\n", data.columns.tolist())

# Clean RAM column (remove 'GB' and convert to int)
data['RAM'] = data['RAM'].str.replace('GB', '', regex=False).astype(int)

# Clean Weight column (remove 'kg' or 'kgs' and convert to float)
data['Weight'] = data['Weight'].str.replace('kgs', '', regex=False).str.replace('kg', '', regex=False)
data['Weight'] = pd.to_numeric(data['Weight'], errors='coerce')
data = data.dropna(subset=['Weight'])

# Clean Screen Size column (remove double quotes and convert to float)
data['Screen Size'] = data['Screen Size'].astype(str).str.replace('"', '').astype(float)

# Parse HDD and SSD from Storage column
if 'Storage' in data.columns:
    def parse_storage(storage_str):
        hdd = 0
        ssd = 0
        matches = re.findall(r'(\d+)(TB|GB)\s*(SSD|HDD)', str(storage_str))
        for size, unit, drive_type in matches:
            size_gb = int(size) * (1000 if unit == 'TB' else 1)
            if drive_type == 'HDD':
                hdd += size_gb
            else:
                ssd += size_gb
        return pd.Series([hdd, ssd])

    data[['HDD', 'SSD']] = data['Storage'].apply(parse_storage)
else:
    raise ValueError("'Storage' column not found in dataset.")

# Create PriceCategory based on Price values
price_bins = [0, 5000000, 10000000, np.inf]
price_labels = ['Low', 'Medium', 'High']
data['PriceCategory'] = pd.cut(data['Price'], bins=price_bins, labels=price_labels)

# Drop irrelevant columns
drop_cols = ['Model Name', 'Screen', 'Storage', 'Operating System Version', 'Price']
data.drop(columns=[col for col in drop_cols if col in data.columns], inplace=True)

# Identify categorical columns to encode
categorical_cols = ['Manufacturer', 'Category', 'CPU', 'GPU', 'Operating System']

# Create dictionary to hold label encoders for features
feature_encoders = {}

# Encode categorical features with separate LabelEncoders and save them
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    feature_encoders[col] = le

# Save feature encoders dict
joblib.dump(feature_encoders, 'feature_encoder.pkl')

# Encode target variable separately
target_encoder = LabelEncoder()
data['PriceCategory'] = target_encoder.fit_transform(data['PriceCategory'])

# Save target encoder
joblib.dump(target_encoder, 'target_encoder.pkl')

# Split back into train/test sets based on original sizes
train_data = data.iloc[:len(train)]
test_data = data.iloc[len(train):]

X_train = train_data.drop(columns=['PriceCategory'])
y_train = train_data['PriceCategory']
X_test = test_data.drop(columns=['PriceCategory'])
y_test = test_data['PriceCategory']

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, 'scaler.pkl')

# Define models to train
models = {
    "ID3_entropy": DecisionTreeClassifier(criterion="entropy", random_state=42),
    "CART_gini": DecisionTreeClassifier(criterion="gini", random_state=42),
    "C4.5_pruned_entropy": DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive_Bayes": GaussianNB(),
    "Decision_Tree_default": DecisionTreeClassifier(random_state=42)
}

# Train, evaluate, and save models
results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, f"{name}.pkl")

    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
    rec = recall_score(y_test, preds, average='weighted', zero_division=0)
    f1 = f1_score(y_test, preds, average='weighted', zero_division=0)

    results.append((name, acc, prec, rec, f1))

# Show evaluation results sorted by F1 score
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
results_df = results_df.sort_values(by='F1 Score', ascending=False)

print("\nModel Evaluation Results:\n")
print(results_df)