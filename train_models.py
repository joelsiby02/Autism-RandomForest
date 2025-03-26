import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

def train_and_save_model(csv_path, model_save_path, metrics_dir):
    # Load the dataset
    dataset = pd.read_csv(csv_path)
    
    # Preprocess categorical variables
    dataset['Sex'] = dataset['Sex'].map({'M': 1, 'F': 0})
    dataset['Jaundice'] = dataset['Jaundice'].map({'Yes': 1, 'No': 0})
    dataset['Family_mem_with_ASD'] = dataset['Family_mem_with_ASD'].map({'Yes': 1, 'No': 0})
    dataset['ASD_traits'] = dataset['ASD_traits'].map({'Yes': 1, 'No': 0})
    
    # Feature selection
    X = dataset[['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10_Autism_Spectrum_Quotient', 
                 'Sex', 'Jaundice', 'Family_mem_with_ASD']]
    y = dataset['ASD_traits']
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train RandomForest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability estimates for ROC curve
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Save model and scaler
    joblib.dump(model, model_save_path)
    joblib.dump(scaler, model_save_path.replace('.pkl', '_scaler.pkl'))

    # Ensure metrics directory exists
    os.makedirs(metrics_dir, exist_ok=True)

    # Save classification report
    report_path = os.path.join(metrics_dir, f"{os.path.basename(model_save_path).replace('.pkl', '_report.txt')}")
    with open(report_path, "w") as f:
        f.write(class_report)

    # Plot and save confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No ASD', 'ASD'], yticklabels=['No ASD', 'ASD'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {os.path.basename(csv_path).split('_')[0]}")
    plt.savefig(os.path.join(metrics_dir, f"{os.path.basename(model_save_path).replace('.pkl', '_conf_matrix.png')}") )
    plt.close()
    
    # Plot and save ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {os.path.basename(csv_path).split('_')[0]}')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(metrics_dir, f"{os.path.basename(model_save_path).replace('.pkl', '_roc_curve.png')}") )
    plt.close()

    print(f"Model trained and metrics saved for: {os.path.basename(csv_path)}")
    print(f"Accuracy: {accuracy:.4f}")

# Define paths for CSV files and model saving
data_dir = 'data'
model_dir = 'models'
metrics_dir = 'metrics'

# Train models for different age groups and save metrics
train_and_save_model(os.path.join(data_dir, 'Children_ASD.csv'), os.path.join(model_dir, 'children_asd_model.pkl'), metrics_dir)
train_and_save_model(os.path.join(data_dir, 'Adolescent_ASD.csv'), os.path.join(model_dir, 'adolescent_asd_model.pkl'), metrics_dir)
train_and_save_model(os.path.join(data_dir, 'Adult_ASD.csv'), os.path.join(model_dir, 'adult_asd_model.pkl'), metrics_dir)
train_and_save_model(os.path.join(data_dir, 'Young_ASD.csv'), os.path.join(model_dir, 'young_asd_model.pkl'), metrics_dir)