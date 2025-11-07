# train_model.py
# Author: Prajwal Mavkar
# Project: Digit Recognition App
# Description: Train a Gradient Boosting model on MNIST digits dataset and save it as model.pkl

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

print("📥 Loading digits dataset...")
X, y = load_digits(return_X_y=True)

print("✂️ Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("🧠 Training Gradient Boosting model...")
model = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
)
model.fit(X_train, y_train)

print("📊 Evaluating model...")
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

print("💾 Saving model as model.pkl...")
joblib.dump(model, "model.pkl")

print("🎉 Training complete! Model saved successfully.")
