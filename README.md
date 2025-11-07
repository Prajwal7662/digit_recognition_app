🔢 Digit Recognition App

This project is an interactive Streamlit web application that allows users to draw handwritten digits (0–9) on a digital canvas.
The app uses a Gradient Boosting Machine Learning model to predict which digit the user drew — in real-time.

🚀 Features

🖌️ Draw a digit on the screen using your mouse or touchpad

🤖 Model predicts which digit (0–9) was drawn

🧠 Built using Gradient Boosting Classifier (Scikit-learn)

💾 Trained model stored in a pickle file

⚡ Simple, fast, and interactive UI made with Streamlit

🧰 Technologies Used
Component	Description
Python	Core programming language
Scikit-learn	For Gradient Boosting model
Streamlit	Web app framework
streamlit-drawable-canvas	Drawing area for digits
NumPy & Pandas	Data handling
Pillow (PIL)	Image processing
📦 Installation Guide
1️⃣ Clone or download this repository
git clone https://github.com/yourusername/digit-recognition-app.git
cd digit-recognition-app

2️⃣ Install dependencies
pip install -r requirements.txt

🧠 Model Training

Use the following script to train your model and create model.pkl:

python train_model.py


This script loads the MNIST digits dataset, trains a Gradient Boosting Classifier, and saves the model.

🖥️ Run the App
streamlit run app.py


Then open your browser (usually http://localhost:8501
).

🎨 How It Works

Draw any digit (0–9) on the black canvas.

The app preprocesses your drawing (resize + normalize).

The trained Gradient Boosting model predicts the digit.

The predicted digit appears instantly on screen.

🧾 Requirements

requirements.txt

streamlit
streamlit-drawable-canvas
scikit-learn
numpy
pandas
pillow

📊 Dataset Details

Dataset: MNIST Digits Dataset (from sklearn.datasets.load_digits)

Total Samples: 1,797 images

Image Size: 8 × 8 pixels

Classes: Digits 0–9

🧩 Project Structure
├── app.py                # Streamlit web app
├── train_model.py        # Model training script
├── model.pkl             # Saved Gradient Boosting model
├── requirements.txt      # Dependencies
└── README.md             # Project documentation

🏆 Example Output

Draw “3” → Model predicts: 3

Draw “9” → Model predicts: 9

👨‍💻 Author

Developed by: Prajwal Mavkar
Project Title: Digit Recognition App
Tools Used: Python · Streamlit · Scikit-learn

💡 Future Enhancements

Extend support for handwritten A–Z alphabets

Integrate deep learning (CNN) for better accuracy

Add live camera digit recognition

Display prediction confidence scores
