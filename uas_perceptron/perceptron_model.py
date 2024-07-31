import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Data Sintetis yang lebih kompleks: [Suhu, Kelembapan, Kecepatan Angin]
X = np.array([
    [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2], 
    [0, 2, 0], [0, 2, 1], [0, 2, 2], [1, 0, 0], [1, 0, 1], [1, 0, 2],
    [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 2, 0], [1, 2, 1], [1, 2, 2], 
    [2, 0, 0], [2, 0, 1], [2, 0, 2], [2, 1, 0], [2, 1, 1], [2, 1, 2], 
    [2, 2, 0], [2, 2, 1], [2, 2, 2]
])
# Labels: 0 untuk cerah, 1 untuk berawan, 2 untuk hujan kecil, 3 untuk hujan petir
y = np.array([
    0, 1, 2, 0, 1, 2, 2, 3, 3, 0, 1, 2,
    1, 2, 3, 2, 3, 3, 0, 1, 2, 1, 2, 3,
    2, 3, 3
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standarisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Melatih model perceptron
clf = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
clf.fit(X_train, y_train)

# Menyimpan model dan scaler ke file
with open('perceptron_model.pkl', 'wb') as model_file:
    pickle.dump((clf, scaler), model_file)

print("Model saved successfully")
