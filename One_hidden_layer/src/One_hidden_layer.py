import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score


# Veri Setini Yükleme ve Hazırlama
def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path, sep=None, engine='python', on_bad_lines='skip')
        df = df.dropna()
        print(f"Veri başarıyla yüklendi. Satır sayısı: {len(df)}")
    except Exception as e:
        print(f"Dosya okuma hatası: {e}")
        return None, None, None, None

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)

    # %80 Eğitim, %20 Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizasyon
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train.T, X_test.T, y_train.T, y_test.T


# 2. Aktivasyon Fonksiyonu
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 3. Parametre Başlatma
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


# 4. İleri Yayılım
def forward_propagation(X, parameters):
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


# 5. Maliyet Hesaplama
def compute_cost(A2, Y):
    m = Y.shape[1]
    logprobs = np.multiply(np.log(A2 + 1e-15), Y) + np.multiply(np.log(1 - A2 + 1e-15), 1 - Y)
    cost = -np.sum(logprobs) / m
    return float(np.squeeze(cost))


# 6. Geri Yayılım
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W2 = parameters["W2"]
    A1, A2 = cache["A1"], cache["A2"]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


# 7. Parametre Güncelleme
def update_parameters(parameters, grads, learning_rate=0.5):
    parameters["W1"] -= learning_rate * grads["dW1"]
    parameters["b1"] -= learning_rate * grads["db1"]
    parameters["W2"] -= learning_rate * grads["dW2"]
    parameters["b2"] -= learning_rate * grads["db2"]
    return parameters


# 8. Ana Model Fonksiyonu
def nn_model(X, Y, n_h, num_iterations=1000):
    n_x, n_y = X.shape[0], Y.shape[0]
    parameters = initialize_parameters(n_x, n_h, n_y)
    costs = []

    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)

        if i % 10 == 0:
            costs.append(cost)

    return parameters, costs


# 9. Tahmin
def predict(parameters, X):
    A2, _ = forward_propagation(X, parameters)
    return (A2 > 0.5).astype(int)


file_name = "C:\\Users\\S.EREN\\PycharmProjects\\PythonProject\\YZM304-Derin-Ogrenme\\Data\\BankNote_Authentication.csv"
X_train, X_test, y_train, y_test = load_and_preprocess_data(file_name)

if X_train is not None:
    hidden_nodes = [3, 5, 8]
    iterations = [500, 1000]

    last_costs = []

    print("\n--- Model Performans Testleri ---")
    for n_h in hidden_nodes:
        for n_step in iterations:
            params, costs = nn_model(X_train, y_train, n_h=n_h, num_iterations=n_step)
            preds = predict(params, X_test)

            acc = accuracy_score(y_test.flatten(), preds.flatten())
            f1 = f1_score(y_test.flatten(), preds.flatten())
            print(f"n_h: {n_h} | Iter: {n_step} >> Accuracy: {acc:.4f}, F1: {f1:.4f}")
            last_costs = costs

    # Öğrenme Eğrisi Grafiği
    plt.figure(figsize=(10, 6))
    plt.plot(last_costs)
    plt.title(f"Model Öğrenme Eğrisi (n_h={n_h}, iter={n_step})")
    plt.xlabel("İterasyon (x10)")
    plt.ylabel("Maliyet (Cost)")
    plt.grid(True)
    plt.show()

    # Detaylı Rapor
    print("\nSon Modelin Detaylı Sınıflandırma Raporu:")
    print(classification_report(y_test.flatten(), preds.flatten()))