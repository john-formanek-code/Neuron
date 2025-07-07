import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=1000):
        self.weights = np.zeros(input_size)  # Inicializace vah
        self.bias = 0  # Inicializace prahu
        self.learning_rate = learning_rate  # Rychlost učení
        self.epochs = epochs  # Počet epoch

    def activation_function(self, x):
        # Heavisideova funkce
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        # Výpočet výstupu perceptronu
        total_input = np.dot(inputs, self.weights) + self.bias
        return self.activation_function(total_input)

    def fit(self, X, y):
        previous_error = float('inf')  # Uložení předchozí chyby pro kontrolu konvergence
        for epoch in range(self.epochs):
            total_error = 0  # Celková chyba v aktuální epoše
            for inputs, target in zip(X, y):
                prediction = self.predict(inputs)  # Predikce
                error = target - prediction  # Výpočet chyby
                self.weights += self.learning_rate * error * inputs  # Aktualizace vah
                self.bias += self.learning_rate * error  # Aktualizace prahu
                total_error += abs(error)  # Přidání chyby do celkové chyby

            # Logování průběhu trénování (bod 7)
            print(f"Epocha {epoch + 1}, Celková chyba: {total_error}")

            # Kontrola konvergence (bod 8)
            if total_error >= previous_error:
                print(f"Konvergence dosažena v epoše {epoch + 1}")
                break
            previous_error = total_error  # Aktualizace předchozí chyby

            # Pokud je chyba nulová, ukončit trénování
            if total_error == 0:
                print(f"Trénování dokončeno v epoše {epoch + 1}")
                break

    def plot_decision_boundary(self, X, y):
        # Vizualizace rozhodovací hranice
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))
        Z = np.array([self.predict(np.array([x, y])) for x, y in zip(xx.ravel(), yy.ravel())])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
        plt.title("Rozhodovací hranice perceptronu")
        plt.show()

# Trénovací data (AND)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Vytvoření a trénování perceptronu
perceptron = Perceptron(input_size=2)
perceptron.fit(X, y)

# Vizualizace rozhodovací hranice
perceptron.plot_decision_boundary(X, y)

# Testování perceptronu
print("Výstup po trénování:")
for inputs in X:
    print(f"Vstup: {inputs}, Predikce: {perceptron.predict(inputs)}")