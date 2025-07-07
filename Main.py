import numpy as np #vektory a matice

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=1000):
        #inicializace parametrů
        self.weights = np.zeros(input_size) #inicializuje váhy perceptoru na nulový vektor o délce input_size
        self.bias = 0 #inicializuje práh na 0
        self.learning_rate = learning_rate #rychlost učení
        self.epochs = epochs #počet iterací
    
    def activation_function(self, x):
        #heavision funkce
        return 1 if x >=0 else 0
        #Vrátí 1, pokud je vstup x větší nebo roven 0, jinak vrátí 0.

    #vypočítá výstup perceptronu pro dané vstupy inputs
    def predict(self, inputs):
        #výpočet výstupu perceptronu
        total_input=np.dot(inputs, self.weights) + self.bias #vypočítá skalární součin vstupů a vah+bias
        return self.activation_function(total_input) #Výsledek se předá aktivační funkci, která vrátí konečný výstup (0 nebo 1)
    
    def fit(self, X, y):
        previous_error = float("inf")
        #učení preceptronu pomocí metoda gradientního sestupu
        for epoch in range(self.epochs):
            for inputs, target in zip(X, y):
                #vypočítání chyby
                prediction = self.predict(inputs)
                error = target - prediction
                #úprava váh a práhu
                self.weights += self.learning_rate * error * inputs #Upraví váhy perceptronu pomocí gradientního sestupu
                self.bias += self.learning_rate * error

#přílad trénovacích dat (problém AND)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

#vytvoření a trénování perceptronu
perceptron = Perceptron(input_size=2)
perceptron.fit(X, y)

#testování perceptronu
print("Výstup po trénování:")
for inputs in X:
    print(f"Vystup: {inputs}, Predikce: {perceptron.predict(inputs)}")
