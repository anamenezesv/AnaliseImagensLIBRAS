import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Carregar os dados
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Dividir os dados em conjuntos de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Inicializar o modelo
model = RandomForestClassifier()

# Treinar o modelo
model.fit(x_train, y_train)

# Fazer previsões
y_train_predict = model.predict(x_train)
y_test_predict = model.predict(x_test)

# Calcular acurácia
train_score = accuracy_score(y_train_predict, y_train)
test_score = accuracy_score(y_test_predict, y_test)

print(f'Train Accuracy: {train_score * 100:.2f}%')
print(f'Test Accuracy: {test_score * 100:.2f}%')

# Salvar o modelo
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
