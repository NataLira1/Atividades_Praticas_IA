import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#Carregamento e exploração dos dados

df_diabetes = pd.read_csv("diabetes.csv")

#Primeiras linhas do banco de dados para visualização das variáveis de cada coluna
print(df_diabetes.head())
print("\n\n")

#Análise exploratória dos dados
print("### ANÁLISE EXPLORATÓRIA DOS DADOS ###\n")
print(df_diabetes.describe())

print("\n\n")


print("### VERIFICANDO VALORES AUSENTES ###\n")
for column in df_diabetes.columns:
    if df_diabetes[column].isnull().any():
        print(f"{column} possui valores ausentes\n")
    else:
        print(f"{column} não possui valores ausentes\n")


#Verificando a presença de outliers

print("### VERIFICANDO A PRESENÇA DE OUTLIERS ###\n")
outliers_dict = {}

for column in df_diabetes.columns:
    #Verificando se a coluna é do tipo numérica        
    if pd.api.types.is_numeric_dtype(df_diabetes[column]):
        
        #Utilizando o IQR para ide  ntificar outliers
        primeiro_quartil = df_diabetes[column].quantile(0.25)
        terceiro_quartil = df_diabetes[column].quantile(0.75)
        
        IQR = terceiro_quartil - primeiro_quartil
        
        outliers = df_diabetes[
            (df_diabetes[column] < primeiro_quartil - 1.5 * IQR) |
            (df_diabetes[column] > terceiro_quartil + 1.5 * IQR)]
        
        outliers_dict[column] = outliers
    else:
        print(f"{column} é uma variável não numérica e portanto não possui outliers\n")
        
for column, outliers in outliers_dict.items():
    if outliers.empty:
        print(f"{column} não possui outliers\n")
        print()
    else:
        print(f"Outliers na coluna '{column}':")
        print(outliers)
        print("\n")

# Questão Teórica 1
"""
Aprendizado Não Supervisionado: É uma abordagem de machine learning onde o algoritmo é alimentado com dados não rotulados. 
O objetivo é descobrir padrões ou estruturas ocultas nesses dados sem qualquer orientação prévia sobre os resultados esperados. 
Um exemplo clássico de aprendizado não supervisionado é o algoritmo K-Means, que agrupa dados em clusters baseados em similaridade.

Aprendizado Supervisionado: Ao contrário, é uma abordagem onde o algoritmo é treinado com dados rotulados, ou seja, cada entrada vem com uma saída desejada. 
O modelo aprende a mapear entradas para saídas corretas, de modo a fazer previsões precisas para novos dados não vistos. 
Exemplos incluem regressão linear e árvores de decisão
"""

# Definindo as variáveis independentes (X) e dependente (y)
X = df_diabetes.drop('Outcome', axis=1)  # Todas as colunas menos a variável alvo
y = df_diabetes['Outcome']  # Variável alvo

# Dividindo os dados em treinamento (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Exibindo o tamanho dos conjuntos
print(f'Tamanho do conjunto de treinamento: {X_train.shape[0]} registros')
print(f'Tamanho do conjunto de teste: {X_test.shape[0]} registros')

# Questão Teórica 2

"""
Importância da Divisão de Dados Explicação: Dividir os dados em treinamento e teste é essencial para avaliar o desempenho do modelo. 
Sem essa divisão, o modelo pode "memorizar" os dados de treinamento, resultando em overfitting, e seu desempenho seria irrealisticamente alto. 
O uso de dados de teste oferece uma avaliação mais justa de como o modelo generaliza para dados não vistos.
"""

#Construção do Modelo

# Criando o modelo de Regressão Logística
model = LogisticRegression(max_iter=200)

# Treinando o modelo com o conjunto de treinamento
model.fit(X_train, y_train)

# Questão Teórica 3
"""
A Regressão Logística é um método eficiente e simples para problemas de classificação binária. 
Ela estima a probabilidade de uma amostra pertencer a uma classe com base em uma função logística. 
Funcionamento: O algoritmo modela a relação entre variáveis independentes e a variável dependente (classe), 
utilizando uma função sigmoide para prever probabilidades entre 0 e 1.
"""

# Avaliando o modelo
accuracy = model.score(X_test, y_test)
print(f'Acurácia do modelo: {accuracy:.2f}')

# Fazendo previsões com o conjunto de teste
y_pred = model.predict(X_test)

# Gerando a matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Exibindo a matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')
plt.title('Matriz de Confusão')
plt.show()

# Questão Teórica 4
"""
A matriz de confusão mostra a relação entre as previsões do modelo e os rótulos verdadeiros. 
Ela organiza as previsões em quatro categorias: Verdadeiro Positivo (TP), Verdadeiro Negativo (TN), Falso Positivo (FP) e Falso Negativo (FN). 
Isso ajuda a entender melhor onde o modelo está errando, além de fornecer métricas como precisão, recall e F1-score.
"""
