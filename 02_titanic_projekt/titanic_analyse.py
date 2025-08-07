import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Lade den Datensatz
df = sns.load_dataset('titanic')

#print(df.shape) # Shape (891, 15) --> 891 Datensätze, 15 Spalten
#print(df.head(100)) # survived  pclass     sex   age  ...  deck  embark_town  alive  alone
#print(df.info())
print(df.describe())

df['age'] = df['age'].fillna(df['age'].median())

replace_sex = {'male': 0, 'female': 1}
df['sex'] = df['sex'].map(replace_sex)

X = df[['pclass', 'age', 'sex']]
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Eine Instanz des Scalers erstellen
scaler = StandardScaler()

# 2. Die Skalierung auf den Trainingsdaten lernen UND sie direkt anwenden
# .fit_transform() ist eine Abkürzung für .fit() gefolgt von .transform()
X_train_scaled = scaler.fit_transform(X_train)

# 3. Die auf den Trainingsdaten gelernte Skalierung auf die Testdaten anwenden
# Wichtig: Hier nur .transform() verwenden, nicht .fit_transform()!
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(solver='liblinear')
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print(accuracy_score(y_test, y_pred))
print(y_pred.mean())




