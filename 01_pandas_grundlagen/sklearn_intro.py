import pandas as pd
# Neu: Wir importieren die Funktion für den Train-Test-Split aus Scikit-learn
from sklearn.model_selection import train_test_split

# --- Daten laden (wie zuvor) ---
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv(url, names=column_names)

# --- Daten vorbereiten für das maschinelle Lernen ---

# 1. Trennung von Features (Eingabemerkmale) und Target (Zielvariable)
# X enthält alle Spalten AUSSER der 'class'-Spalte. Das sind unsere Merkmale.
X = df.drop('class', axis=1)
# y enthält NUR die 'class'-Spalte. Das ist das, was wir vorhersagen wollen.
y = df['class']

# 2. Aufteilen der Daten in Trainings- und Test-Sets
# Wir nutzen die importierte Funktion.
# test_size=0.2 bedeutet, dass 20% der Daten für das Test-Set reserviert werden.
# random_state=42 sorgt dafür, dass die Aufteilung reproduzierbar ist.
# Jedes Mal, wenn du den Code mit diesem Wert ausführst, ist die Aufteilung exakt dieselbe.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Ergebnisse überprüfen ---
print("Shape des gesamten Datensatzes (X):", X.shape)
print("Shape des gesamten Targets (y):", y.shape)
print("-" * 30) # Trennlinie
print("Shape des Trainings-Sets (X_train):", X_train.shape)
print("Shape des Test-Sets (X_test):", X_test.shape)
print("-" * 30) # Trennlinie
print("Shape des Trainings-Targets (y_train):", y_train.shape)
print("Shape des Test-Targets (y_test):", y_test.shape)

# --- Modelltraining ---

# 1. Das Modell importieren
# Wir importieren das Modell der Logistischen Regression aus Scikit-learn.
from sklearn.linear_model import LogisticRegression

# 2. Eine Instanz des Modells erstellen
# Wir erstellen ein Objekt des Modells. Hier könnte man auch Parameter setzen,
# aber für den Anfang nutzen wir die Standardeinstellungen.
model = LogisticRegression()

# 3. Das Modell trainieren
# Das ist der magische Moment. Die .fit()-Methode nimmt die Trainingsdaten (X_train)
# und die zugehörigen richtigen Antworten (y_train) und lernt die Muster.
model.fit(X_train, y_train)

# --- Vorhersagen und erste Bewertung ---

# 4. Vorhersagen auf den Testdaten machen
# Wir verwenden unser trainiertes Modell, um Vorhersagen für das Test-Set zu machen,
# das es noch nie zuvor gesehen hat.
predictions = model.predict(X_test)

# 5. Die Vorhersagen ausgeben
print("\nVorhersagen des Modells auf den Testdaten:")
print(predictions)

# 6. Die tatsächlichen Werte zum Vergleich ausgeben
# Wir konvertieren y_test in ein NumPy-Array, um eine schönere Ausgabe zu bekommen.
print("\nTatsächliche Werte der Testdaten:")
print(y_test.to_numpy())