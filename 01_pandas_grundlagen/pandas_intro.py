# 1. Pandas importieren und ihm einen Spitznamen geben
# Das ist eine universelle Konvention. Jeder importiert pandas als 'pd'.
import pandas as pd

# 2. Die URL zu einem sauberen, einfachen Datensatz
# Der Iris-Datensatz ist ein Klassiker im Machine Learning.
# Er enthält Messungen von drei verschiedenen Schwertlilien-Arten.
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# 3. Spaltennamen definieren, da die Datei selbst keine Überschriften enthält
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# 4. Die CSV-Datei von der URL in einen Pandas DataFrame laden
# pd.read_csv() ist die wichtigste Funktion zum Einlesen von Daten.
# Der Parameter 'names' weist unsere Spaltennamen zu.
df = pd.read_csv(url, names=column_names)

# 5. Die ersten 5 Zeilen des DataFrames ausgeben, um zu prüfen, ob alles geklappt hat
# Die .head()-Methode ist dein bester Freund für einen ersten Blick auf die Daten.
print("Die ersten 5 Zeilen des Iris-Datensatzes:")
print(df.head())

# 6. Die Dimensionen (Shape) des DataFrames ausgeben
# .shape ist keine Funktion, sondern ein Attribut (daher ohne Klammern).
# Es gibt (Anzahl Zeilen, Anzahl Spalten) zurück.
print("\nShape des DataFrames (Zeilen, Spalten):")
print(df.shape)