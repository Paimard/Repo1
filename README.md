# Implementierung des Algorithmus Schritt für Schritt
## 1.
Fügen Sie eine Spalte mit Einsen für den Bias-Term hinzu. Ich habe 1 gewählt, weil sich dieser Wert nicht ändert, wenn Sie einen mit einem beliebigen Wert multiplizieren.

[source,xml]
----
import pandas as pd
import numpy as np
df = pd.read_csv('ex1data2.txt', header = None)
df.head()
----
