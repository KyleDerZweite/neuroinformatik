# Netzwerk und Rechenschritte im aktuellen Code

## Ziel

Dieses Dokument beschreibt nicht irgendeine allgemeine Lehrbuchvariante, sondern genau das,
was der aktuelle Code in `src/neuralnet.py` und `src/simulation.py` implementiert.

## Was der aktuelle Code wirklich macht

Der aktuelle Stand ist:

- voll verbundenes Feedforward-Netz,
- fuer jede trainierbare Schicht ein `Layer`,
- in jeder Schicht `sigmoid` als Aktivierung,
- pro Sample MSE-Verlust:

$$
L = \frac{1}{2}\sum_k(\hat{y}_k - y_k)^2
$$

- nach jedem einzelnen Trainingsbeispiel sofort ein Parameter-Update,
- keine Vektorisierung, sondern bewusst ausgeschriebene Schleifen.

Das ist didaktisch sinnvoll, weil jeder Rechenschritt sichtbar bleibt.

## Architektur im aktuellen Repository

### XOR

Die XOR-Simulation verwendet:

- `input = 2`
- `hidden = 2`
- `output = 1`

Also:

$$
[2, 2, 1]
$$

### Sinus-Approximation

Die Sinus-Simulation verwendet aktuell:

- `input = 1`
- `hidden = 2`
- `output = 1`

Also:

$$
[1, 2, 1]
$$

Wichtig:

- Auch bei der Sinus-Simulation wird aktuell `sigmoid` in allen Schichten benutzt.
- Deshalb werden sowohl Input als auch Zielwerte auf den Bereich `[0, 1]` skaliert.
- Der Code benutzt hier gerade **nicht** `tanh` und **nicht** einen linearen Output.

## Warum Gewichts-Matrix plus Bias-Vektor?

Fuer ein Dense-Layer mit `input_size = n` und `output_size = m` gilt fuer jedes Ausgabeneuron `j`:

$$
z_j = \sum_{i=1}^{n} x_i w_{ij} + b_j
$$

Das bedeutet:

- jedes Ausgabeneuron braucht einen eigenen Gewichtsvektor,
- mehrere Ausgabeneuronen brauchen deshalb mehrere Gewichtsvektoren,
- diese werden zu einer Matrix `W` zusammengefasst,
- dazu kommt ein Bias pro Ausgabeneuron.

Kompakt:

$$
z = x^T W + b
$$

Im aktuellen Code gilt:

- `weights` hat Form `(input_size, output_size)`,
- `biases` hat Form `(output_size,)`.

## Layer-Zaehlung

Bei `2-2-1` gibt es:

- 3 Neuronen-Layer: Input, Hidden, Output
- 2 trainierbare Layer: `2 -> 2` und `2 -> 1`

Die trainierbaren Parameter sind:

1. erstes Layer:

$$
W^{(1)} \in \mathbb{R}^{2 \times 2}, \quad b^{(1)} \in \mathbb{R}^{2}
$$

2. zweites Layer:

$$
W^{(2)} \in \mathbb{R}^{2 \times 1}, \quad b^{(2)} \in \mathbb{R}^{1}
$$

## Forward-Pass in genau der Form des Codes

Fuer ein einzelnes Layer berechnet der Code:

1. weighted sum plus bias:

$$
z_j = b_j + \sum_i x_i w_{ij}
$$

2. Aktivierung:

$$
a_j = \sigma(z_j) = \frac{1}{1 + e^{-z_j}}
$$

Im Code werden dabei drei Dinge gespeichert:

- `last_input`
- `last_pre_activation_values`
- `last_output`

Diese Werte werden spaeter fuer den Backward-Pass gebraucht.

## XOR-Forward-Pass Schritt fuer Schritt

Fuer XOR ist der Input:

$$
x = \begin{bmatrix}x_1 \\ x_2\end{bmatrix}, \quad x_1, x_2 \in \{0,1\}
$$

und die Zielabbildung:

$$
(0,0) \to 0,\; (0,1) \to 1,\; (1,0) \to 1,\; (1,1) \to 0
$$

Nehmen wir ein allgemeines `2-2-1` Netzwerk.

### Erstes Layer

Aus dem Input `x` wird:

$$
z^{(1)} = x^T W^{(1)} + b^{(1)}
$$

danach:

$$
a^{(1)} = \sigma(z^{(1)})
$$

Mit Komponenten:

$$
z^{(1)}_1 = x_1 w^{(1)}_{11} + x_2 w^{(1)}_{21} + b^{(1)}_1
$$

$$
z^{(1)}_2 = x_1 w^{(1)}_{12} + x_2 w^{(1)}_{22} + b^{(1)}_2
$$

und dann:

$$
a^{(1)}_1 = \sigma(z^{(1)}_1), \quad a^{(1)}_2 = \sigma(z^{(1)}_2)
$$

### Zweites Layer

Das Hidden-Layer-Output wird zum Input des naechsten Layers:

$$
z^{(2)} = a^{(1)} W^{(2)} + b^{(2)}
$$

$$
\hat{y} = a^{(2)} = \sigma(z^{(2)})
$$

Da hier nur ein Output-Neuron existiert, ist `\hat{y}` ein einzelner Wert.

## Verlustfunktion im aktuellen Code

Der Code berechnet pro Sample:

$$
L = \frac{1}{2}(\hat{y} - y)^2
$$

Bei mehreren Outputs waere es die Summe ueber alle Output-Komponenten.

Der Faktor `1/2` ist nur praktisch fuer die Ableitung:

$$
\frac{\partial L}{\partial \hat{y}} = \hat{y} - y
$$

Genau damit startet der Backward-Pass im Netzwerk.

## Trainingsschleife im aktuellen Code

Pro Epoche passiert fuer jedes Trainingssample genau dies:

1. `forward(sample_input)`
2. `calculate_sample_loss(sample_target_output)`
3. `backward(sample_target_output)`

Das bedeutet:

- Vorhersage berechnen
- Fehler berechnen
- sofort Gewichte und Biases anpassen

Das ist hier eine einfache Form von stochastischem bzw. sample-weisem Gradient Descent.

## Sinus-Approximation im aktuellen Code

Der Sinus-Teil in `src/simulation.py` macht aktuell Folgendes:

- Rohinput `x` laeuft von `0.0` bis `7.0`
- der Input wird auf `[0,1]` normiert
- `sin(x)` wird ebenfalls auf `[0,1]` normiert
- das Netz hat Architektur `[1, 2, 1]`
- auch hier wird `sigmoid` in Hidden- und Output-Layer verwendet

Darum sieht die Sinus-Dokumentation hier anders aus als in manchen Lehrbuchbeispielen fuer Regression:

- kein `1-8-1`,
- kein `tanh`,
- kein linearer Output.

## Warum das fuer das Lernen gut ist

Der aktuelle Code ist klein genug, dass du fuer ein einzelnes Sample alles nachvollziehen kannst:

- welche Eingaben in ein Layer gehen,
- wie `z` entsteht,
- wie `a` entsteht,
- wie aus dem Fehler ein Gradient wird,
- wie sich einzelne Gewichte veraendern.

Fuer die Rueckwaertsrechnung gibt es eine eigene Schritt-fuer-Schritt-Erklaerung in
`docs/research/06-backward-pass-visualisierung.md`.
