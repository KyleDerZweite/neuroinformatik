# Backward Pass Schritt fuer Schritt

## Ziel

Dieses Dokument erklaert die Rueckwaertspropagation exakt so, wie sie im aktuellen
Code in `src/neuralnet.py` umgesetzt ist.

## Die Idee in einem Satz

Der Forward-Pass berechnet eine Vorhersage.
Der Backward-Pass berechnet, wie stark jedes Gewicht und jeder Bias zum Fehler beigetragen hat.

## Die wichtigsten Variablen im Code

Pro `Layer` werden nach `forward(...)` diese Werte gespeichert:

- `last_input`
- `last_pre_activation_values`
- `last_output`

Das ist noetig, weil der Backward-Pass genau diese Werte fuer die Ableitungen braucht.

## Gesamtbild fuer ein `2-2-1` Netzwerk

```text
Forward:

x = [x1, x2]
  |
  v
[ Layer 1: 2 -> 2 ]
  z^(1) = x * W^(1) + b^(1)
  a^(1) = sigmoid(z^(1))
  |
  v
[ Layer 2: 2 -> 1 ]
  z^(2) = a^(1) * W^(2) + b^(2)
  y_hat = sigmoid(z^(2))
  |
  v
Loss L = 1/2 * (y_hat - y)^2
```

```text
Backward:

Loss
  |
  v
dL/dy_hat = y_hat - y
  |
  v
Output-Layer:
  local gradient = (gradient from next layer) * sigmoid'(output)
  |
  v
gradient fuer Hidden-Layer berechnen
  |
  v
Hidden-Layer:
  local gradient = (gradient from next layer) * sigmoid'(output)
  |
  v
Gewichte und Biases updaten
```

## Schritt 1: Gradient am Netzwerk-Output

In `NeuralNetwork.backward(target_output)` startet der Code mit:

$$
\frac{\partial L}{\partial \hat{y}} = \hat{y} - y
$$

Bei mehreren Outputs passiert das fuer jede Output-Komponente getrennt.

Im Code heisst dieser Vektor:

- `current_gradient_vector`

Wichtig:

- Das ist noch **nicht** direkt der Gewichtsgradient.
- Das ist zuerst der Fehler bezogen auf den Output des letzten Layers.

## Schritt 2: Lokaler Gradient eines Layers

Ein Layer bekommt von rechts einen Gradient-Vektor:

$$
g_j
$$

Im Output-Layer ist das zunaechst:

$$
g_j = \frac{\partial L}{\partial a_j}
$$

Da die Aktivierung `sigmoid` ist, gilt:

$$
\sigma'(z_j) = a_j(1-a_j)
$$

Der Code nutzt die bereits gespeicherte Aktivierung `a_j = last_output[j]` und berechnet:

$$
\delta_j = g_j \cdot a_j(1-a_j)
$$

Im Code heisst dieser Vektor:

- `local_gradient_per_output`

Anschaulich:

- `g_j` sagt: "Wie empfindlich ist der Fehler gegenueber dem Layer-Output?"
- `a_j(1-a_j)` sagt: "Wie stark aendert sich der Layer-Output, wenn sich die weighted sum aendert?"
- das Produkt gibt: "Wie empfindlich ist der Fehler gegenueber der weighted sum?"

## Mini-Visualisierung fuer ein einzelnes Ausgabeneuron

```text
input_i --( * w_ij )--> contribution ----+
                                         |
other inputs ----------------------------+--> z_j --> sigmoid --> a_j --> Loss
                                         |
bias b_j --------------------------------+
```

Rueckwaerts:

```text
Loss --> dL/da_j --> dL/dz_j = dL/da_j * sigmoid'(z_j)
                     |
                     +--> dL/db_j = dL/dz_j
                     |
                     +--> dL/dw_ij = input_i * dL/dz_j
                     |
                     +--> Beitrag zum vorherigen Layer
```

## Schritt 3: Gradient fuer das vorherige Layer berechnen

Damit ein frueheres Layer lernen kann, muss der Fehler weiter nach links propagiert werden.

Fuer jedes Input-Neuron `i` des aktuellen Layers berechnet der Code:

$$
gradient\_for\_previous\_layer_i = \sum_j \delta_j w_{ij}
$$

Das bedeutet:

- jedes Input-Neuron des aktuellen Layers beeinflusst mehrere Ausgabeneuronen,
- deshalb werden die Beitraege aller Ausgabeneuronen aufsummiert.

Im Code heisst das Ergebnis:

- `gradient_for_previous_layer`

Wichtig:

- Dieser Schritt passiert **vor** dem Update der Gewichte.
- Der Gradient wird also mit den aktuell noch gueltigen Gewichten propagiert.

## Schritt 4: Bias-Gradient und Bias-Update

Fuer den Bias eines Ausgabeneurons gilt:

$$
z_j = \sum_i x_i w_{ij} + b_j
$$

Darum ist:

$$
\frac{\partial z_j}{\partial b_j} = 1
$$

Also:

$$
\frac{\partial L}{\partial b_j} = \delta_j
$$

Update-Regel:

$$
b_j \leftarrow b_j - \eta \delta_j
$$

mit Lernrate `\eta`.

## Schritt 5: Gewichtsgradient und Gewichtsupdate

Fuer ein Gewicht `w_ij` gilt:

$$
z_j = \sum_i x_i w_{ij} + b_j
$$

Darum ist:

$$
\frac{\partial z_j}{\partial w_{ij}} = x_i
$$

und damit:

$$
\frac{\partial L}{\partial w_{ij}} = x_i \delta_j
$$

Genau das rechnet der Code als:

$$
weight\_gradient = input\_value \cdot local\_gradient\_per\_output[j]
$$

Update-Regel:

$$
w_{ij} \leftarrow w_{ij} - \eta \frac{\partial L}{\partial w_{ij}}
$$

also:

$$
w_{ij} \leftarrow w_{ij} - \eta x_i \delta_j
$$

## Schritt 6: Was in einem Layer komplett passiert

In exakt derselben Reihenfolge wie im Code:

1. Gradient von rechts empfangen
2. mit `sigmoid`-Ableitung zum lokalen Gradient machen
3. Gradient fuer das vorherige Layer berechnen
4. Biases updaten
5. Gewichte updaten
6. Gradient fuer das vorherige Layer zurueckgeben

## Schritt 7: Auf das ganze Netzwerk angewandt

Bei einem `2-2-1` Netz laeuft die Rueckwaertsrechnung so:

```text
1. Forward-Pass fuer ein Sample
   x -> hidden activations -> output prediction

2. Loss berechnen
   L = 1/2 * (y_hat - y)^2

3. Output-Layer backward
   start gradient = y_hat - y
   -> lokalen Gradient berechnen
   -> W^(2) und b^(2) updaten
   -> Gradient fuer Hidden-Layer erzeugen

4. Hidden-Layer backward
   empfangenen Gradient nehmen
   -> lokalen Gradient berechnen
   -> W^(1) und b^(1) updaten
   -> Gradient fuer Input-Seite erzeugen
```

Danach ist das Sample komplett verarbeitet.

## Warum der Code `last_output` statt `last_pre_activation_values` fuer sigmoid benutzt

Fuer `sigmoid` gilt:

$$
\sigma'(z) = \sigma(z)(1-\sigma(z))
$$

Wenn `a = sigma(z)` bereits bekannt ist, dann reicht:

$$
\sigma'(z) = a(1-a)
$$

Darum braucht der Code fuer die Aktivierungsableitung nicht nochmal explizit `z`.

`last_pre_activation_values` wird trotzdem gespeichert, weil es fuer Debugging und spaetere Erweiterungen hilfreich ist.

## Was wirklich rueckwaerts fliesst

Rueckwaerts fliessen nicht die originalen Inputs und auch nicht die Targets.
Rueckwaerts fliessen Gradienten, also Sensitivitaeten des Fehlers:

- "Wenn sich dieser Wert leicht aendert, wie stark aendert sich dann der Gesamtfehler?"

Das ist die Kernidee von Backpropagation.

## Hauefige Denkfehler

- Es wird nicht der Fehlerwert selbst rueckwaerts kopiert, sondern seine Ableitungsinformation.
- Der Gewichtsgradient ist nicht einfach `prediction - target`.
- Ohne gespeicherte Zwischenwerte aus dem Forward-Pass kann der Layer seinen lokalen Gradient nicht korrekt berechnen.
- Ein Layer muss den Gradient erst fuer das vorherige Layer berechnen und danach die eigenen Parameter updaten.
