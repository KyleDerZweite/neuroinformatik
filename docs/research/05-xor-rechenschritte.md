# XOR Schritt fuer Schritt (mit LaTeX und Beispielwerten)

## Ziel

Du willst verstehen, warum ein Layer nicht nur aus zwei Vektoren bestehen kann, wenn mehrere Ausgaenge berechnet werden sollen.

Beim XOR ist der Input ein Vektor mit zwei Werten:

$$
x = \begin{bmatrix}x_1 \\ x_2\end{bmatrix}, \quad x_1, x_2 \in \{0,1\}
$$

und die Zielwerte sind:

$$
(0,0) \to 0,\; (0,1) \to 1,\; (1,0) \to 1,\; (1,1) \to 0
$$

## Warum Gewichts-Matrix + Bias-Vektor?

Fuer ein Dense-Layer mit `input_size = n` und `output_size = m` gilt fuer jedes Ausgabeneuron $j$:

$$
z_j = \sum_{i=1}^{n} x_i w_{ij} + b_j
$$

- Jedes Ausgabeneuron hat einen eigenen Gewichtsvektor der Laenge $n$.
- Bei $m$ Ausgabeneuronen hast du also $m$ solcher Vektoren.
- Diese $m$ Vektoren stapelst du zu einer Matrix $W \in \mathbb{R}^{n \times m}$.
- Dazu kommt der Bias-Vektor $b \in \mathbb{R}^{m}$.

Kompakt:

$$
z = x^T W + b
$$

Nur ein einzelner Gewichtsvektor reicht nur fuer ein einziges Ausgabeneuron.

## Layer-Zaehlung kurz geklaert

Es gibt zwei gaengige Arten zu zaehlen:

- Neuronen-Layer: Input, Hidden, Output
- Trainierbare Layer: nur Uebergaenge mit Gewichten

Beispiel `2-2-1`:

- Neuronen-Layer: 3 (Input, Hidden, Output)
- Trainierbare Layer: 2

Die beiden trainierbaren Layer sind:

1. von 2 nach 2: $W^{(1)} \in \mathbb{R}^{2 \times 2}$, $b^{(1)} \in \mathbb{R}^{2}$
2. von 2 nach 1: $W^{(2)} \in \mathbb{R}^{2 \times 1}$, $b^{(2)} \in \mathbb{R}^{1}$

## XOR-Netz: 2-2-1 Architektur

Wir nehmen ein kleines Netz mit:

- 2 Inputs
- 2 Hidden-Neuronen
- 1 Output-Neuron
- Sigmoid-Aktivierung

$$
\sigma(t) = \frac{1}{1 + e^{-t}}
$$

Beispielparameter (funktionieren fuer XOR sehr gut):

$$
W^{(1)} = \begin{bmatrix}
20 & 20 \\
20 & 20
\end{bmatrix},
\quad
b^{(1)} = \begin{bmatrix}-10 & -30\end{bmatrix}
$$

$$
W^{(2)} = \begin{bmatrix}20 \\ -20\end{bmatrix},
\quad
b^{(2)} = -10
$$

Forward-Pass:

$$
z^{(1)} = x^T W^{(1)} + b^{(1)}, \quad a^{(1)} = \sigma(z^{(1)})
$$

$$
z^{(2)} = a^{(1)} W^{(2)} + b^{(2)}, \quad \hat{y} = \sigma(z^{(2)})
$$

## Vollstaendige Rechnung fuer Input x = (1,1)

### 1) Hidden-Layer

Mit $x = [1,1]^T$:

$$
z^{(1)}_1 = 1 \cdot 20 + 1 \cdot 20 - 10 = 30
$$

$$
a^{(1)}_1 = \sigma(30) \approx 0.9999999999999065
$$

$$
z^{(1)}_2 = 1 \cdot 20 + 1 \cdot 20 - 30 = 10
$$

$$
a^{(1)}_2 = \sigma(10) \approx 0.9999546021312976
$$

### 2) Output-Layer

$$
z^{(2)} = 20 \cdot a^{(1)}_1 - 20 \cdot a^{(1)}_2 - 10
$$

$$
z^{(2)} \approx 20 \cdot 0.9999999999999065 - 20 \cdot 0.9999546021312976 - 10
$$

$$
z^{(2)} \approx -9.999092042628
$$

$$
\hat{y} = \sigma(z^{(2)}) \approx \sigma(-9.999092042628) \approx 0.0000454
$$

Interpretation:

- Ausgabe ist sehr nahe bei 0
- Das passt zu XOR fuer $(1,1) \to 0$

## Kurzer Check fuer alle 4 XOR-Inputs

Mit denselben Gewichten/Biases ergibt sich ungefaehr:

| Input | Erwartet | Vorhersage $\hat{y}$ |
|---|---:|---:|
| (0,0) | 0 | 0.000045 |
| (0,1) | 1 | 0.999955 |
| (1,0) | 1 | 0.999955 |
| (1,1) | 0 | 0.000045 |

## Verbindung zu deinem Code

In deinem `Layer` gilt konzeptionell:

- `weights` hat Form `(input_size, output_size)`
- `biases` hat Form `(output_size,)`

Bei XOR im ersten Layer also z.B.:

- `input_size = 2`
- `output_size = 2`
- `weights` ist eine 2x2 Matrix
- `biases` ist ein 2er Vektor

Genau deshalb ist die Kombination aus Gewichts-Matrix und Bias-Vektor fuer einen allgemeinen Layer sinnvoll und notwendig.

## Zweites Beispiel: Sinus-Approximation mit 1-8-1

Fuer eine Regression wie $y = \sin(x)$ ist eine typische Architektur:

- 1 Input
- 8 Hidden-Neuronen
- 1 Output

Also: `1-8-1`.

### Parameterformen

Auch hier gibt es genau 2 trainierbare Layer:

1. Hidden-Layer:

$$
W^{(1)} \in \mathbb{R}^{1 \times 8}, \quad b^{(1)} \in \mathbb{R}^{8}
$$

2. Output-Layer:

$$
W^{(2)} \in \mathbb{R}^{8 \times 1}, \quad b^{(2)} \in \mathbb{R}^{1}
$$

Das sieht oft "vektorartig" aus, ist mathematisch aber weiterhin Matrixform:

$$
W^{(1)} = \begin{bmatrix}w_1 & w_2 & \dots & w_8\end{bmatrix}
$$

$$
W^{(2)} = \begin{bmatrix}v_1 \\ v_2 \\ \vdots \\ v_8\end{bmatrix}
$$

### Forward-Pass fuer 1-8-1

Wir behandeln den Skalar-Input als 1D-Vektor $x \in \mathbb{R}^{1}$:

$$
z^{(1)} = x W^{(1)} + b^{(1)} \in \mathbb{R}^{8}
$$

$$
a^{(1)} = \tanh(z^{(1)}) \in \mathbb{R}^{8}
$$

$$
z^{(2)} = a^{(1)} W^{(2)} + b^{(2)} \in \mathbb{R}^{1}
$$

$$
\hat{y} = z^{(2)}
$$

Hinweis: Bei Regression wird am Output haeufig eine lineare Aktivierung verwendet (also keine Sigmoid am Ende).

### Warum das wichtig ist

- Bei `1-8-1` sind es trotzdem zwei Gewichtsmatrizen.
- Nur bei `1-1` waere die Gewichtsmatrix effektiv `1x1` (also ein einzelner Skalar).
- Sobald mehrere Hidden-Neuronen beteiligt sind (hier 8), brauchst du mehrere Gewichte pro Uebergang.
