## **Gradient Descent (Optimisation IA)**

**Théorie** : Algorithme qui **descend la pente** d'une fonction pour trouver le minimum (ex: coût d'un modèle ML). Base de TOUS les apprentissages ! 

## Code ultra-simple (6 lignes)

```python
import numpy as np
import matplotlib.pyplot as plt

# Fonction coût : y = x² (minimum à x=0)
def cost(x): return x**2

# Gradient descent
x = 10  # point de départ
learning_rate = 0.1
history = [x]

for i in range(50):
    grad = 2*x  # dérivée
    x -= learning_rate * grad  # descente
    history.append(x)

print("✅ Minimum trouvé:", x)  # ≈ 0.0
```

## Visualisation bluffante

```python
x_range = np.linspace(-11, 11, 100)
plt.plot(x_range, cost(x_range), 'b-', label='Coût y=x²')
plt.plot(history, cost(np.array(history)), 'ro-', label='Descente')
plt.legend()
plt.show()
```

**Résultat** : **La courbe montre le chemin** du point qui glisse vers le minimum !

## Démo avec tes données Word2Vec

```python
# Optimiser hyperparamètres (ex: learning rate)
costs = []
for lr in [0.01, 0.1, 0.5]:
    x = 10
    for i in range(20): x -= lr * 2*x
    costs.append(cost(x))
print("Meilleur learning rate:", np.argmin(costs))
```

**Parfait** : montre **comment TOUTES les IA apprennent** mathématiquement ! 

**Installation** : rien (numpy/matplotlib déjà installés)


```python

```
