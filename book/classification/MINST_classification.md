# PyTorch : classification MINST 

Cette tâche d'apprentissage automatique consiste à déterminer, pour une image correspondant à un chiffre écrit à la main, à prédire quel est ce chiffre. Par exemple, si l'image est 

```{image} img/chiffre5.png
:alt: image du chiffre 5
:class: bg-primary
:width: 200px
:align: center
```
le modèle doit prédire le chiffre 5.




Cette tâche est de l'apprentissage **supervisé**. Pour l'entraînement, on dispose d'un dataset d'entraînement de 60000 images. Pour chaque image, on a le chiffre correspondant sur cette image (son label). 

Étant donné une image en entrée, le modèle doit prédire un chiffre entre 0 et 9. 

`````{important}
Le modèle prédit pour chaque chiffre la probabilité que l'image en entrée corresponde à ce chiffre. Le modèle prédit donc un vecteur $v$ de taille 10, tel que :
- chaque coordonnée est positive ou nulle,
- la somme est égale à 1.

On considère que le modèle prédit le chiffre dont la valeur associée dans le vecteur est maximum, c'est-à-dire qu'il prédit $\argmax_{i \in \{0,\dots,9\}} v_i$.

```{figure} img/mlp_image5_histogramme.png
Exemple de prédiction. Le modèle prend en entrée une image 28 * 28 pixels, et prédit la probabilité que cette image corresponde à chaque chiffre. Comme la probabilité de prédire 5 est la plus grande, on considère que le modèle prédit pour cette image le chiffre 5.
```
`````


Pour l'évaluation du modèle, on dispose d'un dataset de test de 10000 images avec leur label.

## Image

Chaque image fait 28 * 28 pixels. Chaque pixel correspond à un entier entre 0 et 255, correspondant au niveau de blanc (0: le pixel est noir, 255: le pixel est blanc). On transforme cette image en un vecteur de taille 784 (28 * 28 = 784), chaque case contenant le niveau de gris d'un pixel (les pixels sont stackés ligne par ligne en partant du haut).

Le modèle prend donc en entrée un vecteur de taille 784 et prédit un vecteur de taille 10.



## Tenseurs

En apprentissage automatique, la structure de données la plus utilisée est le *tenseur*. Celui-ci correspond à un tableau multi-dimensionnel. Les tenseurs sont une généralisation des matrices puisque ces dernières correspondent à des tenseurs de dimension 2. Un tenseur de dimension 1 correspond à un vecteur et un tenseur de dimension 0 à un scalaire. 


### Création de tenseurs

On peut créer des tenseurs avec les fonctions `zeros`, `ones` et `rand` du module `torch`. On donne en paramètre une séquence d'entiers. Le nombre de valeurs correspond au nombre de dimensions et la valeur de chaque entier correspond à la taille du tenseur dans cette dimension. 
Les valeurs du tenseur sont des zéros, des uns, ou des nombres flottants aléatoires selon la fonction utilisée.



```{code} python
import torch

# t1 est un tenseur à deux dimensions.
#  Il "correspond" à une matrice à 2 lignes et 3 colonnes
t1 = torch.zeros(2,3)

# t2 correspond à un vecteur de taille 5 contenant que des 1
t2 = torch.ones(5)

# t3 est un tenseur à 3 dimensions. La première dimension est de taille 2, 
# la deuxième dimension de taille 4 et la troisième de taille 3. 
# Il contient donc 2 * 4 * 3 = 24 valeurs aléatoires.
t3 = torch.rand(2,4,3)
```

Il est aussi possible de créer un tenseur à partir d'une liste (simple ou imbriquée) avec `torch.tensor`.
```{code} python
t4 = torch.tensor([1, 2, 3, 4])

t5 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
```



### Dimensions d'un tenseur

On peut récupérer la *forme* d'un tenseur, c'est-à-dire le nombre de dimensions et la taille dans chaque dimension du tenseur en utilisant la méthode `size`.

```{code} python
print(t1.size()) 
```

````{hint}
On peut utiliser la méthode `size` pour créer un tenseur de la même forme qu'un autre. Par exemple, pour créer un tenseur aléatoire de même forme (même nombre de dimensions et même taille dans chaque dimension) que `t1`, on peut faire 
```{code} python
t6 = tensor.rand(t1.size())
```
````

### Vues d'un tenseur

En mémoire, les tenseurs sont stockés sous la forme d'un tableau d'une seule dimension. Les valeurs sont stockées de sorte qu'en avançant d'une case dans le tableau, c'est l'indice de la dernière dimension qui est incrémenté en premier (et on ne passe à la dimension précédente que quand celle-ci "déborde"). C'est ce qu'on appelle l'ordre [row-major](https://en.wikipedia.org/wiki/Row-_and_column-major_order).


La méthode `view` permet de changer le nombre de dimensions et la taille de chaque dimension d'un tenseur sans modifier le stockage des valeurs. Elle crée un nouveau tenseur dont la forme est spécifiée entre parenthèses **mais sans copier les données**.

Par exemple, si l'on a un tenseur `x` de dimension `(3,4)` (trois lignes et deux colonnes), on peut obtenir un vecteur de taille 12, ou un tenseur de dimensions `(3,2,2)` sans copier ni modifier les données.

```{code} python
x = torch.rand(3,4) # crée un tenseur 3 * 4

y = x.view(12) # vue sur le tenseur

z = x.view(3,2,2) #autre vue

```

```{important}
Dans l'exemple, `x`, `y` et `z` correspondent aux mêmes valeurs. Si on change les données d'un tenseur, cela change les données des deux autres.

Par exemple, la case `x[1,2]` correspond à la case `y[6]` et à la case `z[1,1,0]`.
```





## Récupération du dataset

Le dataset MINST est accessible facilement à travers `pytorch` de la manière suivante.


```{code} python
from torchvision import datasets, transforms

train_data = datasets.MNIST(root="data", download=True, transform=transforms.ToTensor())

test_data = datasets.MNIST(root="data", download=True, transform=transforms.ToTensor(), train=False)

```

```{note} 
Avec le paramètre `transform=transforms.ToTensor()`, les images sont directement transformées en tenseurs de la forme `(1, 28, 28)` (3 dimensions : la première de taille 1 (car une seule information par pixel, le taux de blanc), les deux autres correspondent à la taille de l'image en pixels). Les valeurs sont des flottants dans l'intervalle $[0,1]$.
```

### Découpage en batchs

Pour accélérer l'apprentissage, on peut proprécesser les images par lot ou *batch*. Les calculs pour les images d'un même batch sont faits en parallèles grâce aux GPU.

Pour cela, on charge les données le `Dataloader`.

```{code} python
from torch.utils.data import DataLoader


batch_size = 256
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
```

```{note}
Le paramètre `shuffle` indique si les batchs sont créés en mélangeant ou non les images. Cela n'a pas d'intérêt pour le dataset de test mais à un intérêt pour le dataset de train car cela stabilise l'entraînement lorsque l'on utilise les méthodes de type gradient stochastiques.
```

````{important}
L'objet retourné par le `DataLoader` est un itérable sur les batchs. On peut donc parcourir les batchs avec une boucle `for` mais on ne peut pas les indexer.

```{code} python
for  (images, labels) in train_dataloader:
    # images contient les images du batch
    # labels contient les labels du batch
    pass
```
````

## Création d'un perceptron multicouches

Un réseau de neurones est codé comme une classe héritant de `torch.nn.Module`:  
* Dans le constructeur, on construit le réseau
* on implémente la méthode `forward` qui prend une entrée et retourne la prédiction.

```{note}
Pas besoin d'implémenter une méthode pour le backward, cela est fait automatiquement par `pytorch`.
```

Un perceptron multicouches est une suite de couches linéaires et de fonctions d'activation. On peut donc utiliser pour construire le perceptron :
* `nn.Sequential` : permet de créer un réseau à partir d'une séquence de couches passées en paramètre,
* `nn.Linear(in, out)` : crée une couche linéaire dont l'entrée est de dimension `in` et la sortie de dimension `out`,
* `nn.ReLU()` : crée une couche d'activation.

```{note} Remarque
Chaque couche dense est codée comme une couche linéaire suivie d'une couche d'activation.
```
Comme les images sont codées sous forme de tenseur à trois dimensions (1,28,28), on peut les "aplatir" (flatten en anglais), c'est-à-dire les transformer en un vecteur d'une seule dimension de taille $1 \times 28 \times 28 = 784$, avec la couche `nn.Flatten`.

Lorsque l'on a créé un réseau de neurones, on peut obtenir un forward en appelant le réseau avec une entrée.

Pour coder en pytorch le réseau de neurones de la [](#param_MLP_example), on peut coder la classe

```{code} python
class MLP(nn.Module):
    def __init__(self):
        super().__init__() # on appelle le constructeur de la classe nn.Module
    
        self.neuralNetwork = nn.Sequential(
            nn.Linear(4,2),
            nn.ReLU(),
            nn.Linear(2,2)
        )

    def forward(self, x):
        return self.neuralNetwork(x)
```

## Utiliser le GPU

Pour accélérer les calculs, il est intéressant (quand c'est possible), d'utiliser un processeur GPU plutôt qu'un processeur CPU. En effet, beaucoup de calculs peuvent être faits en parallèle. Par exemple, en utilisant un batch, le forward peut être fait pour chaque instance du batch en parallèle. L'architecture GPU permet justement de faire ces calculs en parallèle. La librairie `pytorch` a justement été développée pour tirer parti des proceseurs GPU[^cuda].

[^cuda]: La librairie `pytorch` utilise par défaut la bibliothèque propriétaire `cuda` qui permet de faire tourner du calcul parallèle sur des GPU de NVIDIA. Il est aussi possible d'utiliser la bibliotèque `ROCm` pour faire tourner `pytorch` sur des GPU d'AMD. On supposera ici que le processeur GPU est un processeur NVIDIA.


Pour savoir si l'on a un GPU disponible, il suffit d'écrire l'instruction : 
```{code} python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
`device` vaudra `"cuda"` si un GPU est disponible, et `"cpu"` sinon.

Pour copier un batch ou un modèle sur GPU, il suffit d'utiliser la méthode (du modèle ou du tenseur) `to(device)`.


## Évaluer un modèle

Pour évaluer un modèle, on l'utilisera pour prédir la classe pour chaque entrée du dataset de test et on comptera combien de fois le modèle s'est trompé. 

```{code} python
def evaluer(model, loader):
    """ Retourne la précision du modèle sur le dataset loader, 
    c'est-à-dire nombre de bonnes prédictions / nombre prédictions"""
    correct = 0
    total = 0
    with torch.no_grad(): # pas de rétropropagation sur ces instructions 
        for x, y in loader:
            x, y = x.to(device), y.to(device) # On envoie sur gpu
            scores = model(x) # Passe forward : on obtient les scores pour chaque chiffre
            pred = scores.argmax(dim=1) # retourne l'indice correspondant au score max, donc le chiffre prédit
                                        # Attention : on travaille par batch donc on a l'indice pour chaque élément
                                        # du batch
            correct += (pred == y).sum().item() # on compte le nombre de fois où la prédiction = label
            total += y.size(0) # taille du batch
    return correct / total
```

```{note} Remarque
L'instruction `torch.no_grad` indique à `pytorch` qu'il n'y aura pas de rétropropagation sur ces calculs et qu'il n'est donc pas nécessaire de les stocker en mémoire. Cela permet d'accélérer nettement l'évaluation du modèle. 
```

## Entraîner un modèle

Entraîner un modèle consiste à modifier itérativement les paramètres du modèle de manière à ce que la loss diminue sur le dataset d'entraînement. On utilisera pour la mise à jour des paramètres l'optimiseur ADAM avec les paramètres par défaut.

Pour la loss, on utilise la log vraisemblance négative (negative log Likelihood). Pour la classification discrète, cette loss est équivalente avec l'entropie croisée. On utilisera en `pytorch` la `CrossEntropyLoss`[^urlCELoss] comme fonction de loss. Cette loss applique elle-même le softmax aux prédictions du modèle. 

[^urlCELoss]:https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss

```{code} python
def entrainer(model, train_loader, epochs=10):
    """Boucle d'entraînement.
    Affiche la valeur de la loss d'entraînement à chaque itération.
    Affiche le temps d'entraînement à la fin.
    """
    model = model.to(device) # On met le modèle sur GPU
    opt = torch.optim.Adam(model.parameters()) # On utilise Adam pour l'optimisation
    loss_function = nn.CrossEntropyLoss(reduction="sum") # critère d'évaluation : loss utilisée
    
    t0 = time.time()
    # Pour chaque époque
    for ep in range(1, epochs + 1):
        somme, n = 0.0, 0

        # Pour chaque batch
        for x, y in train_loader:
            x, y = x.to(device), y.to(device) # copie du batch sur GPU (si dispo)
            opt.zero_grad() # on remet à 0 les gradients
            loss = loss_function(model(x), y) # on calcule la loss
            loss.backward() # on rétropropage
            opt.step() # on met à jour les paramètres de l'optimiseur
            somme += loss.item() # On calcule la loss sur le dataset entier
            n += y.size(0)
        print(f"epoque {ep:2d}/{epochs} - loss {somme/n:.4f}")

    duree = time.time() - t0
    print(f"Entraînement termine en {duree:.1f}s")
```
