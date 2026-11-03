# PyTorch: réseaux convolutifs


La librairie PyTorch permet de créer facilement des réseaux de neurones convolutifs.
Par rapport à un perceptron multicouches, cela revient à ajouter des couches de convolution et des filtres max-pooling.

## Couche de convolution

Pour créer une couche de convolution, on utilise en PyTorch la classe `Conv2d` héritant de la classe `torch.nn.Module`. 
Son constructeur prend notamment en paramètre (dans l'ordre) : 
- nombre de canaux en entrée,
- nombre de canaux en sortie,
- taille du noyau,
- stride (valeur 1 par défaut),
- padding (valeur 0 par défaut).

Le code 
```{code} python
m = nn.Conv2d(8, 16, 3, stride=2, padding=1)
```
crée une couche de convolution prenant en entrée un tenseur de profondeur 8, retournant un tenseur de profondeur 16. Le noyau utilisé pour la convolution est de taille $3 \times 3$.
La convolution utilise un stride de 2 et les hauteur et largeur des tenseurs d'entrée et sortie sont identiques avec le padding de 1.

## Max-pooling

Pour créer un filtre de max-pooling, on utilise en PyTorch la classe `MaxPool2d` héritant de la classe `torch.nn.Module`. Son constructeur prend en paramètre : 
- taille du champ réceptif,
- stride (par défaut égal à la taille du champ réeptif).

Le code 
```{code} python
m = nn.MaxPool2d(2)
```
crée un filtre de max-pooling avec un champ réceptif de taille $2 \times 2$ et un stride de 2.


Il est aussi possible d'appliquer un max-pooling en lui donnant uniquement la hauteur et la largeur du tenseur de sortie. PyTorch détermine alors lui-même la taille du champ réceptif et du padding. Pour cela, on utilise la classe `AdaptiveMaxPool2d`. L'exemple crée un max-pooling retournant un tenseur de hauteur et largeur égales à 2 :
```{code} python
m = nn.AdaptiveMaxPool2d((2,2)) # ou m = nn.AdaptiveMaxPool2d(2) puisque la hauteur et la largeur sont égales
```

## Bloc de convolution, ReLU et max-pooling

Il est possible de créer en PyTorch les blocs combinant : 
- une convolution, 
- un ReLU, 
- un max-pooling.

Si l'on fixe la taille des noyaux à 3 pour la convolution et 2 pour le max-pooling, alors on le définit de la manière suivante : 
```{code} python
def bloc(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), 
        nn.ReLU(),
        nn.MaxPool2d(2),
    )
```

On peut ensuite créer un réseau de neurones convolutif en utilisant ce bloc.
À titre d'exemple le code suivant crée un réseau de convolution partant d'une image $64 \times 64$ avec 3 canaux et retournant une valeur correspondant à une combinaison linéaire
des valeurs du tenseur de sortie après 3 passages de blocs.

```{code} python
class CNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.neuralNetwork = nn.Sequential(
            bloc(3, 8),     # Taille du tenseur de sortie : (8, 32, 32)
            bloc(8, 16),    # Taille du tenseur de sortie : (16, 16, 16)
            bloc(16, 32),   # Taille du tenseur de sortie : (32, 8, 8)
            nn.Flatten(),   # Taille du vecteur de sortie 32 * 8 * 8 = 2048
            nn.Linear(2048,1)
        )
    def forward(self, x):
        return self.neuralNetwork(x)

```

