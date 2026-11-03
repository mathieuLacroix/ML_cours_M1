# PyTorch : autoencodeurs variationnels

Ce chapitre présente les notions manquantes pour l'implémentation d'un autoencoder variationnel pour générer des images.


```{important}
Il n'est pas possible de générer des images nettes avec un autoencodeur variationnel simple.
En effet, pour générer des images nettes, il faudrait complexifier l'architecture (ajouter d'autres couches, appliquer un autoencodeur variationnel hiérarchique, augmenter la taille de l'espace latent, etc), et surtout, il faudrait bien plus de puissance de calcul et de temps pour l'entraînement ![^ex:temps_entrainement] 

[^ex:temps_entrainement]: Le réseau VAE hiérachique proposé dans @vahdat2020nvae génère de très belles images mais il faut près de 4 jours d'entraînement sur le dataset Celeba d'où est tiré le dataset que l'on utilisera pour générer des images.
```

Comme la génération porte sur des images, nous utiliserons des réseaux convolutifs.
L'encodeur sera un réseau convolutif diminuant progressivement la hauteur et la largeur du tenseur tout en augmentant sa taille. Pour cela, on pourra utiliser la convolution présentée [ici](#cnn:couches-de-convolution). Par contre, contrairement à une tâche de classification, l'objectif est de revenir au même format pour le tenseur de sortie que celui d'entrée. On utilisera alors la **_convolution transposée_** ou **_déconvolution_**. De plus, les valeurs dans le tenseur de sortie doivent être comprises dans l'intervalle [0,1][^niveau_canal]. On utilisera alors la fonction d'activation sigmoïde.

[^niveau_canal]: Pour chaque canal, la valeur indique le niveau de couleur de chaque pixel.

## Fonction d'activation sigmoïde

La fonction sigmoïde (ou fonction logistique) est une fonction d'activation en deep learning lorsque l'on souhaite une sortie dans l'intervale $[0,1]$. Cette fonction est appliquée à chaque composante d'un vecteur, comme la fonction ReLU. Elle est définie de la manière suivante :
$$
x \mapsto \frac{1}{1 + e^{-x}} 
$$



La [](#fonction_logistique)[^src_image_sigmoide] donne le graphe de la fonction sigmoïde.

```{figure} img/logistiquef1.png
:label: fonction_logistique
```

En PyTorch, pour ajouter une fonction d'activation sigmoïde dans un réseau de neurones, on utilise la méthode `Sigmoid` de la classe `torch.nn`: 
```{code} python
nn.Sigmoid()
```

[^src_image_sigmoide]: L'image est issue du site [bibmath.net](https://www.bibmath.net/dico/index.php?action=affiche&quoi=./s/sigmoide.html).


## Couche de convolution transposée

```{error}
TO DO
```
