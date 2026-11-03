# Réseaux de neurones convolutifs


```{error}
Notation pytorch (Channels, Height, Width) ou (Batch, Channels, Height, Width)
```


Jusqu'à présent, nous n'avons vu qu'une seule architecture de réseaux de neurones : le perceptron multicouches. Pour la reconnaissance de chiffres pour le dataset MINST, c'est largement suffisant. Cependant, cette architecture a plusieurs inconvénients pour des problèmes de classification d'images plus complexes, notamment sur des images en couleur. Par exemple, il est difficile d'atteindre un bon niveau de classification pour [CIFAR-10](https://cave.cs.toronto.edu/kriz/cifar.html) ou [MINST-1D](https://github.com/greydanus/mnist1d) par exemple.

Un premier problème du perceptron multicouches vient du nombre de paramètres. Pour des images de 224 * 224 pixels en couleurs (3 valeurs par pixel), cela représente un vecteur de taille 150528 en entrée. Comme la première couche est généralement de taille supérieure ou égale à l'entrée (surtout si le nombre de couches cachées est faible), le nombre de paramètres de la première couche est supérieur ou égal à 22 658 829 312 !

Un deuxième problème vient du fait que le perceptron multicouches est sensible à la translation. Comme on a pu le voir en TP, les performances du modèle chutent pour MNIST lorsque les chiffres sont décalés.

Un moyen de résoudre ces deux problèmes est d'utiliser comme architecture des réseaux de neurones convolutifs, c'est-à-dire un réseau de neurones contenant des couches spécifiques appelées **_couches de convolutions_**.


## Images et tenseurs 3D


Un perceptron multicouches prend en entrée un vecteur donc on aplatit une image (flatten) pour former un vecteur. L'entrée d'un réseau de neurones convolutif est un **_tenseur 3D_**, c'est-à-dire un tableau à trois dimensions (profondeur, hauteur et largeur). Les tailles des dimensions d'un tenseur $T$ sont respectivement notées $d^T_p$, $d^T_h$ et $d^T_l$. 


La hauteur et la largeur d'un tenseur représentant une image sont la hauteur et la largeur de l'image en pixels. La profondeur est constituée d'un certain nombre de **_canaux_** pour représenter un pixel. Pour une image en couleur, on a 3 canaux correspondant respectivement aux niveaux de Rouge, Vert et Bleu du pixel. Pour une image en noir et blanc, on a un seul canal correspondant au niveau de gris du pixel.



(cnn:couches-de-convolution)=
## Couches de convolution

Une couche de convolution prend en entrée un tenseur 3D et retourne un autre tenseur 3D dont la hauteur, la largeur et la profondeur peuvent être différente de celles du tenseur en entrée de la couche.

Le principe de la convolution est le suivant. La valeur à la position $(p,h,l)$[^posTen3D] dans le tenseur de sortie est obtenue en appliquant une combinaison linéaire des valeurs de tous les canaux de la couche d'entrée correspondant au "pixel" à la position $(h,l)$ et aux pixels "voisins". On applique ensuite sur chaque valeur du tenseur de sortie une fonction d'activation.

La convolution est similaire à une application linéaire d'une couche dense d'un perceptron multicouches qui ne s'applique que localement, c'est-à-dire que toutes les entrées ne sont pas utilisées pour calculer la sortie. La convolution est suivie de l'application d'une fonction d'activation (ReLU dans ce cours) pour chaque valeur du tenseur de sortie.

[^posTen3D]: la valeur qui se trouve à la position $l$ pour la largeur, $h$ pour la hauteur et $p$ pour la profondeur.


Pour présenter de manière formelle une convolution, nous commencerons à présenter celle-ci lorsque l'entrée et la sortie sont tenseurs 3D contenant une seul canal, c'est-à-dire que la profondeur est égale à 1.
Nous généraliserons ensuite à n'importe quel nombre de canaux.

```{note}
La convolution est une opération classique utilisée dans le traitement d'images pour flouter, faire ressortir les contours, etc. Voir cette [vidéo](https://www.youtube.com/watch?v=KuXjwB4LzSA) sur la convolution pour plus de détails[^tempsConvolutionImage].

[^tempsConvolutionImage]: La partie de la vidéo sur la convolution pour le traitement d'images est à partir de 8mn33 (jusqu'à 10mn30) mais vous pouvez regarder toute la vidéo pour mieux comprendre la convolution.
```

### Convolution lorsque l'entrée et la sortie ont un seul canal

Comme les tenseurs d'entrée $T^e$ et de sortie $T^s$ ont un seul canal, chaque valeur est uniquement représentée par sa position $(h,l)$ (hauteur et hauteur).


Pour effectuer une convolution, on considère une matrice $W$ de taille $k \times k$ et un biais $b$. Cette matrice, appelée **_noyau_**, et le biais $b$ sont les paramètres de la couche de convolution.

La valeur du tenseur de sortie à la position $(w,h)$ est donnée par : 
```{math}
:label: formula:conv1channel
T^s(h,l) = \sum_{i = 0}^{k-1} \sum_{j = 0}^{k-1} W(i, j) \times T^e(h+i, l+j) + b
```
La partie du tenseur d'entrée utilisée pour calculer la valeur $T^s(l,h)$ est appelée **_champ réceptif_**. Le champ réceptif est décalé pour calculer chaque valeur partie du tenseur de sortie.

La [](#fig:ex_convolution)[^src_fig_ex_conv] donne un exemple de calcul d'une valeur du tenseur de sortie. 

[^src_fig_ex_conv]: La figure est vient de ce [site Web](https://iq.opengenus.org/downsampling-and-upsampling-in-cnn/#gsc.tab=0).

```{figure} img/cnn/cnn_exemple_convolution.png
:label: fig:ex_convolution
Exemple de calcul d'une valeur de tenseur de sortie par convolution (le noyau est en violet).
```




#### Taille du tenseur de sortie 

D'après la formule [](#formula:conv1channel), le tenseur de sortie est de taille $(d^{T^e}_h - k + 1) \times (d^{T^e}_l - k + 1)$ car les valeurs $T^s(h,l)$ ne sont pas définies pour les $(k-1)$ dernières lignes et colonnes du tenseur d'entrée.
La [](#fig:conv)[^source_img_conv] illustre ceci en indiquant quelles valeurs de sortie sont calculées par convolution.

```{figure} img/cnn/no_padding_no_strides.gif
:label: fig:conv
Exemple de convolution $3\times 3$. La sortie est de taille $(3-1) \times (3-1)$.
```

[^source_img_conv]: Cette image et les suivantes viennent de ce [site](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) associé à la publication @dumoulin2016guide.


#### Padding

Si l'on souhaite augmenter la taille du tenseur de sortie, on peut ajouter du **_padding_**, c'est-à-dire entourer le tenseur d'entrée avec $n$ colonnes à gauche et à droite, et $n$ lignes en haut et en bas, comme sur la [](#fig:conv_padding). 
On peut ajouter jusqu'à $k-1$ lignes et colonnes autour du tenseur d'entrée (si l'on ajoute plus, on se retrouve également avec des lignes et colonnes avec des valeur nulles en sortie). La [](#fig:conv_padding) et la [](#fig:conv_padding_full) donnent deux exemples de padding différents.

```{figure} img/cnn/padding_no_strides.gif
:label: fig:conv_padding
Exemple de convolution $3\times 3$ avec un padding de 1. Les cases en pointillées correspondent au padding, c'est-à-dire à des cases avec la valeur 0. La sortie a la même taille que l'entrée.
```
```{figure} img/cnn/full_padding_no_strides.gif
:label: fig:conv_padding_full
Exemple de convolution avec $3\times 3$ un padding de taille 2. La sortie est de taille plus grande que l'entrée.
```

````{note} 
Si le noyau est de taille impair et que l'on considère un padding de taille $\frac{k-1}{2}$, alors le tenseur de sortie est de même taille que le tenseur d'entrée (*c.f.* [](#fig:conv_padding)).

````

#### Stride

Dans les convolutions que l'on a vues, le champ réceptif est décalé de 1 vers la droite[^changementligne] à chaque fois pour calculer la valeur suivante dans le tenseur de sortie. Il est possible de compresser encore plus le tenseur de sortie en décalant le champ réceptif de plus d'une case vers la droite à chaque fois, comme dans la [](#fig:conv_padding_stride).  

[^changementligne]: C'est vrai sauf pour les changements de ligne. Dans ce cas, on décale de 1 vers le bas et on revient complètement à gauche.

```{figure} img/cnn/padding_strides.gif
:label: fig:conv_padding_stride
Exemple de convolution avec padding et strides. On décale la convolution de deux à chaque fois, donc la sortie a une taille correspondant plus ou moins à la moitié de l'entrée (cela dépend du padding).
```




### Généralisation pour n'importe quel tenseur 3D en entrée et sortie

Si le tenseur d'entrée à plusieurs canaux, alors l'étape de convolution correspond à une application linéaire de tous les canaux du champ réceptif. Le calcul d'une valeur pour le champ de sortie devient : 
```{math}
:label: formula:convPlrschannels
T^s(h,l) = \sum_{ p = 0}^{d^{T_e}_p - 1}\sum_{i = 0}^{k-1} \sum_{j = 0}^{k-1} W(p, i, j) \times T^e(p, h+i, l+j) + b
``` 

Le noyau est alors un tenseur avec une profondeur égale à celle du tenseur d'entrée, voir la [](#fig:convMultiChannel)[^fig:src_conv_multi_channels] pour un exemple.

[^fig:src_conv_multi_channels]: Cette image est issue du livre @prince2023understanding.

```{figure} img/cnn/ConvImage.svg
:label: fig:convMultiChannel
Exemple de convolution avec un tenseur d'entrée à 3 canaux.
```

Si le tenseur de sortie a plusieurs canaux, alors une convolution différente est appliquée sur chaque canal. On a donc autant de noyaux et de biais que l'on a de canaux de sortie.


### Nombre de paramètres

Le nombre de paramètres d'une convolution est égal à :
$$d^{T^s}_p \times (d^{T^e}_p \times k \times k + 1)$$

Si le tenseur d'entrée est de taille $224 \times 224 \times 3$ (3 canaux) et que le tenseur de sortie a la même dimension, en prenant un noyau de taille $15 \times 15$, le nombre de paramètres est $3 \times (3 \times 15 \times 15 + 1) = 2028$, à comparer avec le nombre de paramètres de la première couche du perceptron multicouches donné en exemple au début du chapitre !

## Filtre max-pooling

Pour réduire la taille d'un tenseur, on peut également appliquer un filtre sans paramètre appelé **_max-pooling_**. Cette opération s'applique sur chaque canal d'entrée indépendamment des autres. On obtient alors un tenseur avec la même profondeur que celui en entrée.

Pour chaque canal, le max-pooling consiste à considérer comme pour la convolution un champ réceptif de taille $k$, où $k$ est un hyper-paramètre du filtre max-pooling, et à appliquer un stride de taille $k$. Au lieu de calculer une somme pondérée, on conserve uniquement la valeur maximum du champ réceptif. Un exemple est donné dans la [](#fig:max_pooling)[^fig:src_max_pooling].

```{figure} img/cnn/max_pooling.webp
:label: fig:max_pooling
Max-pooling avec un filtre $2 \times 2$.
```

[^fig:src_max_pooling]: Figure provenant de cette [page web](https://medium.com/@ilyurek/what-is-max-pooling-and-why-do-we-need-max-pooling-57247a3fbca9).  

```{note}
On peut appliquer un stride différent de la taille du filtre de max-pooling pour avoir du chevauchement.
```


Le filtre de max-pooling est utilisé pour :
- réduire la taille du tenseur,
- rendre le modèle robuste au bruit en ne gardant que les valeurs les plus déterminantes.

Contrairement à une convolution (avec le même stride), le max-pooling n'a pas de paramètres, cela simplifie l'apprentissage.

## Réseau de neurones convolutif

Un réseau de neurones convolutif est un réseau contenant plusieurs couches de convolution. L'entrée du réseau est un tenseur 3D et les représentations intérmédiaires (après chaque couche) des premières couches sont également des tenseurs 3D. À un moment, un tenseur peut être aplati pour entrer par exemple dans un perceptron multicouches. Ce sera notamment le cas lorsque l'on utilisera un réseau de neurones convolutif pour la classification d'images. Dans ce cas, on terminera généralement avec un tenseur de hauteur et largeur égales à 1 avant aplatissement.


Dans un réseau de neurones convolutif, on enchaîne généralement des blocs constitués de : 
- une convolution, 
- une activation ReLU,
- un max pooling.

En général, la convolution change peu ou pas la taille de l'entrée en ajoutant du padding et le max pooling diminue la largeur et la hauteur du tenseur. Par contre, surtout quand l'entrée est une image (donc avec une profondeur d'au plus 3), la profondeur augmente après chaque bloc.
Le choix du nombre de blocs, de la taille des noyaux et du nombre de canaux sont des hyperparamètres et dépendent de la tâche d'apprentissage et de la puissance de calcul. 




```{important}
Pour bien comprendre les réseaux de neurones convolutifs, vous pouvez utiliser le formidable outil interactif de @wangCNNExplainerLearning2020. Il se trouve [ici](https://poloclub.github.io/cnn-explainer/). Il montre un réseau de neurones convolutif déjà entraîné et permet de comprendre toute la passe forward (la prédiction) pour un certain nombre d'images. C'est très visuel et permet de bien comprendre chaque étape : convolution, max pooling, flatten, scores, softmax et prédiction de la classe ! À voir absolument !
```
