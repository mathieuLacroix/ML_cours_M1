# Apprentissage par réseaux de neurones

Ce chapitre présente le principe de l'apprentissage automatique via les réseaux de neurones (*deep learning* ou *machine learning* en anglais). Il se focalise sur l'apprentissage **_supervisé_**, qui correspond au cas où l'on possède un ensemble de données pour lesquels on connaît la réponse. 

## Tâche d'apprentissage automatique

Une tâche d'apprentissage automatique consiste à approximer un *mapping* entre des entrées et les sorties correspondantes. 

La notion de tâche est large. Voici quelques exemples.
* **Classification d'images d'animaux :** les entrées sont des photographies d'animaux, la sortie associée à chaque image est le type de l'animal (chien, chat, oiseau, etc) sur cette image. 
* **Prédiction du prix d'un logement :** les entrées sont les données pour un ensemble de logements (surface, localisation, etc), la sortie associée à chaque logement est le prix au $m^2$.
* **Analyse syntaxique :** Les entrées sont un ensemble de phrases, la sortie associée est la structure grammaticale de la phrase (généralement représentée par la catégorie grammaticale de chaque mot).


## Réseaux de neurones

Étant donné une tâche, l'apprentissage automatique consiste à apprendre grâce aux données les paramètres d'un réseau de neurones qui approxime ce mapping.

```{note} Définition
Mathématiquement parlant, un réseau de neurones est une fonction mathématique
$$
f_\theta: X \rightarrow Y
$$

où $\theta$ est l'ensemble des paramètres du réseau.

```

Il existe différentes architectures de réseaux de neurones qui correspondent à différentes formes mathématiques de la fonction.

```{topic} Forme et paramètres d'une fonction
La forme du fonction est définie par la forme du graphe de cette fonction. Dans le cas d'une fonction à une variable, la forme peut être linéaire, quadratique, exponentielle, etc. 

Les paramètres correspondent aux coefficients de la fonction. Pour une fonction à une variable de forme quadratique : 
$$
a x^2 + b x + c
$$
les coefficients $a$, $b$ et $c$ sont les paramètres de la fonction.
```




```{important}
Entraîner un réseau de neurones consiste à trouver les paramètres $\theta$ tels que $f_\theta(x)$ est le plus proche de $y$ pour chaque entrée $x \in X$.  

```

## Datasets

La taille de l'ensemble $X$ peut être très grand, voire infini. Il n'est donc pas possible de considérer toutes ces entrées. On considère un sous-ensemble d'entrées $x^1, \dots, x^n$ de $X$ sur lequel on va entraîner notre modèle. Si cet ensemble de données est suffisamment représentatif de l'ensemble des entrées, alors notre modèle entraîné sur $x^1, \dots, x^n$ pourra prédire des réponses correctes ou de bonne qualité pour n'importe quelle entrée de $X$.

Par exemple, si l'on souhaite entraîner un modèle qui doit prédire si une image comporte ou non un chat, on ne peut pas l'entraîner avec l'ensemble de toutes les photos possibles. On utilisera un ensemble de photos, par exemple 100 000 photos, pour entraîner notre modèle. Une fois entraîné, il pourra prédire la réponse (y a-t-il un chat sur la photo ?) pour n'importe quelle photo.

Comme on est dans le cadre d'un apprentissage supervisé, on connaît la sortie pour chacune des entrées $x^1,\dots, x^n$. On les notes $y^1,\dots,y^n$ et on les appelle **_labels_**. L'ensemble 
$$
\mathcal{D} = \{(x^i,y^i) \mid i=1,\dots,n\}
$$
forme ce que l'on appelle le **_dataset_**.


````{note} Définition : capacité de généralisation
La capacité à prédire correctement la sortie pour des entrées qui ne sont pas dans le dataset est la **capacité de généralisation** du modèle. C'est évidemment cette capacité que l'on recherche lorsque l'on entraîne un modèle.

Pour évaluer la capacité de généralisation d'un modèle, on utilise un dataset qui contient des entrées qui n'ont pas été vues pour l'entraînement. On appelle ce dataset le **_dataset de test_**. 

```{warning}
Le dataset de test ne doit être utilisé que pour l'évaluation du modèle, une fois l'entraînement terminé.
```



Le dataset $\mathcal{D}$ est appelé **_dataset d'entraînement_**.  

````



## Fonction de loss

Pour évaluer, étant donné $(x^i,y^i) \in \mathcal{D}$,  comment $f_\theta(x^i)$ est proche du label $y^i$, on utilise une fonction de dissimilarité appelée **_fonction de loss_**. Cette fonction de loss $L$ est une fonction de signature :
$$
L : Y \times Y \rightarrow \mathbb{R}_+.
$$

Pour un couple $(x^i,y^i)$, plus la valeur de $L(f_\theta(x^i),y^i)$ est proche de zéro, plus l'erreur est faible, c'est-à-dire que la prédiction du modèle est correcte. La loss vaut 0 si la prédiction est identique à la sortie.


```{important}
Entraîner un modèle sur le dataset d'entraînement $\mathcal{D}$ consiste à trouver $\theta$ qui minimise
$$
\mathcal{L}(\theta) = \sum_{i = 1}^n L(f_\theta(x^i),y^i)
$$
```

La fonction de loss dépend de la tâche d'apprentissage. 

(entrainer-un-modele)=
## Entraîner un modèle

Pour entraîner un modèle, on utilise une méthode de type descente de gradient. Cette méthode va modifier itérativement les paramètres $\theta$ de manière à minimiser la fonction $\theta \rightarrow L(\theta)$.

```{important}
La fonction $\mathcal{L}(\theta)$ n'est pas convexe, donc le minimum obtenu par une méthode de type descente de gradient est un **minimum local**. Malheureusement, comme $\mathcal{L}$ est une fonction hautement non linéaire dépendant de beaucoup de variables, il n'est pas possible de déterminer un minimum global de manière efficace.

```

Si $\theta \in \mathbb{R}^k$, alors le gradient de la loss par rapport à $\theta$, noté $\nabla_\theta(\mathcal{L})$, est :
```{math}
\nabla_\theta(\mathcal{L}) = \begin{pmatrix}
\frac{\partial \mathcal{L}}{\partial \theta_1}\\
~\\
\vdots \\
~\\
\frac{\partial \mathcal{L}}{\partial \theta_k}\\
\end{pmatrix}
```

### Gradient

Le gradient d'une fonction en un point donne la direction dans laquelle se diriger pour augmenter la valeur de la fonction au voisinage de ce point. Dans la [](#figGradientsNonconvexFunction), les gradients sont tracés en 5 points. Les gradients aux deux minimums sont nuls.


```{figure} img/gradient_descent_nonconvex2.png
:label: figGradientsNonconvexFunction
```

```{note} Remarque
Si le gradient est nul, cela ne signifie pas obligatoirement que l'on est sur un extrêmum. On peut aussi être sur un plateau.
```

### Descente de gradients et variantes

Pour minimiser une fonction, on va donc calculer le gradient au point courant et partir dans la direction opposée. Comme le gradient ne donne qu'une information locale (ie, au voisinage du point courant), on ne fait qu'un "petit" pas dans cette direction.

````{important}
Le pas que l'on fait est donné par :
$$- \alpha \times \nabla_\theta(\mathcal{L})$$

où $\alpha \ge 0$ est un scalaire, appelé *learning rate* (ou taux d'apprentissage) et $\nabla_\theta(\mathcal{L})$ est le gradient de la loss par rapport à $\theta$.

Le learning rate est un **_hyperparamètre_**[^hyperparametre] de l'entraînement. Il est très important et difficile à calibrer. S'il est trop petit, alors la solution courante ne bouge pas beaucoup (pas trop petits) et la convergence de l'algorithme est lente. S'il  est trop grand, alors on risque de "sauter" par dessus le minimum local et osciller sur les bords, voir [](#lr_pb_grad) pour l'influence du learning rate sur la convergence.

[^hyperparametre]: Un hyperparamètre est un paramètre qui n'est pas appris durant l'entraînement mais fixé avant. Cela peut être le learning rate, le nombre de couches ou la taille des couches cachées dans un perceptron multicouches, etc.

```{figure} img/lr_pb_GradientDescent.avif
:label: lr_pb_grad
Exemple de descente de gradient avec un petit (gauche) et un grand (droite) learning rate[^srcImgIBM].
```

Le learning rate varie généralement durant la descente car il est paramétré de manière dynamique avec un scheduler. Par exemple, le learning rate est initialisé avec une grande valeur et décroit toutes les $N$ itérations ($N$ est un paramètre). Différents schedulers sont décrits [ici](https://d2l.ai/chapter_optimization/lr-scheduler.html).

````



[^srcImgIBM]: Image issue  de la page Web d'IBM "Qu'est-ce que la descente de gradient" (https://www.ibm.com/fr-fr/think/topics/gradient-descent).


La descente de gradient fonctionne de la manière suivante : 
```python
# Initialisation des theta
theta = theta0

# Tant que l'entraînement n'est pas fini
while not finished_traininig:

    # Calcul de la loss pour le dataset d'entraînement
    loss = model(theta, X, Y) 

    # Gradient de la loss par rapport à theta
    gradient = gradient(loss) 

    # Mise à jour des paramètres
    # On fait un pas dans la direction opposée au gradient
    theta = theta - alpha * gradient 

return theta
```

```{hint} Définition
Une itération de la boucle, qui consiste à calculer la loss pour tout le dataset d'entraînement s'appelle une **_époque_** (ou *epoch*).
```


La condition d'arrêt peut être lorsque le gradient de la loss est (quasi-)nul ou lorsqu'un nombre suffisant d'époques est atteint.


```{note} Gradient stochastique
Calculer la loss sur tout le dataset prend du temps lorsque ce dernier est grand. Dans ce cas, les paramètres ne sont pas mis à jour fréquemment, ce qui rend la méthode de résolution peu efficace. 

Pour accélérer la convergence (et stabiliser les résultats), on utilise une *méthode de type gradient stochastique*. Le dataset est divisé en *batchs*, c'est-à-dire en sous-ensembles d'instances de taille fixe. Les batchs sont processés itérativement jusqu'à processer tout le dataset. Le gradient est calculé puis les paramètres sont mis à jour après chaque batch.

Pour plus de robustesse, les batchs sont créés de manière alétoire à chaque itération. 

Pour plus d'efficacité, les instances du batchs sont traités en parallèle si une architecture GPU est disponible.

```

Il existe différentes variantes de descente de gradient stochastique. On choisira comme optimiseur la méthode ADAM [@kingma2014adam]. Pour plus d'information sur le gradient stochastiques et les différentes variantes, se référer à [@prince2023understanding, chapitre 6].


### Obtenir le gradient de la loss par rapport aux paramètres

Afin d'appliquer une méthode de type descente de gradient, il faut pouvoir calculer le gradient de la loss par rapport aux paramètres. Pour cela, on utilise la règle de dérivation en chaîne :

$$
\frac{\partial{\mathcal{L}}}{\partial \theta} = \frac{\partial{\mathcal{L}}}{\partial s}\frac{\partial{s}}{\partial \theta}
$$
où $s$ sont les sorties prédites par le modèle. 

La loss est choisie de manière à pouvoir calculer de manière analytique $\frac{\partial{\mathcal{L}}}{\partial s}$. Par ailleurs, les réseaux de neurones sont construits de manière à pouvoir calculer automatiquement $\frac{\partial{s}}{\partial \theta}$ via la différentiation automatique, c'est-à-dire que les calculs pour obtenir la loss par rapport aux paramètres sont stockés et tous différentiables ; le calcul du gradient peut alors être fait automatiquement.

```{seealso}
Voir la Section @obtenir-le-gradient pour un exemple de gradient obtenu pour un perceptron multicouches dans une tâche de classification.
```



## Réseaux de neurones multicouches


Un réseau de neurones *profond* (ou *deep learning model*) est formé de plusieurs couches, où une couche est un réseau de neurones "simple" (*i.e.*, une brique). Les couches sont "empilées" successivement, c'est-à-dire que **la sortie d'une couche correspond à l'entrée de la couche suivante**. L'entrée de la première couche est l'entrée $x$ et la sortie de la dernière couche correspond à la prédiction du réseau de neurones profond. Chaque couche $f_k$ est une fonction mathématique paramétrée par $\theta_k$. Les paramètres $\theta$ du réseau de neurones profond correspondent à la concaténation des paramètres $\theta^1, \dots, \theta^{\ell}$. La [](#fig_stack_layers) illustre le concept de réseaux de neurones profond.


```{figure} img/empilement_couches_deep_learning.svg
:label: fig_stack_layers
Réseau de neurones profond constitué de $\ell$ couches (*ie.*, neurones "simples").
```

Mathématiquement parlant, un réseau de neurones profond correspond à une composition de fonctions (correspondant aux couches).

$$
f_\theta(x) = f^\ell_{\theta^\ell} \circ f^{\ell-1}_{\theta^{\ell-1}} \circ \dots \circ f^1_{\theta^1}(x)
$$

```{important}
La rétropropagration nécessite de calculer la loss par rapport aux paramètres du réseau de neurones. Ce gradient peut être calculé à l'aide de la règle de dérivation en chaîne à partir de $\frac{\partial \mathcal{L}}{h_\ell}$, $\frac{\partial f_k}{\theta_k}$ et $\frac{\partial f_k}{h_{k-1}}$ pour tout $k=1,\dots,\ell$. Voir la Section @obtenir-le-gradient pour un exemple.
```








