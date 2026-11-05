# Modèle de classification

Une tâche classique d'apprentissage est celle de classification. L'objectif consiste à classer les entrées, c'est-à-dire à prédire à quelle classe appartient une entrée. Le nombre et le type de classes sont spécifiques à chaque tâche. Voici quelques exemples de tâches de classification : 

* détecter si un mail est un spam ou non,
* déterminer quel animal est sur une photo,
* déterminer si une tumeur est tumeur maligne ou bégnigne en fonction d'informations données en entrée (scanner, irm, résultats d'analyse, etc).

On note $\mathcal{C} = \{c_1,\dots,c_k\}$ l'ensemble des classes. 



## Approche probabiliste

Afin de classer une entrée, le modèle va donner pour chaque classe la probabilité que cette entrée appartienne à cette classe. La classe prédite est celle de plus grande probabilité.



Pour cela, étant donné une entrée $x \in X$, le modèle va prédire un score $s_c$ pour chaque classe $c \in \mathcal{C}$. Les scores vont être transformés en probabilités en utilisant la fonction  *softmax*. La probabilité $P(y = c | x)$ que l'entrée $x$ appartienne à la classe $c \in \mathcal{C}$ est donnée par :
$$
P(y = c | x) = \frac{e^{s_c}}{\sum_{c \in \mathcal{C}} e^{s_c}}
$$

```{note} Explications

Une distribution de probabilités sur les classes correspond à une valeur positive ou nulle associée à chaque classe (ie, sa probabilité) telle que la somme des valeurs est égale à 1. Ceci est vérifié avec la fonction softmax car : 
* l'exponentiel assure que l'on a des valeurs positives,
* le dénominateur assure que la somme des valeurs pour toutes les classes soit égale à 1.

```

```{tip} Exemple
Dans le cadre d'une classifcation d'images avec quatre classes (chien, chat, oiseau, vélo), si le modèle prédit les scores $-12.3$, $45.6$, $35.45$ et $7.43$ pour les quatre classes, la probabilité que l'image soit classée comme un chat est :
$$
\frac{e^{45.6}}{e^{-12.3} + e^{45.6} + e^{35.45} + e^{7.43}} \approx 0.99996
$$

On obtient la distribution de probabilité suivante[^errArrondi] : 
* $P(y = \text{chien} | x) = 7.1505\mathrm{e}{-26}$, 
* $P(y = \text{chat} | x) = 0.99996$, 
* $P(y = \text{oiseau} | x) = 3.9075\mathrm{e}{-5}$,
* $P(y = \text{vélo} | x) = 2.6483\mathrm{e}{-17}$.



   

[^errArrondi]: La somme des valeurs ne fait pas exactement un car les valeurs affichées sont arrondies.
```

Par la suite, on notera par $p_\theta(\cdot|x)$ la distribution de probabilité sur les classes calculée à partir des scores prédits par le modèle $f$ paramétré par $\theta$ pour l'entrée $x$. Autrement dit : 
 
$$p_\theta(\cdot|x) = \text{softmax}(f_\theta(x))$$


## Fonction de loss

Durant l'entraînement, pour un dataset $\mathcal{D} = \{(x^i,y^i) | i=1,\dots,n\}$, on évalue la qualité du modèle en utilisant une fonction de loss. Généralement, pour des tâches de classification, on utilise la *Negative Log Likelihood (NLL)* :

$$
- \frac{1}{n} \sum_{i=1}^n \log (p_\theta(y^i|x^i))
$$


```{note} Explications

On souhaite maximiser la probabilité que le modèle prédise les bonnes classes pour toutes les entrées du dataset : 
$$
\prod_{i=1}^n p_\theta(y^i|x^i)
$$

En utilisant, le logarithme, le produit est transformé en somme[^log]. De plus, cela stabilise les calculs.

Comme on veut minimiser une loss, on minimise l'opposé du log likehood.[^maxMoinsf]

Diviser par $n$ permet d'avoir une loss moyenne mais ce n'est pas obligatoire.
```


## Entraînement 

Entraîner un modèle de classification $f$ paramétré par $\theta$ sur un dataset $\mathcal{D} = \{(x^i,y^i) | i=1,\dots,n\}$ consiste à trouver les paramètres $\theta$ qui minimisent la loss NLL, c'est-à-dire :
$$
\argmin_\theta - \frac{1}{n} \sum_{i=1}^n \log (p_\theta(y^i|x^i))
$$

Une fois le modèle entraîné, le modèle prédit pour une entrée la distribution de probabilité sur les classes et retourne la classe de probabilité maximum. Cela revient à calculer les scores pour chaque classe et à retourner la classe avec le score maximum.


[^log]: car $\log(ab) = \log(a) + \log(b)$
[^maxMoinsf]: car $\max f = - \min -f$. 

(obtenir-le-gradient)=
## Obtenir le gradient

Le gradient s'obtient en utilisant la règle de dérivation en chaîne : 
* On commence par calculer le gradient de la loss par rapport aux sorties du modèle.
* Pour chaque couche dans le sens inverse, on calcule le gradient de la loss par rapport aux paramètres de la couche. Ceci est possible car pour chaque couche, on sait dériver la sortie de la couche par rapport aux paramètres de la couche.

### Gradient de la loss par rapport aux sorties

On commence par calculer gradient de la loss par rapport aux scores $s$ prédits. Comme on utilise la NLL loss et que les probabilités sont calculées par $\textrm{softmax}$, on obtient[^grad_output] :

```{math}
:label: eq:gradient_loss_sorties
\nabla_s L(x) = p - y
```
où $p$ est la distribution de probabilités calculée et $y$ est un one-hot vecteur qui vaut 1 pour la classe à laquelle appartient l'entrée $x$, et 0 sinon. 

On calcule ensuite le gradient de la loss par rapport aux paramètres en utilisant la règle de dérivation en chaîne.


`````{tip} Exemple

On considère une tâche d'apprentissage où $x$ est de taille 4, et le nombre de classes est 2. On a un perceptron avec une couche cachée de dimension 2 (fonction d'activation ReLU). On a alors 16 paramètres définis de la manière suivante ([](#param_MLP_example)).

```{figure} img/mlp_4_2_2_notation_W1_W2.svg
:align: center
:width: 90%
:label: param_MLP_example

Paramètres du réseau
```



Les neurones 5 et 6 correspondent aux deux neurones de la couche cachée et les neurones 7 et 8 aux deux neurones de sortie. Dans les calculs, on notera par $z_1$ et $z_2$ les valeurs des neurones 5 et 6 avant activation, par $h_1$ et $h_2$ leurs valeurs après activation, et par $s_1$ et $s_2$ les scores associés aux neurones 7 et 8.

On suppose que l'on a une entrée $\bar x = (0.8, 0.2, 0.7, 0.3)$ avec la sortie correspondant à la deuxième classe. 

````{note} Passe Forward
La passe forward nous donne les scores 

 

```{math}
:name: eq:scores_exemple
s = 
\begin{pmatrix}
1.6\\
2.04
\end{pmatrix}
```

En effet, les valeurs des couches cachées avant activation, notées $z_1$ et $z_2$, sont :
* $z_1 = 0.8 \times −2.5 + 0.2 \times 1.2 + 0.7 \times 0.1 + 0.3 \times 0.4 + 0.8 = -0.77$
* $z_2 = 0.8 \times 1.4 + 0.2 \times 0.8 + 0.7 \times −0.6 + 0.3 \times −0.7  + 0.5 = 1.15$

L'activation ReLU donne les valeurs (notées $h_1$ et $h_2$) : 
* $h_1 = 0$ (puisque $z_1$ est négatif)
* $h_2 = 1.15$

Les scores de sortie sont obtenus par combinaison linéaires de $h_1, h_2$ et les paramètres de la deuxième couche.

La fonction $\text{softmax}$, à partir des scores [](#eq:scores_exemple),  donne la distribution de probabilité de 
0.3917 pour la classe 1 et 0.6083 pour la classe 2.


La fonction de loss pour cette entrée est approximativement : 
$$- \log (0.6083) \approx 0.4971$$
````

````{note} Passe Backward
Pour entraîner notre modèle à l'aide de méthode de type gradient, il nous faut calculer le gradient $\nabla_\theta L(x)$[^gradient_matrice_et_biais]. Pour cela, on utilise la rétropropagation. 



[^gradient_matrice_et_biais]: $\theta$ correspond à l'ensemble des paramètres du modèle : $W^1$, $W^2$ et les biais.


**Gradient de la loss par rapport aux sorties**

On utilise la formule [](#eq:gradient_loss_sorties) pour calculer le gradient de la loss par rapport aux sorties. Dans cet exemple, $y=\begin{pmatrix} 0\\ 1\end{pmatrix}$ et $p \approx \begin{pmatrix}0.3917 \\ 0.6083\end{pmatrix}$.

Avec les valeurs calculées, on obtient
```{math}
\nabla_s L(x) \approx 
\begin{pmatrix}
~~~0.3917\\
-0.3917
\end{pmatrix}
```

**Dérivées partielles de la loss par rapport aux paramètres de la deuxième couche.** 

Les paramètres $W^2_{57}$, $W^2_{67}$ et $b_7$ interviennent uniquement pour le calcul de $s_1$ selon $s_1 = 0 \times W^2_{57} + 1.15 \times W^2_{67} + b_7$ car 0 et 1.15 sont les valeurs de la couche intermédiaire après activation. 

On a donc $\frac{\partial s_1}{\partial W^2_{57}} = 0$ et $\frac{\partial s_1}{\partial W^2_{67}} = 1.15$. 
Par la règle de dérivation en chaîne (*chain rule* en anglais), on a : $\frac{\partial L}{\partial W^2_{57}} = 0$ et $\frac{\partial L}{\partial W^2_{67}} \approx 0.3917 \times 1.15  \approx 0.4505$. 
Pour le biais $b_7$, on a $\frac{\partial L}{\partial b_7} \approx 0.3917 \times 1  \approx 0.3917$.




De manière similaire avec $s_2$, on trouve $\frac{\partial L}{\partial W^2_{58}} = 0$, $\frac{\partial L}{\partial W^2_{68}} \approx -0.3917 \times 1.15  \approx -0.4505$ et $\frac{\partial L}{\partial b_8} \approx -0.3917$.


**Gradient de la loss par rapport à $h_1$ et $h_2$.**

Par la règle de dérivation en chaîne [](#eq:backprop_mlp_hjt), on a :

$$\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial s_1}\frac{\partial s_1}{\partial h_1} + \frac{\partial L}{\partial s_2}\frac{\partial s_2}{\partial h_1} = 0.3917 \times 0.9 - 0.3917 \times 1.3 \approx 0.1567$$
car les deux variables intermédiaires $s_1$ et $s_2$ dépendent de $h_1$. De la même manière pour, $h_2$, on a :
$$\frac{\partial L}{\partial h_2} = \frac{\partial L}{\partial s_1}\frac{\partial s_1}{\partial h_2} + \frac{\partial L}{\partial s_2}\frac{\partial s_2}{\partial h_2} = 0.3917 \times 0.8 - 0.3917 \times 1.6 \approx -0.3134$$


**Gradient de la loss par rapport à $z_1$ et $z_2$.**

Comme $z_1 < 0$, la fonction ReLU correspond à $x \rightarrow 0$ dont la dérivée vaut 0. On a 
$$\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial h_1}\frac{\partial h_1}{\partial z_1} \approx  -0.1567 \times 0 = 0$$

Comme $z_2 \ge 0$, la fonction ReLU correspond à $x \rightarrow x$ dont la dérivée vaut 1. On a 
$$\frac{\partial L}{\partial z_2} = \frac{\partial L}{\partial h_2}\frac{\partial h_2}{\partial z_2} \approx  -0.3134 \times 1 = -0.3134$$


**Dérivées partielles par rapport aux paramètres de la première couche.**

On applique encore une fois la règle de dérivation en chaîne (avec la variable intermédiaire $z_1$ ou $z_2$ suivant les paramètres). On déduit les dérivées partielles de la loss par rapport à $W^1_{15}$, $W^1_{25}$, $W^1_{35}$ et $W^1_{45}$ valent 0 puisque $\frac{\partial L}{\partial z_1} = 0$.

Pour $W^1_{16}$, on a $\frac{\partial L}{\partial W^1_{16}} = \frac{\partial L}{\partial z_2}\frac{\partial z_2}{\partial W^1_{16}} \approx -0.3134 \times 0.8 = -0.2507$. On calcule les autres dérivées partielles de la même manière.

On obtient
$$
\nabla_\theta L(x)
= \begin{pmatrix} \frac{\partial L}{\partial W^1_{15}} \\ \frac{\partial L}{\partial W^1_{25}} \\ \frac{\partial L}{\partial W^1_{35}}\\ \frac{\partial L}{\partial W^1_{45}}\\ \frac{\partial L}{\partial b_{5}} \\ \frac{\partial L}{\partial W^1_{16}}\\ \frac{\partial L}{\partial W^1_{26}}\\\frac{\partial L}{\partial W^1_{36}}\\\frac{\partial L}{\partial W^1_{46}}\\\frac{\partial L}{\partial b_{6}}\\\frac{\partial L}{\partial W^2_{57}}\\\frac{\partial L}{\partial W^2_{67}}\\\frac{\partial L}{\partial b_{7}} \\ \frac{\partial L}{\partial W^2_{58}}\\\frac{\partial L}{\partial W^2_{68}}\\\frac{\partial L}{\partial b_{8}}\end{pmatrix} \approx \left(\begin{array}{r}0 \\ 0 \\ 0 \\ 0\\ 0\\-0.2507 \\-0.06027 \\ -0.2194 \\ -0.0940 \\-0.3134 \\ 0 \\ 0.4505 \\ 0.3917 \\ 0 \\ −0.4505 \\ −0.3917\end{array}\right)
$$

````


`````

[^grad_output]: *Détails des calculs.* Soit $t$ la classe correspondant à la sortie de l'entrée $x$. On $\alpha_t = \log(p_t) = s_t - \log(\sum_{c \in \mathcal{C}} e^{s_c})$.  Il faut distinguer deux cas. 

    * Premier cas, on calcule $\frac{\partial L}{\partial s_t}$.

        Si on note $\alpha_t = \log(p_t) = s_t - \log(\sum_{c \in \mathcal{C}} e^{s_c})$, alors on a
        $$
        \begin{array}{rcl}
        \frac{\partial L}{\partial s_t} & =  & \frac{\partial L}{\partial \alpha_t}\frac{\partial \alpha_t}{\partial s_t}\\
        &&\\
        & = & \frac{\partial(- s_t + \log(\sum_{c \in \mathcal{C}} e^{s_c}))}{\partial s_t}\\
         &&\\
        & = & -1 + p_t
        \end{array}
        $$
        car, en posant $Z = \sum_{c \in \mathcal{C}} e^{s_c}$, on a : 
        
        \begin{equation}
        \frac{\partial \log(Z)}{\partial s_t}  = \frac{1}{Z}\frac{\partial Z}{\partial s_t} = \frac{e^{s_t}}{Z} = p_t
        \end{equation}
        

    * Deuxième cas, on calcule $\frac{\partial L}{\partial s_j}$, $j \neq t$.
    $$
        \begin{array}{rcl}
        \frac{\partial L}{\partial s_j} & =  & \frac{\partial L}{\partial \alpha_t}\frac{\partial \alpha_t}{\partial s_j}\\
        &&\\
        & = &  \frac{\partial( - s_t + \log(\sum_{c \in \mathcal{C}} e^{s_c}))}{\partial s_j}\\
         &&\\
        & = &  p_j
        \end{array}
        $$


