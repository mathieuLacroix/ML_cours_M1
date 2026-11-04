---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Perceptron multicouches

Dans ce chapitre, nous définissons le perceptron. Celui-ci a été initialement proposé par @rosenblatt1958perceptron. Le perceptron a été ensuite généralisé au perceptron multicouche par @minsky1969perceptrons. Ce type de réseaux de neurones ont été largement utilisés à partir des années 1980 quand l'entraînement par rétropropagation a été introduite. Ils sont encore largement utilisés comme briques de réseaux de neurones plus complexes tels que les réseaux de neurones convolutifs par exemple.

## Couche dense

Une couche dense est une brique utilisée pour construire différents réseaux de neurones profonds. Elle consiste  en une application linéaire[^biais_affine] des entrées suivi de l'application d'une fonction non linéaire appelée *fonction d'activation* appliquée à chaque composante du vecteur obtenu. Notons par $d_x$ la taille de l'entrée et par $d_y$ la taille de la sortie. 


L'application linéaire[^biais_affine] donne un vecteur $z \in \mathbb{R}^{d_y}$ calculé selon 
$$
z = Wx + b
$$
où $W$ est une matrice de $\mathbb{R}^{d_y \times d_x}$ et $b$ un vecteur de $\mathbb{R}^{d_y}$ appelé *biais*.

La sortie de la couche dense est donnée en appliquant une fonction non linéaire $\sigma: \mathbb{R} \rightarrow \mathbb{R}$ à chaque composante du vecteur $z$. On notera la couche dense de la manière suivante :
$$
f(x) = \sigma(Wx + b)
$$

```{warning} Attention
La notation laisse penser que la fonction $\sigma$ est n'importe quelle fonction prenant en entrée un vecteur. Ce n'est pas le cas ! La fonction est appliquée sur chaque composante séparément afin de pouvoir facilement dériver à travers cette fonction. 
```


[^biais_affine]: En réalité, l'application est affine puisque l'on considère un biais mais on utilisera le terme linéaire puisque c'est celui utilisé dans la communauté.




Dans ce cours, on considère uniquement comme fonction d'activation la fonction ReLU (*Rectified Linear Unit*) qui est définie comme suit : 
$$
ReLU(x) = \max\{0,x\}
$$
La [](#fonctionReLU) est le graphe de la fonction ReLU.

```{figure} img/ShallowReLU.svg
:label: fonctionReLU
:width:50%
Fonction ReLU.
```



````{note} Fonctions d'activation
Il existe d'autres fonctions d'activation qui peuvent être utilisées dans un réseau de neurones. Voici quelques exemples : tangeante hyperbolique, sigmoïde, ReLU et ReLU linéaire. 
```{figure} img/activation_functions.png
```
L'objectif de ces fonctions est à chaque fois de créer un seuil.
````

### Neurones actifs et inactifs

Pour une entrée $x$, un neurone de la sortie est dit *actif* si la valeur de la combinaison affine entrant dans ce neurone est positive, c'est-à-dire qu'une valeur positive sort de ce neurone. Il est dit *inactif* autrement. Autrement dit, le $j^{\text eme}$ neurone de la sortie est actif si 
$$
\sum_{i = 1}^{d_x} w_{ji}x_i > 0
$$

### Représentation d'un réseau de neurones sous forme de schéma

Les réseaux de neurones sont souvent représentés par des schémas permettant de visualiser la structure du réseau de neurones. La [](#couche_dense_ex) donne un exemple d'une couche dense à 2 entrées et 3 sorties. 
```{figure} img/perceptron_relu_exemple.svg
:label: couche_dense_ex
Exemple de couche dense avec une entrée de dimension 2 et une sortie de dimension 3. 
```

On peut représenter le biais sous la forme d'une coordonnée additionnelle pour l'entrée. Concrètement, on obtient 
$$
f(x) = \sigma \left( \begin{bmatrix} W & b \end{bmatrix} \begin{pmatrix} x \\ 1\end{pmatrix}\right)
$$

Les paramètres de la couche dense correspondent alors à une seule matrice $\begin{bmatrix} W & b \end{bmatrix}$ contenant aussi les biais.
Par la suite, on notera $s$ la sortie du modèle[^sortieDiffLabels] et on représentera schématiquement la couche cachée de la manière suivante : 

```{figure} img/couche_dense_factorisee.svg
:width: 80%
:label: coucheDenseFactorisee
Représentation d'une couche dense sous forme factorisée.
```
[^sortieDiffLabels]: pour différentier la prédiction (ie, sortie $s$) du modèle et le label $y^i$ de l'entrée $x^i$. 


### Nombre de paramètres

Les paramètres d'une couche dense correspond à la matrice $W$ et au vecteur de biais $b$ (il n'y a pas de paramètres pour la fonction d'activation). Si l'entrée est de dimension $d_x$ et la sortie de dimension $d_y$, le nombre de paramètres de la couche dense est : $$d_x \times d_y + d_y$$
Le nombre de paramètres de la couche dense de la [](#couche_dense_ex) est 2 * 3 pour la matrice $W$ et de 3 pour le vecteur $b$, soit un total de 9 paramètres.

## Perceptron


Le perceptron consiste en une couche dense suivie d'une application linéaire[^biais_affine]. Le perceptron prend en entrée un vecteur $x \in \mathbb{R}^{d_x}$ et retourne un vecteur $s \in \mathbb{R}^{d_y}$. Le vecteur correspondant à la sortie de la couche dense est appelé *couche cachée* (hidden layer). Sa dimension $d$ est un hyperparamètre du perceptron. Mathématiquement, le perceptron est donné par : 
$$
\begin{array}{l}
h = \sigma(W^1 x + b^1)\\
s = W^2 h + b^2
\end{array}
$$
Schématiquement, un perceptron est représentée par la [](#fig_perceptron).



```{figure} img/perceptron_factorise.svg
:label: fig_perceptron
Schéma d'un perceptron.
```




```{note}

Historiquement, le perceptron correspondait à une couche dense dont la sortie était de dimension 1 et servait pour la classification binaire. On utilise ici le terme perceptron dans un cadre un peu plus général (couche dense suivi d'une couche linéaire).

```


### Nombre de paramètres

Les paramètres d'un perceptron sont $W^1$, $b^1$, $W^2$ et $b^2$. Le nombre total de paramètres est :
$$d \times d_x + d + d_y \times d + d_y$$



### Approximateur universel

Le perceptron est un **_approximateur universel_**, c'est-à-dire que n'importe quelle fonction[^fonctionapprox] peut être approximée par un perceptron pour n'importe quelle précision. On ne présente pas de preuve mais plutôt l'idée de la preuve. On commence par le cas où la fonction est une fonction de $\mathbb{R}$ dans $\mathbb{R}$. On explique ensuite brièvement comment cela se généralise. 

[^fonctionapprox]: On peut approximer n'importe quelle fonction continue et bornée sur un compact (ensemble fermé et borné).



#### Cas une dimension

Considérons une fonction $f$ de $\mathbb{R}$ dans $\mathbb{R}$ continue et bornée sur un intervalle fini $[a,b]$. Cette fonction peut être approximée par une fonction linéaire par morceaux avec un nombre fini de morceaux pour n'importe quelle précision[^approximationFLPM]. Le nombre de morceaux augmente avec la précision de l'approximation. La [](#approximationFLPM) donne un exemple d'approximation par une fonction linéaire par morceaux.

```{figure} img/ShallowApproximate.svg
:label: approximationFLPM
Une fonction non linéaire (en pointillés) est approximée par une fonction linéaire par morceaux. a) Approximation par une fonction à 5 morceaux. b) Approximation par une fonction à 10 morceaux. a) Approximation par une fonction à 20 morceaux. 
```

```{important}
Un perceptron à 1 entrée, une couche cachée de $d$ neurones et une sortie permet de représenter une fonction linéaire par morceaux avec $d$ morceaux sur un intervalle fini $[a,b]$.
```

````{seealso}
Il est possible de visualiser la fonction linéaire donnée par un perceptron avec une entrée et une sortie de dimension 1 et une couche cachée de dimension 3 [ici (Figure "*1D shallow network (ReLU)*")](https://udlbook.github.io/udlfigures/).
````


```{tip} Comment déterminer les paramètres d'un perceptron pour représenter une fonction linéaire par morceaux ?
Les paramètres du perceptron pour représenter une fonction linéaire par morceaux $f$ (avec $d$ morceaux) peuvent être trouvés facilement. 

Comme l'entrée et la sortie sont de dimension 1, alors $W^1$ et $W^2$ sont des vecteurs notés $w^1$ et $w^2$, $b^1$ est un vecteur et $b^2$ est un scalaire.


Si le minimum de la fonction sur l'intervalle $m = \min_{x \in [a,b]} f(x)$ est positif ou nul, alors $b^2 = 0$. Autrement, $b^2 = -m$ et on approxime la fonction $f'$ définie par $f'(x) = f(x) + m$. 


Comme l'entrée est de dimension 1, le $j^{\text eme}$ neurone de la couche cachée est associé à deux paramètres : $w^1_j$ et $b^1_j$.

Le $j^{\text eme}$ neurone de la couche cachée est actif à partir du $j^{\text eme}$ morceau de la fonction.
Afin que le $j^{\text eme}$ neurone soit inactif pour les morceaux précédents, on fixe $w^2_j = 1$ si le $j^{\text eme}$ morceau est croissant, et -1 autrement. Chaque morceau $j$ est alors défini par :
$$
\sum_{i = 1}^j w^2_i( w^1_i x + b^1_i)
$$

Pour déterminer les valeurs de $w^1$ et $b^1$, on détermine itérativement en commençant du premier morceau en résolvant un système à deux équations et deux inconnus pour déterminer les coefficients d'une fonction affine (droite) passant par deux points.
```






[^approximationFLPM]: Cela provient du théorème de Heine qui assure que pour toute erreur $\epsilon$, on peut considérer un sous-intervalle de taille $\delta$ tel que la fonction ne variera pas de plus d'$\epsilon$ sur cet intervalle. On peut donc diviser l'intervalle $[a,b]$ en un nombre fini d'intervalles de taille $\delta$ et déterminer une fonction affine sur chaque intervalle dont l'erreur par rapport à la fonction à approximer est au plus $\epsilon$.



```{hint} Remarque
On ne discute du résultat que pour un perceptron dont la fonction d'activation est la fonction ReLU mais le résultat peut être généralisé à d'autres fonctions d'activation non linéaires @cybenko1989approximation, @hornik1991approximation.
```


#### Cas général

Ce résultat d'approximateur universel s'étend pour les fonctions de $\mathbb{R}^m$ dans $\mathbb{R}^n$. En effet,  
une telle fonction continue sur un compact peut être approximée par une fonction linéaire par morceaux[^extensionHeine].

De plus, un perceptron peut approximer n'importe fonction linéaire par morceaux (nombre fini de morceaux) sur un compact @leshno1993Multilayer. Un perceptron peut donc approximer n'importe quelle fonction continue sur un compact.


````{note}
Contrairement au cas des fonctions de $\mathbb{R}$ dans $\mathbb{R}$, un perceptron ne peut pas représenter n'importe quelle fonction linéaire par morceaux (mais seulement les approximer). C'est le cas par exemple de la fonction de $\mathbb{R}^2$ dans $\mathbb{R}$ définie par $(x,y) \mapsto \max\{x,y,0\}$ :

```{code-cell} python
---
tags: [remove-input]
---
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

D = Path("../../data/results")
X = np.load(D / "grid_X.npy")
Y = np.load(D / "grid_Y.npy")
Z_exact = np.load(D / "Z_exact.npy")

# 2. Tracé de la surface 3D avec Plotly
fig = go.Figure(data=[go.Surface(
    x=X, 
    y=Y, 
    z=Z_exact, 
    colorscale='Viridis', # Correspond au cmap='viridis' de matplotlib
    colorbar=dict(title="Valeur de Z") # Ajoute la barre d'échelle de couleurs
)])

# 3. Configuration des labels des axes et du style
fig.update_layout(
    # title="Surface 3D : Z = max(x, 0, y)",
    scene=dict(
        xaxis_title='Axe X',
        yaxis_title='Axe Y',
        zaxis_title='Z = max(x, 0, y)'
    ),
    autosize=True,
    margin=dict(l=0, r=0, b=0, t=40) # Ajuste les marges pour le web
)

# 4. Affichage de la figure interactive
fig.show()
```

Pour le voir, il faut comprendre que la sortie $ReLU((W_i^1)^\top x + b_i)$ du neurone $i$ de la couche cachée définit un hyperplan d'équation $(W_i^1)^\top x + b_i = 0$ correspondant à une "pliure" de l'espace (d'un côté la valeur 0, de l'autre la valeur $(W_i^1)^\top x + b_i$ >0). Un perceptron ne peut donc représenter que des fonctions linéaires par morceaux dont les "pliures" sont des hyperplans (des droites en dimension deux). Dans la figure, ce n'est pas le cas. En effet, la fonction est "pliée" selon la droite $x = y$ seulement sur une partie de l'espace (que pour $x \geq 0$).

Par contre, il est possible d'approximer cette fonction avec un perceptron. Les figures ci-dessous représentent les approximations de la fonction obtenues en entraînant un perceptron contenant respectivement 8 et 64 neurones dans la couche cachée.

```{code-cell} python
---
tags: [remove-input]
---
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

D = Path("../../data/results")
X = np.load(D / "grid_X.npy")
Y = np.load(D / "grid_Y.npy")
Z_exact = np.load(D / "Z_exact.npy")
widths = [int(n) for n in np.load(D / "widths.npy")]
Z_hat = {n: np.load(D / f"Z_hat_n{n}.npy") for n in widths}
max_err = dict(zip(widths, np.load(D / "max_err.npy")))

# --- Figure 1 : surfaces approchées pour quelques largeurs ---
# shown = [2, 4, 8, 32]
shown = [8, 64]
fig = make_subplots(
    rows=2, cols=1,
    specs=[[{"type": "surface"}], [{"type": "surface"}]],
    subplot_titles=[f"{n} neurones (erreur abs max = {max_err[n]:.3f})" for n in shown],
    vertical_spacing=0.05,
)
zmin, zmax = Z_exact.min(), Z_exact.max()
for k, n in enumerate(shown):
    fig.add_trace(
        go.Surface(x=X, y=Y, z=Z_hat[n], colorscale="Viridis",
                   cmin=zmin, cmax=zmax, showscale=(k == 0),
                   colorbar=dict(title="Z")),
        row=k + 1, col=1,
    )

# Plage z commune, calculée sur toutes les surfaces affichées
lo = min(Z_hat[n].min() for n in shown)
hi = max(Z_hat[n].max() for n in shown)
zr = [min(lo, zmin) - 0.1, max(hi, zmax) + 0.1]

cam = dict(eye=dict(x=1.6, y=-1.6, z=0.9))
fig.update_layout(
    height=900,
    margin=dict(l=0, r=0, b=0, t=40),
    scene=dict(camera=cam, zaxis=dict(range=zr)),
    scene2=dict(camera=cam, zaxis=dict(range=zr)),
)
fig.show()
# # --- Figure 2 : erreur max en fonction de n (échelle log-log) ---
# fig2 = go.Figure(go.Scatter(
#     x=widths, y=[max_err[n] for n in widths], mode="lines+markers"))
# fig2.update_layout(
#     xaxis=dict(title="Nombre de neurones n", type="log"),
#     yaxis=dict(title="Erreur max sur [-5, 5]²", type="log"),
# )
# fig2.show()

# --- Figure 3 : carte de l'erreur pour une largeur donnée ---
# n = 128
# fig3 = go.Figure(go.Surface(x=X, y=Y, z=np.abs(Z_hat[n] - Z_exact),
#                             colorscale="Reds", colorbar=dict(title="|erreur|")))
# fig3.update_layout(title=f"Erreur absolue, n = {n}",
#                    margin=dict(l=0, r=0, b=0, t=40))
# fig3.show()
```
```{note} Remarque
La fonction $(x,y) \mapsto \max\{x,y,0\}$ peut être représentée de manière exacte par un perceptron à seulement deux couches cachées contenant au total 5 neurones.
```


````



[^extensionHeine]: Le théorème de Heine est valide pour les fonctions de $\mathbb{R}^m$ dans $\mathbb{R}^n$. On peut donc découper le compact en un nombre fini de petits morceaux, appelés *simplexes*, de manière à ce que sur chaque simplexe, la fonction varie d'au plus $\epsilon$. On approxime ensuite la fonction sur un simplexe par une fonction linéaire.





## Perceptron multicouches


Le perceptron multicouches consiste en une succession de couches denses suivi d'une couche linéaire. Le nombre de couches cachées ainsi que la dimension de chaque couche cachée sont des hyperparamètres.


Mathématiquement, un perceptron multicouches avec une entrée de dimension $d_x$, $\ell$ couches cachées de dimension $d_1,\dots,d_\ell$ et une sortie $s$ de dimension $d_y$ est donné par :



\begin{align*}
& h^1 = \sigma(W^1 x + b^1)\\
& h^{t} = \sigma(W^t h^{t-1} + b^{t}) & & \forall t \in \{2,\dots,\ell\}\\
& s = W^{\ell + 1}h^{\ell} + b^{\ell + 1}
\end{align*}

Le nombre de paramètres est égal à :
$$(d_x + 1) \times d_1 + \sum_{t =2}^\ell ((d^{t-1}+1) \times d^t) + (d^{\ell} +1) \times d_y$$

Un perceptron multicouches peut être représenté comme sur la [](#fig_mlp_generique).

```{figure} img/mlp_generique2.svg
:label:fig_mlp_generique
Perceptron multicouches.
```


### Capacité d'approximation

Si n'importe quelle fonction continue sur un compact peut être approximée par un perceptron, le nombre de neurones dans la couche cachée peut être très important (même si fini). Le perceptron multicouches permet d'approximer une fonction avec généralement moins de neurones, c'est-à-dire moins de paramètres.
En effet, il existe des perceptrons multicouches avec un nombre polynomial de neurones cachés qui, lorsque l'on souhaite les approximer avec un perceptron (monocouche), nécessite un nombre exponentiel de neurones cachés [@liang2016deep].


Il faut garder à l'esprit qu'en pratique, le nombre de neurones, et donc de paramètres, peut ne pas être suffisant pour approximer correctement une tâche d'apprentissage. Un autre problème qui peut apparaître est que le nombre de paramètres est trop important pour entraîner correctement ce réseau de neurones, c'est-à-dire que la méthode d'entraînement basée sur le gradient stochastique fournit un minimum local de mauvaise qualité ou met trop de temps pour converger vers un minimum. 
Il est donc possible en pratique d'avoir un mauvais classifieur. Dans ce cas, une alternative peut être de considérer une architecture plus adaptée à la tâche d'apprentissage. C'est le cas par exemple de l'utilisation de réseaux de neurones convolutifs pour des tâches d'apprentissage impliquant des images, telles que la classification d'images.

## Rétropropagation

Il est facile avec un perceptron multicouches d'obtenir le gradient de la sortie par rapport aux différents paramètres du modèle. Grâce à la règle de dérivation en chaîne, si l'on sait obtenir le gradient de la loss par rapport aux sorties du perceptron multicouches, on peut alors obtenir le gradient de la loss par rapport aux paramètres, et donc entraîner le modèle (c.f. [](#entrainer-un-modele)). 


Pour chaque couche cachée $t$, on note par $z^t$ la valeur avant activation et $h^t$ la valeur après activation.
La sortie $s$ correspond à $z^{\ell + 1}$.

En utilisant la règle de dérivation en chaîne, on obtient :
```{math}
:label: eq:backprop_mlp_param_chaine_rule
\frac{\partial L}{\partial w_{ij}^t} = \frac{\partial L}{\partial h_j^t} \frac{\partial h_j^t}{\partial z_j^t} \frac{\partial z_j^t}{\partial w_{ij}^t}
```

```{math}
:label: eq:backprop_mlp_biais_chaine_rule
\frac{\partial L}{\partial b_{j}^t} = \frac{\partial L}{\partial h_j^t} \frac{\partial h_j^t}{\partial z_j^t} \frac{\partial z_j^t}{\partial b_{j}^t}
```


Comme $z_j^t = \sum_{i = 1}^{d_{t-1}} w_{ij}^t h_i^{t-1} + b_j^t$, on a :
```{math}
:label: eq:backprop_mlp_param
\frac{\partial z_j^t}{\partial w_{ij}^t} = h_i^{t-1}
```
où $h_i^{t-1}$ est la valeur calculée lors du forward pour le $i^{\text eme}$ neurone de la couche cachée $t-1$. Pour le biais, on a : 
```{math}
:label: eq:backprop_mlp_biais
\frac{\partial z_j^t}{\partial b_{j}^t} = 1
```

La fonction d'activation est la fonction ReLU. Elle est dérivable[^ReLU_derivable] et l'on connaît analytiquement sa dérivée. On obtient alors : 
```{math}
:label: eq:backprop_mlp_zjt
\frac{\partial h_j^t}{\partial z_j^t} =  \left\{  \begin{array}{ll} 1 & \text{ si } z_j^t \geq 0,\\ 0 & \text{ sinon.} \end{array} \right.
```

Il reste maintenant à déterminer $\frac{\partial L}{\partial h_j^t}$. Pour cela, on utilise la règle de dérivation en chaîne : 
```{math}
:label: eq:backprop_mlp_hjt
\frac{\partial L}{\partial h_j^t} = \sum_{i = 1}^{d_{t+1}} \frac{\partial L}{\partial z_j^{t+1}}\frac{\partial z_j^{t+1}}{\partial h_j^t}  = \sum_{i = 1}^{d_{t+1}} \frac{\partial L}{\partial z_j^{t+1}}w_{ji}^{t+1}
```


La fonction de loss est choisie de manière à pouvoir calculer analytiquement $\frac{\partial L}{\partial s}$. On peut alors calculer la dérivée de la loss par rapport aux paramètres de la dernière couche, puis calculer la dérivée de la loss par rapport à la dernière couche cachée. On continue ensuite ces calculs jusqu'aux paramètres de la première couche en utilisant les équations  [](#eq:backprop_mlp_param), [](#eq:backprop_mlp_biais), [](#eq:backprop_mlp_zjt) et [](#eq:backprop_mlp_hjt).


```{seealso}
Pour un exemple de calcul du gradient de la loss par rapport aux paramètres d'un perceptron, voir [](#obtenir-le-gradient).
```



[^ReLU_derivable]: La fonction ReLU est dérivable sauf au point 0.
