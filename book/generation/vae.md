# Autoencodeurs variationnels

Nous avons vu ici un exemple de classification d'images, qui consiste, étant donné une image, à la classer dans une catégorie (chiffre, animal, etc). 
Nous abordons dans cette partie la génération d'images. On parle alors d'*intelligence artificielle générative (IAG)*.
Ces modèles ont été popularisés par les modèles de langues larges (LLM) type [chatGPT](https://chatgpt.com/), [le chat](chat.mistral.ai/chat) ou [Claude](https://claude.ai/). Le modèle présenté ici est dédié à la génération d'images.


Contrairement au cas précédent, on est dans un apprentissage **non supervisé**. Notre dataset est constitué uniquement des entrées (ie, images). Le dataset d'entraînement est :
$$
\mathcal{D} = \{x^1,\dots,x^n\}
$$


Il existe plusieurs types de modèles génératifs. Nous nous intéressons ici à l'*autoencodeur variationnel* (Variational Auto-Encoder (VAE) en anglais) introduit dans [@kingma2013auto]. 


Les VAE sont des modèles probabilistes, c'est-à-dire que l'on va apprendre une distribution de probabilités de telle manière que :
1. il est facile de sampler dans cette distribution,
2. il semble vraisemblable que les données du dataset aient été samplées à partir de cette distribution de probabilités (voir [](#iag_distribution)[^figUnderstanding]).

[^figUnderstanding]: Figure issue du livre [@prince2023understanding]

```{figure} img/iag_proba_distribution.png
:label: iag_distribution
Chaque image représente une distribution de probabilités sur $x_1$ et $x_2$ : plus l'image est claire, plus la probabilité est élevée. Les points en cyan correspondent aux entrées de notre dataset. Durant l'entraînement, on apprend à paramétrer une distribution sur l'ensemble $X$ dans laquelle les entrées du dataset ont une grande probabilité. Les quatre images correspondent à différentes époques de l'apprentissage (évolution de gauche à droite).
```

## Variables latentes

Il est extrêmement compliqué de définir directement une distribution de probabilités sur l'ensemble $X$ satisfaisant les deux conditions données ci-dessus. Pour cela, on va utiliser un *espace latent* $Z$ avec une distribution de probabilités $p(z)$ qui va nous permettre de conditionner la probabilité sur l'espace $X$ selon : 
$$
p(x) = \int_z p(x,z) = \int_z p(x|z)p(z) 
$$
où :
* $p(x)$ est la distribution que l'on cherche à apprendre,
* $p(x|z)$ est la *vraisemblance* (*likelihood*) de $x$ sachant $z$,
* $p(z)$ est la *distribution préalable* (*prior distribution*).

L'espace latent est un espace de $\mathbb{R}^\ell$ où $\ell$ est un hyperparamètre du modèle qui dépend de l'entrée $X$. On suppose que la distribution $p(z)$ est une loi normale multivariée de centre $\mathbf{0}$ et variance $I_{\ell}$ (*i.e.*, les variables $Z_1, \dots, Z_{\ell}$ sont indépendantes). De cette manière, il est facile de sampler dans cette distribution.

On considère aussi que $p(x|z)$ est une loi normale de centre $\mu$ et de variance $\Sigma$. Il est également facile de sampler dans cette distribution. Par contre, il est difficile de déterminer le centre $\mu$ et la variance $\Sigma$, c'est pour cela que l'on utilise un **réseau de neurones** pour prédire ces paramètres $\mu$ et $\Sigma$ en fonction de $z$. Ce réseau est appelé **_décodeur_**. Le décodeur est noté dans la suite $f$ et dépend de paramètres $\phi$. On notera par $\mu_\phi(z)$ et $\Sigma_\phi(z)$ les valeurs prédites par le réseau de neurones $f$ pour le centre et la variance. De même, on écrira $p_\phi(x\vert z)$ le likelihood paramétré par le décodeur.



La distribution de probabilités est donc : 
$$
p(x) = \int_z p_\phi(x\vert z) p(z) = \mathbb{E}_{z \sim p(z)}\left[p_\phi(x \vert z)\right]
$$
avec 
$$
p_\phi(x\vert z) = \mathcal{N}(\mu_{\phi}(z), \Sigma_\phi(z))
$$

```{important}
En pratique, on supposera que la variance $\Sigma_\phi(z)$ est fixée à $\sigma_{dec}^2 I$ où $\sigma_{dec}$ est un scalaire fixé (non prédit par le décodeur) et $I$ la matrice identitaire. **Le décodeur prédit uniquement $\mu_{\phi}(z)$**.


De plus, on ne samplera pas un $x$ selon la distribution $p_\phi(x\vert z)$ mais on prendra directement $\mu_{\phi}(z)$ qui correspond au mode de la distribution.
Sampler ne fait qu'ajouter du bruit et rend les images plus granuleuses. 
```

### Est-ce que cela permet de modéliser des distributions de probabilités complexes ?

Oui. La distribution $p(x)$ est un *modèle de mélange gaussien* (Gaussian mixture model) qui permet de modéliser des distribution complexes. La [](#figGMM)[^figUnderstanding] montre un exemple (simple) de modèle de mélange gaussien.

```{figure} img/VAENonLinearLVM.svg
:label:figGMM
Modèle de mélange gaussien. Pour chaque valeur de $z$, la distribution $p(x|z)$ est une gaussienne dont la moyenne et la variance dépendent de la valeur de $z$. La distribution est illustrée pour deux valeurs de $z$ (correspondant aux figures sur les deux "tranches"). Chaque distribution a la forme d'un cercle. En marginalisant selon $z$, on obtient une distribution complexe  représentant $p(x)$.
```




## Entraînement d'un VAE

Un modèle génératif cherche, étant donné un dataset $\mathcal{D} = \{x^1,\dots,x^n\}$ à déterminer une distribution de probabilités $p(x)$ qui maximise la log vraisemblance du dataset :
$$
\sum_{i = 1}^n \log(p(x^i))
$$

```{danger} Modèle impossible à entraîner !
Il n'est pas possible d'entraîner un tel modèle directement, à savoir
$$
\argmax_\phi \sum_{i=1}^n \log \mathbb{E}_{z \sim p(z)}\left[p_\phi(x^i \vert z)\right]
$$
car :
* il n'existe pas de forme close pour le calcul de l'espérance,
* il n'est pas possible d'approximer correctement cette espérance via des méthodes de Monte-Carlo à moins d'utiliser des milliards d'échantillons, car la grande majorité des valeurs de $z$ ont une probabilité (quasi) nulle de générer un $x$ donné (fléau de la dimension).
```

### Approche variationnelle

Puisqu'il n'est pas possible d'entraîner un tel modèle, on utilise une approche d'*échantillonnage préférentiel* (*Importance Sampling* en anglais), c'est-à-dire que l'on utilise pour échantillonner une distribution de probabilités $q(z)$ qui concentre sa masse autour des valeurs de $z$ qui ont une probabilité non négligeable de générer $x$. On a alors : 
$$
p(x) = \mathbb{E}_{z \sim p(z)}\left[p_\phi(x \vert z)\right] = \mathbb{E}_{z \sim q(z)}\left[\frac{p(z)}{q(z)}p_\phi(x \vert z)\right]
$$

Idéalement, on souhaite échantillonner selon $p(z \vert x)$. On utilise donc une **_approche variationnelle_**, c'est-à-dire que l'on choisit une distribution $q(z\vert x)$ qui approxime la distribution $p(z \vert x)$. On suppose que **$q(z\vert x)$ est une distribution normale** de centre $\mu$ et de variance $Diag(\sigma^2)$[^posteriorVarInd]. Comme les paramètres de cette distribution sont difficiles à déterminer, on suppose qu'il sont prédits par un **deuxième réseau de neurones**, appelé **_encodeur_** dont les paramètres seront notés par la suite $\theta$. Pour mettre en évidence que les paramètres et la distribution dépendent de la prédiction de l'encodeur, on les notera $\mu_\theta(x)$, $\sigma_\theta(x)$ et $q_{\theta}(z \vert x)$. On a donc :


[^posteriorVarInd]: étant donné la forme de la variance, les $\ell$ variables aléatoires de $q_{\theta}(z \vert x)$ sont indépendantes. 






```{math}
:label: eq:approche_var_px
p(x) = \mathbb{E}_{z \sim p(z)}\left[p_\phi(x \vert z)\right] = \mathbb{E}_{z \sim q_\theta(z\vert x)}\left[\frac{p(z)}{q_\theta(z \vert x)}p_\phi(x \vert z)\right]

```





### Fonction de loss

On ne va pas entraîner le modèle pour maximiser la log vraisemblance mais plutôt utiliser une **borne inférieure** de cette log vraisemblable. En effet, par l'[inégalité de Jensen](https://www.statlect.com/fundamentals-of-probability/Jensen-inequality), on a : 
$$
\begin{array}{lll}
log(p(x)) &=& \log\left(\mathbb{E}_{z \sim q_\theta(z\vert x)}\left[\frac{p(z)}{q_\theta(z \vert x)}p_\phi(x \vert z)\right]\right)\\ 
&\geq&       \mathbb{E}_{z \sim q_\theta(z\vert x)}\left[\log \left(\frac{p(z)}{q_\theta(z \vert x)}p_\phi(x \vert z)\right)\right]\\
 & = &  \mathbb{E}_{z \sim q_\theta(z\vert x)}\left[\log \left(p_\phi(x \vert z)\right )\right] + \mathbb{E}_{z \sim q_\theta(z\vert x)}\left[\log \left(\frac{p(z)}{q_\theta(z \vert x)}\right)\right]\\
 &=& \mathbb{E}_{z \sim q_\theta(z\vert x)}\left[\log \left(p_\phi(x \vert z)\right )\right] - KL(q_\theta(z\vert x) \parallel p(z))
\end{array}
$$

La dernière ligne provient de la définition de la [divergence de Kullback-Leibler](#divergenceKL).


```{tip} Définition : borne inférieure de l'évidence
Étant donné $x$, $\theta$ et $\phi$, la valeur 
$$
ELBO_{\theta, \phi}(x) = \mathbb{E}_{z \sim q_\theta(z\vert x)}\left[\log \left(p_\phi(x \vert z)\right )\right] - KL(q_\theta(z\vert x) \parallel p(z))
$$
est appelée **_borne inférieure de l'évidence_** (*Evidence Lower BOund (ELBO)* en anglais) car $p(x)$ est appelé l'*évidence* dans l'approche Bayésienne.
```


L'entraînement consiste à maximiser l'ELBO, donc il devient : 

```{math}
:label: eq_training_VAE
\argmax_{\theta,\phi} \sum_{i=1}^n \mathbb{E}_{z \sim q_\theta(z\vert x)}\left[\log \left(p_\phi(x \vert z)\right )\right] - KL(q_\theta(z\vert x) \parallel p(z))

```
Le premier terme de l'ELBO est un terme de **reconstruction** : il pousse le modèle à bien reconstruire $x$ à partir d'un $z$ échantillonné selon $q_\theta$. Le second terme est un terme de **régularisation** : la [divergence de Kullback-Leibler](#divergenceKL) mesure la dissimilarité entre $q_\theta(z\vert x)$ et le prior $p(z)$, et empêche l'encodeur de s'éloigner arbitrairement de $\mathcal{N}(0, I_\ell)$.


La [](#figArchiVAE) schématise un autoencodeur variationnel.

```{figure} img/VAEArch.svg
:label: figArchiVAE

Architecture d'un autoencodeur variationnel.
``` 


```{tip} Remarque
On peut remarquer que l'ELBO s'écrit aussi comme
$$
\begin{array}{lll}
ELBO(\theta, \phi) &=& \mathbb{E}_{z \sim q_\theta(z\vert x)}\left[\log \left(\frac{p(z)}{q_\theta(z \vert x)}p_\phi(x \vert z)\right)\right]\\ 
& = &      \mathbb{E}_{z \sim q_\theta(z\vert x)}\left[\log \left(\frac{p(x)p_\phi(z \vert x)}{q_\theta(z \vert x)}\right)\right] \\
 & = &  \mathbb{E}_{z \sim q_\theta(z\vert x)}\left[\log \left(p_\phi(x)\right )\right] + \mathbb{E}_{z \sim q_\theta(z\vert x)}\left[\log \left(\frac{p(x \vert z)}{q_\theta(z \vert x)}\right)\right]\\
 &=& \mathbb{E}_{z \sim q_\theta(z\vert x)}\left[\log \left(p_\phi(x)\right )\right] - KL(q_\theta(z\vert x) \parallel p(z\vert x))
\end{array}
$$
Cela signifie que maximiser l'ELBO revient à maximiser la log vraissemblance de $x$ tout en minimisant la divergence  de l'approximation $q_\theta(z\vert x)$ par rapport à la distribution $p(z\vert x)$.
```



```{note} Pourquoi des gaussiennes ?
Le choix de $p(z)$ et $q_\theta(z\vert x)$ comme lois normales n'est pas anodin : il permet de calculer le terme $KL(q_\theta(z\vert x) \parallel p(z))$ **analytiquement** (forme close), sans avoir recours à une estimation Monte-Carlo — contrairement au terme de reconstruction, qui lui doit être estimé en échantillonnant $z \sim q_\theta(z \vert x)$.
```

#### Formule analytique du terme de régularisation (divergence KL)

la divergence KL entre deux gaussiennes multivariées $\mathcal{N}(\mu_1, \Sigma_1)$ et $\mathcal{N}(\mu_2, \Sigma_2)$ est donnée par :

```{math}
:label: eq:divKL_lois_normales
KL\big(\mathcal{N}(\mu_1, \Sigma_1) \parallel \mathcal{N}(\mu_2, \Sigma_2)\big) = \frac12\left[\text{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2 - \mu_1)^\top \Sigma_2^{-1} (\mu_2 - \mu_1) - \ell + \log\frac{\det \Sigma_2}{\det \Sigma_1}\right]

```


Comme $q_\theta(z\vert x) = \mathcal{N}\left(\mu_\theta(x),Diag(\sigma_\theta(x)^2)\right)$ et $p = \mathcal{N}\left(0,I_\ell\right)$, pour calculer $KL\big(q_\theta(z\vert x) \parallel p(z)\big)$, on peut simplifier [](#eq:divKL_lois_normales) car [^fn:notationmu_theta_i]:
* $tr(\Sigma_2^{-1}\Sigma_1) = tr(\Sigma_1) = \sum_{i = 1}^\ell \sigma_\theta(x)_i^2$,
* $(\mu_2 - \mu_1)^\top \Sigma_2^{-1} (\mu_2 - \mu_1) = - \sum_{i = 1}^\ell \mu_\theta(x)_i$
* $\log\left(\frac{\det \Sigma_2}{\det \Sigma_1}\right) = - \sum_{i = 1}^\ell \log(\sigma_\theta(x)_i^2)$.

[^fn:notationmu_theta_i]: Les notations $\mu_\theta(x)_i$ et $\sigma_\theta(x)_i$ correspondent respectivement aux $i^\text{eme}$ composantes des vecteurs $\mu_\theta(x)$ et $\sigma_\theta(x)$.

On obtient alors : 
$$
KL\big(q_\theta(z\vert x) \parallel p(z)\big) = \frac{1}{2} \sum_{i=1}^\ell \left(\sigma_\theta(x)_i^2 + \mu_\theta(x)_i^2 - 1 - \log (\sigma_\theta(x)_i^2)\right)
$$

Ce terme s'évalue en **une seule passe** à partir des sorties de l'encodeur, il est différentiable par rapport à $\theta$, et ne nécessite aucune estimation Monte-Carlo.


#### Terme de reconstruction

La densité de probabilité d'une loi normale multivariée $p(x) = \mathcal{N}\left(\mu, \Sigma\right)$, avec $u \in \mathbb{R}^N$ et $\Sigma \in \mathbb{R}^{N \times N}$[^fn:defN] est donnée par :
```{math}
:label: eq:dens_proba_normale
p(x) = \frac{1}{(2\pi)^{\frac{N}{2}} \det(\Sigma)^{\frac{1}{2}}} \exp\left(-\frac{1}{2} (x-\mu)^T \Sigma^{-1}(x - \mu)\right)
```

[^fn:defN]: $N$ correspond à la dimension de $\mathcal{X}$ aplati. Si l'entrée est une image couleur $64 \times 64$ pixels, alors $N = 3 \times 64 \times 64$.


Comme la distribution $p_\phi(x \vert z)$ est une loi normale $\mathcal{N}\left(\mu_\phi, \sigma_{dec}^2 I\right)$, on peut simplifier le terme de reconstruction car :
* $\log(\frac{b}{a}) = \log(b) - \log(a)$,
* $\det (\Sigma) = (\sigma_{dec}^2)^N$,
* $\Sigma^{-1} = \frac{1}{\sigma_{dec}^2}I_N$.

Le terme de reconstruction devient : 
$$
\log p_\phi(x\vert z) = -\frac{N}{2}\log(2\pi \sigma_{dec}^2) - \frac{\|x - \mu_\phi(z)\|^2}{2\sigma_{dec}^2}
$$




Maximiser cette vraisemblance revient, à constantes près, à **minimiser l'erreur quadratique moyenne (MSE)** $\|x - \mu_\phi(z)\|^2$ entre l'entrée $x$ et sa reconstruction $\mu_\phi(z)$. 

```{important}
La fonction de loss cherche à maximiser le terme de reconstruction tout en minimisant la régularisation KL. Comme le terme de reconstruction est équivalent (à constante près) à l'erreur quadratique moyenne, on utilisera comme loss une somme pondérée de l'erreur quadratique moyenne et de l'opposé de la divergence KL. La loss est alors égale à : 
$$
\|x - \mu_\phi(z)\|^2 - \beta \left(\frac{1}{2} \sum_{i=1}^\ell \left(\sigma_\theta(x)_i^2 + \mu_\theta(x)_i^2 - 1 - \log (\sigma_\theta(x)_i^2)\right)\right)
$$
où la pondération $\beta$ est un hyperparamètre.
```



### L'astuce de reparamétrisation

Un problème subsiste : pour optimiser $\theta$ par descente de gradient, il faut rétropropager le gradient à travers l'échantillonnage $z \sim q_\theta(z \vert x)$. Or, l'opération d'échantillonnage n'est pas différentiable par rapport à $\theta$.

L'*astuce de reparamétrisation* (*reparameterization trick*) contourne ce problème en réécrivant $z$ comme une fonction déterministe de $\theta$ et d'une variable aléatoire indépendante :
$$
z = \mu_\theta(x) + \sigma_\theta(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I_\ell)
$$
Le caractère aléatoire est ainsi déplacé entièrement sur $\epsilon$, qui ne dépend pas de $\theta$. Le gradient peut alors se propager à  travers $\mu_\theta(x)$ et $\sigma_\theta(x)$, et l'entraînement peut se faire par rétropropagation classique.


## Génération d'images

Une fois l'autoencodeur entraîné, il est possible de générer de nouvelles images. Pour cela, on sample une représentation latente $z$ selon $\mathcal{N}(0, I_\ell)$. Cette représentation latente est utilisé comme entrée pour le décodeur qui prédit une image. L'encodeur n'est pas utilisé pour la génération.

 




