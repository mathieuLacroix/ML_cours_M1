(divergenceKL)=
# Divergence de Kullback-Liebler

La divergence de Kullback-Liebler (divergence KL ou entropie (croisée) relative) permet de mesurer la dissimilarité entre deux distributions de probabilités. 

La divergence KL entre deux distributions $p$ et $q$ est notée $KL(p\parallel q)$. Plus cette valeur est proche de zéro, plus les deux distributions de probabilités se ressemblent.

On considère ici le cas de distributions de probabilités discrètes sur un ensemble $\mathcal{X}$.

## Surprise

La surprise qu'un événement arrive est inversement proportionnel à sa probabilité. Autrement dit, plus un événement est rare, plus sa réalisation surprend. Par exemple, il est plus surprenant de réussir à prédire les numéros du loto que de réussir à prédire si une pièce retombe sur pile ou face.

On peut définir mathématiquement la surprise $s(x)$ qu'un événement $x \in X$ survienne par :
$$
S(x) = - \log(p(x))
$$

**Remarques :**
* Plus un événement est sûr, plus faible est sa surprise.
* Si un événement est certain ($p(x) = 1$), alors il n'y a aucune surprise liée à sa réalisation ($S(x) = 0$). 
* La surprise liée à la réalisation de deux événements correspond à la somme des deux surprises.

## Entropie de Shannon

L'entropie de Shannon, noté $H(p)$, permet de définir la surprise attendue d'une distribution de probabilités $p$. Elle correspond à l'espérance des surprises, c'est-à-dire à la somme des surprises de la réalisation de chacun des événements possibles pondérées par leur probabilités :
$$
H(p) = -\sum_{x \in \mathcal{X}} p(x) log(p(x))
$$

**Remarques :**
* L'entropie $H$ est nulle uniquement s'il n'y a pas d'aléatoire, c'est-à-dire qu'il existe $\tilde{x} \in X$ tel que $p(\tilde{x}) = 1$.
* L'entropie de Shannon est une valeur positive ou nulle.


## Entropie croisée

C'est la surprise attendue si on approxime $p$ avec une distribution $q$. 
$$
H(p,q) = \mathbb{E}_{x \sim p} - \log(q(x)) = - \sum_{x \in \mathcal{X}} p(x) \log(q(x))
$$

Remarques : 
* On a $H(P,Q) \ge H(P)$ (se déduit de la non-négativité de la divergence KL).
* $H(p,q) = H(p)$ si et seulement si $p$ et $q$ sont identiques.


## Divergence KL

La divergence de Kullback-Liebler "centre" l'entropie croisée sur 0 en soustrayant l'entropie : 
$$
KL(p \parallel q) = H(p,q) - H(p) = \sum_{x \in \mathcal{X}} p(x) \log\left(\frac{p(x)}{q(x)}\right)
$$

Ainsi, si $p$ et $q$ sont identiques, alors la divergence est nulle.

La divergence KL est aussi appelée *entropie (croisée) relative* et correspond à l'augmentation de la surprise attendue lorsque l'on approxime $p$ avec $q$.

**Remarques :**
* $KL(p \parallel q) = 0$ si et seulement si $p$ et $q$ sont identiques (car $log (1) = 0$).
* $KL(p \parallel q) \ge 0$ (Pour une preuve, voir par exemple @prince2023understanding page 461).
* Attention, la divergence n'est pas symétrique ! Autrement dit, $KL(p \parallel q)  \neq KL(q \parallel p)$.
* Si $p(x) > 0$ et $q(x) = 0$ pour un événement $x \in \mathcal{X}$, alors la divergence tend vers l'infini ! Il faut dans ce cas utiliser la divergence de [Jensen-Shannon](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence). 


## Divergence KL pour des lois continues

Si $p$ et $q$ sont des lois continues, alors la divergence KL s'écrit :

$$
KL(p \parallel q) = \int_{x} p(x) \log\left(\frac{p(x)}{q(x)}\right)
$$

## Divergence KL de deux lois normales

Si $p$ et $q$ sont deux lois normales de dimension $k$, c'est-à-dire $p \sim \mathcal{N}(\mu_p, \Sigma_p)$ et $q \sim \mathcal{N}(\mu_q, \Sigma_q)$, alors la divergence KL est égale à :
$$
KL(p \parallel q) = \frac{1}{2}\left[\log\left(\frac{\det(\Sigma_q)}{\det(\Sigma_p)}\right) - k  + (\mu_p - \mu_q)^T \Sigma_q^{-1} (\mu_p - \mu_q)  + tr (\Sigma_q^{-1}\Sigma_p)\right]
$$

Si $q$ est une loi normale standard, c'est-à-dire, $q \sim \mathcal{N}(0,I_k)$, alors la divergence se simplifie selon :
$$
KL(p \parallel q) = \frac{1}{2}\left[\mu_p^T\mu_p + tr(\Sigma_p) - k -\log\left(\det(\Sigma_p)\right) \right]
$$

Une preuve de ces deux résultats est donnée [ici](https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/).