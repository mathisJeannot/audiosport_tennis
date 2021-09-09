# audiosport_tennis
Descrition des différents fichiers python

## algos_nmfs.py
Ce fichier contient les fonctions qui prennent en entrer un spectrogrammes et les différents paramètres d'une factorisation matricielle (NMF, CNMF, SNMF) et renvoie la matrice dictionnaire (W) et la matrice d'activation (H). Les algorithmes des fonctions "NMF" et "CNMF" sont en fait implémenté dans les fichiers nmf_from_sklearn.py et convolutive_MM.py.

## beta_divergence.py
Contient une unique fonction qui calcul la divergence entre deux tableaux (x et y). On peut choisir le paramètre beta de la divergence.

## convertion.py
contient différentes fonction pour faire des conversions qui m'ont été utiles. Notemment la conversion de secondes en colonnes de du spectrogramme.

## convolutive_MM.py
Algorithme de calcul de CNMF. La fonction "convolutive_MM" est appelée dans la fonction "CNMF" de algos_nmfs.py

## metrique_nmf.py
Fonction utiles pour le calcul de score: score d'une ligne de H, métrique par segment, calcul du meilleur seuil, affichage des résultats. 
La fonctionnalité de chacune d'elle est expliquée en commentaire dans le code.

## nmf_from_sklearn
Code issu de scikit learn permettant de caluler une NMF. Le code y est ici modifié pour permettre de faire des NMF semi-supervisées. La fonction "\_fit_multiplicative_update" est appelée dans la fonction "NMF" de algos_nmfs.py

## plot_nmfs.py
Contient la fonction qui affiches les tableaux en lien avec la factorisation (Spectrogramme, W, H et WH).
Je ne suis plus sur que l'affichage fonctionne vraiment bien

## regression_logistique.py
contient les fonctions qui exécutent des régressions logistiques avec différents descripteurs selon la fonction.
y_pred_proba prend simplement chaque colonne commme descripteur.
y_pred_proba_overlap utilise comme descripteur la concaténation de la colonne de l'instant choisi et les colonnes environnantes
y_pred_proba_overlap_mean_std utilise le même principe que "y_pred_proba_overlap", mais utilise comme descripteurs simplement la moyenne et l'ecartypes des colonnes environnantes. Cela permet d'avoir moins de paramètre à détérminer pour la régression, donc une complexité plus faible, tout en gardant des bon résultats.
