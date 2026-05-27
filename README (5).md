# 🦠 Analyse COVID-19 en France par Département

**Auteur :** Berramdani Aymen  
**Outils :** Python · Pandas · GeoPandas · Scikit-learn · Matplotlib · Seaborn  
**Export :** GeoJSON compatible QGIS

---

## 📋 Description

Ce projet analyse les données COVID-19 en France à l'échelle départementale. Il comprend deux volets :

1. **Analyse statistique et cartographique** (`analyse_covid.py`) — nettoyage des données, agrégation par département, visualisation temporelle nationale et cartographie choroplèthe.
2. **Machine Learning** (`ml_covid.py`) — clustering KMeans pour identifier des zones de risque et régression pour prédire les décès.

Les résultats sont exportés en GeoJSON pour être directement exploitables dans QGIS.

---

## 📁 Structure du projet

```
covid/
├── data/
│   ├── covid_france_brut.csv          # Données brutes COVID (source nationale)
│   └── departements.geojson           # Contours géographiques des départements
│
├── output/                            # Fichiers générés (créés automatiquement)
│   ├── 01_analyse_temporelle.png      # Graphiques temporels nationaux
│   ├── 02_carte_choroplethe.png       # Cartes choroplèthes par département
│   ├── 03_ml_analyse.png             # Résultats ML (clusters + régression)
│   ├── 04_carte_clusters.png         # Carte des zones de risque
│   ├── covid_summary.csv             # Résumé agrégé par département
│   ├── covid_departements_qgis.geojson  # GeoJSON pour QGIS (analyse)
│   └── covid_ml_qgis.geojson         # GeoJSON pour QGIS (ML / clusters)
│
├── analyse_covid.py                   # Script principal d'analyse
└── ml_covid.py                        # Script Machine Learning
```

---

## ⚙️ Installation

### Prérequis

Python 3.8+ et les bibliothèques suivantes :

```bash
pip install pandas geopandas matplotlib seaborn scikit-learn numpy
```

> **Note :** GeoPandas peut nécessiter des dépendances système (`libgdal`, `libproj`). Sur Ubuntu/Debian :
> ```bash
> sudo apt-get install libgdal-dev
> ```
> Sur macOS avec Homebrew :
> ```bash
> brew install gdal
> ```

---

## 🚀 Utilisation

Les deux scripts doivent être exécutés **depuis le dossier `covid/`** (le répertoire racine du projet), car les chemins vers les données sont relatifs.

### Étape 1 — Analyse statistique et cartographique

```bash
cd covid
python analyse_covid.py
```

Ce script effectue dans l'ordre :
1. Chargement et nettoyage des données brutes
2. Filtrage sur la granularité `departement`
3. Agrégation par département (cas max, décès max, taux de mortalité)
4. Graphiques temporels nationaux (cas, décès, hospitalisations, réanimation)
5. Cartes choroplèthes (cas confirmés + taux de mortalité)
6. Export `covid_summary.csv` et `covid_departements_qgis.geojson`

### Étape 2 — Machine Learning

```bash
python ml_covid.py
```

> ⚠️ Ce script utilise les fichiers générés par `analyse_covid.py`. Il doit donc être lancé en second.

Ce script effectue :
1. **KMeans Clustering (k=3)** — classification des départements en zones verte / orange / rouge selon le niveau de risque
2. **Régression** — prédiction des décès via Random Forest et Ridge Regression
3. Génération des graphiques d'analyse ML et de la carte des clusters
4. Export `covid_ml_qgis.geojson`

---

## 📊 Résultats produits

### Graphiques

| Fichier | Contenu |
|---|---|
| `01_analyse_temporelle.png` | Évolution nationale : cas, décès, hospitalisations, réanimation |
| `02_carte_choroplethe.png` | Carte France : cas confirmés et taux de mortalité par département |
| `03_ml_analyse.png` | Méthode du coude, scatter clusters, prédictions RF, importance des variables |
| `04_carte_clusters.png` | Carte des zones de risque (vert / orange / rouge) |

### Fichiers de données

| Fichier | Contenu |
|---|---|
| `covid_summary.csv` | `code_dept`, `maille_nom`, `cas_max`, `deces_max`, `hospit_max`, `rea_max`, `gueris_max`, `taux_mortalite` |
| `covid_departements_qgis.geojson` | Géométries + données agrégées, ouvrir dans QGIS |
| `covid_ml_qgis.geojson` | Géométries + cluster + zone_risque, ouvrir dans QGIS |

---

## 🗺️ Utilisation dans QGIS

1. Ouvrir QGIS
2. Glisser-déposer `covid_departements_qgis.geojson` ou `covid_ml_qgis.geojson`
3. Styliser via **Propriétés de la couche → Symbologie** :
   - Pour les analyses : utiliser `cas_max` ou `taux_mortalite` en mode *Gradué*
   - Pour les clusters ML : utiliser `zone_risque` en mode *Catégorisé*

---

## 🤖 Modèles Machine Learning

| Modèle | Objectif | Variables d'entrée | Cible |
|---|---|---|---|
| KMeans (k=3) | Clustering départements | `cas_max`, `hospit_max`, `rea_max`, `gueris_max` | `zone_risque` |
| Random Forest | Régression | idem | `deces_max` |
| Ridge | Régression | idem | `deces_max` |

Les performances (MAE, R²) sont affichées dans la console à l'exécution.

---

## 📌 Notes

- Les données sont au format cumulatif (valeurs maximales sur toute la période).
- La période couverte et le nombre de départements sont affichés à l'exécution.
- Les départements sans données apparaissent en gris sur les cartes.
