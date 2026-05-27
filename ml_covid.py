"""
Projet : Machine Learning sur données COVID-19
Auteur : Berramdani Aymen

Modèles utilisés :
  1. KMeans Clustering       → regrouper les départements par niveau de risque
  2. Régression (RF + Ridge) → prédire les décès
  3. Export GeoJSON          → visualiser les clusters dans QGIS
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# CONFIG

DATA_CSV   = "output/covid_summary.csv"
DATA_GEO   = "output/covid_departements_qgis.geojson"
OUTPUT_DIR = "output"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 55)
print("  ML COVID-19 — FRANCE PAR DÉPARTEMENT")
print("=" * 55)


# CHARGEMENT

df = pd.read_csv(DATA_CSV)
df = df.dropna()
print(f"\n✓ {len(df)} départements chargés")

features = ["cas_max", "hospit_max", "rea_max", "gueris_max"]
target   = "deces_max"

X = df[features]
y = df[target]


# 1. CLUSTERING KMEANS

print("\n[1/3] KMeans Clustering...")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Trouver le bon nombre de clusters (méthode du coude)
inertias = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Appliquer KMeans avec 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

# Nommer les clusters selon le niveau de cas
cluster_means = df.groupby("cluster")["cas_max"].mean().sort_values()
cluster_labels = {
    cluster_means.index[0]: "Zone verte  (peu touché)",
    cluster_means.index[1]: "Zone orange (moyennement touché)",
    cluster_means.index[2]: "Zone rouge  (très touché)"
}
df["zone_risque"] = df["cluster"].map(cluster_labels)

print("\n  Résultats par zone :")
print(f"  {'Zone':<35} {'Depts':>5} {'Cas moy':>10} {'Décès moy':>10}")
print("  " + "-" * 62)
for zone, group in df.groupby("zone_risque"):
    print(f"  {zone:<35} {len(group):>5} {group['cas_max'].mean():>10.0f} {group['deces_max'].mean():>10.0f}")


# 2. RÉGRESSION — PRÉDIRE LES DÉCÈS

print("\n[2/3] Régression — Prédiction des décès...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

mae_rf    = mean_absolute_error(y_test, y_pred_rf)
r2_rf     = r2_score(y_test, y_pred_rf)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
r2_ridge  = r2_score(y_test, y_pred_ridge)

print(f"\n  {'Modèle':<20} {'MAE':>10} {'R²':>10}")
print("  " + "-" * 42)
print(f"  {'Random Forest':<20} {mae_rf:>10.0f} {r2_rf:>10.3f}")
print(f"  {'Ridge':<20} {mae_ridge:>10.0f} {r2_ridge:>10.3f}")

# Importance des variables
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print(f"\n  Variables les plus importantes (Random Forest) :")
for feat, imp in importances.items():
    bar = "█" * int(imp * 30)
    print(f"  {feat:<15} {bar} {imp:.3f}")


# 3. VISUALISATIONS

print("\n[3/3] Génération des graphiques et cartes...")

colors_zone = {
    "Zone verte  (peu touché)"       : "#22c55e",
    "Zone orange (moyennement touché)": "#f97316",
    "Zone rouge  (très touché)"       : "#ef4444"
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("ML COVID-19 — Analyse par département", fontsize=15, fontweight="bold")

# Graphique 1 — Méthode du coude
axes[0,0].plot(range(2, 8), inertias, "bo-", linewidth=2, markersize=8)
axes[0,0].axvline(x=3, color="red", linestyle="--", alpha=0.7, label="k=3 choisi")
axes[0,0].set_title("Méthode du coude — Choix du k", fontweight="bold")
axes[0,0].set_xlabel("Nombre de clusters (k)")
axes[0,0].set_ylabel("Inertie")
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Graphique 2 — Scatter clusters
for zone, group in df.groupby("zone_risque"):
    axes[0,1].scatter(
        group["cas_max"], group["deces_max"],
        c=colors_zone[zone], label=zone.split("(")[1].replace(")", "").strip(),
        s=60, alpha=0.8, edgecolors="white", linewidth=0.5
    )
axes[0,1].set_title("Clusters — Cas vs Décès", fontweight="bold")
axes[0,1].set_xlabel("Cas confirmés (max)")
axes[0,1].set_ylabel("Décès (max)")
axes[0,1].legend(fontsize=9)
axes[0,1].grid(True, alpha=0.3)

# Graphique 3 — Prédictions vs réel
axes[1,0].scatter(y_test, y_pred_rf, alpha=0.7, color="#3b82f6", edgecolors="white", s=60)
lim = max(y_test.max(), max(y_pred_rf))
axes[1,0].plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Prédiction parfaite")
axes[1,0].set_title(f"Random Forest — Prédictions vs Réel\nR²={r2_rf:.3f}  MAE={mae_rf:.0f}", fontweight="bold")
axes[1,0].set_xlabel("Valeurs réelles (décès)")
axes[1,0].set_ylabel("Valeurs prédites")
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Graphique 4 — Importance des variables
bars = axes[1,1].barh(importances.index, importances.values,
                       color=["#3b82f6","#60a5fa","#93c5fd","#bfdbfe"])
axes[1,1].set_title("Importance des variables\n(Random Forest)", fontweight="bold")
axes[1,1].set_xlabel("Importance")
for bar, val in zip(bars, importances.values):
    axes[1,1].text(val + 0.005, bar.get_y() + bar.get_height()/2,
                   f"{val:.3f}", va="center", fontsize=10)
axes[1,1].grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_ml_analyse.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Graphiques ML sauvegardés")


# CARTE QGIS — CLUSTERS

gdf = gpd.read_file(DATA_GEO)
gdf = gdf.rename(columns={"code": "code_dept"})
df["code_dept"] = df["code_dept"].astype(str).str.zfill(2)
gdf_merged = gdf.merge(df[["code_dept", "zone_risque", "cluster", "cas_max", "deces_max"]], 
                        on="code_dept", how="left")

# Carte des clusters
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

color_map = {0: "#22c55e", 1: "#f97316", 2: "#ef4444"}
cluster_to_label = {v: k for k, v in {
    cluster_means.index[0]: 0,
    cluster_means.index[1]: 1,
    cluster_means.index[2]: 2
}.items()}

gdf_merged["color"] = gdf_merged["cluster"].map(
    lambda x: "#22c55e" if x == cluster_means.index[0]
    else "#f97316" if x == cluster_means.index[1]
    else "#ef4444" if x == cluster_means.index[2]
    else "#d1d5db"
)

gdf_merged.plot(color=gdf_merged["color"], ax=ax, edgecolor="white", linewidth=0.5)
ax.set_title("COVID-19 — Zones de risque par département\n(Clustering KMeans — 3 zones)", 
             fontsize=13, fontweight="bold", pad=15)
ax.axis("off")

patches = [
    mpatches.Patch(color="#22c55e", label="Zone verte — peu touché"),
    mpatches.Patch(color="#f97316", label="Zone orange — moyennement touché"),
    mpatches.Patch(color="#ef4444", label="Zone rouge — très touché"),
    mpatches.Patch(color="#d1d5db", label="Données manquantes"),
]
ax.legend(handles=patches, loc="lower left", fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_carte_clusters.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Carte des clusters sauvegardée")

# Export GeoJSON pour QGIS
gdf_merged.drop(columns=["color"], errors="ignore").to_file(
    f"{OUTPUT_DIR}/covid_ml_qgis.geojson", driver="GeoJSON"
)
print(f"  ✓ GeoJSON ML exporté → ouvrir dans QGIS")


# RÉSUMÉ

print("\n" + "=" * 55)
print("  RÉSUMÉ ML")
print("=" * 55)
print(f"  Meilleur modèle  : Random Forest (R²={r2_rf:.3f})")
print(f"  Variable clé     : {importances.index[0]}")
print(f"  Zones identifiées: 3 (verte / orange / rouge)")
print(f"\n  Fichiers générés :")
print(f"  → output/03_ml_analyse.png")
print(f"  → output/04_carte_clusters.png")
print(f"  → output/covid_ml_qgis.geojson  ← OUVRIR DANS QGIS")
print("=" * 55)
