"""
Projet : Analyse COVID-19 en France par département
Outils : Python (Pandas, GeoPandas, Matplotlib, Seaborn)
Export  : GeoJSON pour QGIS
Auteur  : Berramdani Aymen
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import json
import os


# CONFIG

DATA_CSV     = "data/covid_france_brut.csv"
DATA_GEO     = "data/departements.geojson"
OUTPUT_DIR   = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 55)
print("  ANALYSE COVID-19 — FRANCE PAR DÉPARTEMENT")
print("=" * 55)


# 1. CHARGEMENT & NETTOYAGE

print("\n[1/5] Chargement des données...")
df = pd.read_csv(DATA_CSV, sep=",", low_memory=False)
df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")

# Filtrer uniquement les départements
dept = df[df["granularite"] == "departement"].copy()
dept["code_dept"] = dept["maille_code"].str.replace("DEP-", "")

# Colonnes utiles
cols = ["date", "code_dept", "maille_nom", "cas_confirmes", "deces",
        "hospitalises", "reanimation", "gueris"]
dept = dept[cols].copy()

# Convertir en numérique
for col in ["cas_confirmes", "deces", "hospitalises", "reanimation", "gueris"]:
    dept[col] = pd.to_numeric(dept[col], errors="coerce")

print(f"   ✓ {len(dept):,} lignes chargées")
print(f"   ✓ Période : {dept['date'].min().date()} → {dept['date'].max().date()}")
print(f"   ✓ {dept['code_dept'].nunique()} départements")


# 2. AGRÉGATION PAR DÉPARTEMENT

print("\n[2/5] Agrégation par département...")

summary = dept.groupby(["code_dept", "maille_nom"]).agg(
    cas_max        = ("cas_confirmes", "max"),
    deces_max      = ("deces", "max"),
    hospit_max     = ("hospitalises", "max"),
    rea_max        = ("reanimation", "max"),
    gueris_max     = ("gueris", "max")
).reset_index()

summary["taux_mortalite"] = (
    summary["deces_max"] / summary["cas_max"] * 100
).round(2)

print(f"   ✓ {len(summary)} départements agrégés")

# Top 5 cas confirmés
top5 = summary.nlargest(5, "cas_max")[["maille_nom", "cas_max", "deces_max"]]
print("\n   Top 5 départements (cas confirmés) :")
for _, r in top5.iterrows():
    print(f"   → {r['maille_nom']:<30} {r['cas_max']:>10,.0f} cas  |  {r['deces_max']:>6,.0f} décès")


# 3. ANALYSE TEMPORELLE NATIONALE

print("\n[3/5] Analyse temporelle...")

national = df[df["granularite"] == "pays"].copy()
national["date"] = pd.to_datetime(national["date"])
for col in ["cas_confirmes", "deces", "hospitalises", "reanimation"]:
    national[col] = pd.to_numeric(national[col], errors="coerce")
national = national.sort_values("date")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Analyse COVID-19 en France — Vue nationale", 
             fontsize=16, fontweight="bold", y=1.01)

palette = ["#2563eb", "#dc2626", "#16a34a", "#d97706"]

# Cas confirmés
axes[0,0].plot(national["date"], national["cas_confirmes"], 
               color=palette[0], linewidth=2)
axes[0,0].fill_between(national["date"], national["cas_confirmes"], 
                        alpha=0.1, color=palette[0])
axes[0,0].set_title("Cas confirmés cumulés", fontweight="bold")
axes[0,0].set_ylabel("Nombre")
axes[0,0].grid(True, alpha=0.3)
axes[0,0].tick_params(axis='x', rotation=45)

# Décès
axes[0,1].plot(national["date"], national["deces"], 
               color=palette[1], linewidth=2)
axes[0,1].fill_between(national["date"], national["deces"], 
                        alpha=0.1, color=palette[1])
axes[0,1].set_title("Décès cumulés", fontweight="bold")
axes[0,1].set_ylabel("Nombre")
axes[0,1].grid(True, alpha=0.3)
axes[0,1].tick_params(axis='x', rotation=45)

# Hospitalisations
axes[1,0].plot(national["date"], national["hospitalises"], 
               color=palette[2], linewidth=2)
axes[1,0].fill_between(national["date"], national["hospitalises"], 
                        alpha=0.1, color=palette[2])
axes[1,0].set_title("Hospitalisations actives", fontweight="bold")
axes[1,0].set_ylabel("Nombre")
axes[1,0].grid(True, alpha=0.3)
axes[1,0].tick_params(axis='x', rotation=45)

# Réanimation
axes[1,1].plot(national["date"], national["reanimation"], 
               color=palette[3], linewidth=2)
axes[1,1].fill_between(national["date"], national["reanimation"], 
                        alpha=0.1, color=palette[3])
axes[1,1].set_title("Patients en réanimation", fontweight="bold")
axes[1,1].set_ylabel("Nombre")
axes[1,1].grid(True, alpha=0.3)
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_analyse_temporelle.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   ✓ Graphique temporel sauvegardé")


# 4. CARTE CHOROPLÈTHE

print("\n[4/5] Création de la carte choroplèthe...")

gdf = gpd.read_file(DATA_GEO)
gdf = gdf.rename(columns={"code": "code_dept", "nom": "nom_dept"})
gdf_merged = gdf.merge(summary, on="code_dept", how="left")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("COVID-19 en France — Cartographie par département",
             fontsize=15, fontweight="bold")

# Carte 1 : Cas confirmés
gdf_merged.plot(
    column="cas_max",
    ax=axes[0],
    cmap="YlOrRd",
    legend=True,
    legend_kwds={"label": "Cas confirmés", "shrink": 0.6},
    missing_kwds={"color": "lightgrey"},
    edgecolor="white",
    linewidth=0.5
)
axes[0].set_title("Cas confirmés (max)", fontweight="bold", pad=10)
axes[0].axis("off")

# Carte 2 : Taux de mortalité
gdf_merged.plot(
    column="taux_mortalite",
    ax=axes[1],
    cmap="Reds",
    legend=True,
    legend_kwds={"label": "Taux mortalité (%)", "shrink": 0.6},
    missing_kwds={"color": "lightgrey"},
    edgecolor="white",
    linewidth=0.5
)
axes[1].set_title("Taux de mortalité (%)", fontweight="bold", pad=10)
axes[1].axis("off")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_carte_choroplethe.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   ✓ Carte choroplèthe sauvegardée")


# 5. EXPORT GEOJSON POUR QGIS

print("\n[5/5] Export GeoJSON pour QGIS...")

geojson_path = f"{OUTPUT_DIR}/covid_departements_qgis.geojson"
gdf_merged.to_file(geojson_path, driver="GeoJSON")

print(f"   ✓ GeoJSON exporté → {geojson_path}")
print(f"   ✓ Colonnes disponibles dans QGIS : cas_max, deces_max, taux_mortalite...")

# Export CSV aussi
csv_path = f"{OUTPUT_DIR}/covid_summary.csv"
summary.to_csv(csv_path, index=False)
print(f"   ✓ CSV résumé exporté → {csv_path}")


# RÉSUMÉ FINAL

print("\n" + "=" * 55)
print("  RÉSUMÉ DES RÉSULTATS")
print("=" * 55)
total_cas    = summary["cas_max"].max()
total_deces  = summary["deces_max"].max()
dept_plus    = summary.loc[summary["cas_max"].idxmax(), "maille_nom"]
taux_moy     = summary["taux_mortalite"].mean()

print(f"  Département le plus touché : {dept_plus}")
print(f"  Maximum cas confirmés      : {total_cas:,.0f}")
print(f"  Maximum décès              : {total_deces:,.0f}")
print(f"  Taux mortalité moyen       : {taux_moy:.2f}%")
print("\n  Fichiers générés :")
print("  → output/01_analyse_temporelle.png")
print("  → output/02_carte_choroplethe.png")
print("  → output/covid_departements_qgis.geojson  ← OUVRIR DANS QGIS")
print("  → output/covid_summary.csv")
print("=" * 55)
print("\n  Projet terminé avec succès !")
