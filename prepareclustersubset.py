import pandas as pd
from pathlib import Path

# --- Fichiers source ---
clusters_path = Path("data/cluster_reps_checkpoint_final_20251106_163826.parquet")
jobs_path = Path("data/sampled_engineers_with_clusters_20251105_175242.parquet")

# --- Chargement ---
df_clusters = pd.read_parquet(clusters_path)
df_jobs = pd.read_parquet(jobs_path)

# --- Sélection de 10 clusters aléatoires ---
selected_clusters = df_clusters["cluster_id"].dropna().sample(30, random_state=42).tolist()
print("Clusters sélectionnés :", selected_clusters)

# --- CV correspondants (dans le petit fichier) ---
cv_subset = df_clusters[df_clusters["cluster_id"].isin(selected_clusters)][["cluster_id", "cv_standard"]]

# --- Job descriptions associées (dans le grand fichier) ---
jobs_subset = df_jobs[df_jobs["cluster_label"].isin(selected_clusters)][["cluster_label", "description"]]
jobs_subset = jobs_subset.rename(columns={"cluster_label": "cluster_id"})

# --- Sauvegarde du sous-ensemble pour embedding ---
output_dir = Path("data/subset")
output_dir.mkdir(exist_ok=True)

cv_subset.to_parquet(output_dir / "selected_cvs.parquet", index=False)
jobs_subset.to_parquet(output_dir / "selected_job_descriptions.parquet", index=False)

print(f"{len(cv_subset)} CV sauvegardés")
print(f"{len(jobs_subset)} descriptions sauvegardées")
