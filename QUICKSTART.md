# Guide de Démarrage Rapide - Système de Recommandation

## Installation (5 minutes)

```bash
# 1. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou venv\Scripts\activate sur Windows

# 2. Installer les dépendances
pip install -r requirements.txt
```

## Utilisation Rapide

### Option 1: Évaluation Complète (recommandé)
```bash
python evaluate.py
```
- Compare 4 modèles (MF, BPR, EmbeddingRecommender, HybridEmbedding)
- Affiche Recall@10, NDCG@10, Precision@10
- Sauvegarde les résultats dans `results/`
- Durée: ~5-10 minutes

### Option 2: Démonstration Interactive
```bash
python demo.py
```
- Affiche des recommandations pour 3 utilisateurs exemples
- Montre les films vus et les recommandations
- Durée: ~2-3 minutes

### Option 3: Visualisation des Résultats
```bash
python visualize_results.py
```
- Génère des graphiques comparatifs
- Crée un graphique radar
- Nécessite d'avoir exécuté `evaluate.py` d'abord

### Option 4: Exploration Interactive (Jupyter)
```bash
jupyter notebook exploration.ipynb
```
- Notebook interactif avec analyses et visualisations
- Idéal pour comprendre le système en détail

## Structure des Fichiers

```
système de recom/
├── evaluate.py                    # ⭐ Script principal d'évaluation
├── demo.py                        # Démonstration simple
├── visualize_results.py          # Génération de graphiques
├── exploration.ipynb             # Notebook Jupyter
├── data_loader.py                # Chargement des données
├── models/
│   ├── __init__.py
│   └── embedding_recommender.py  # Modèles basés sur embeddings
├── requirements.txt              # Dépendances
├── README.md                     # Documentation complète
└── QUICKSTART.md                 # Ce fichier
```

## Commandes Importantes

```bash
# Lancer l'évaluation complète
python evaluate.py

# Test du chargement des données
python data_loader.py

# Démonstration rapide
python demo.py

# Visualiser les résultats
python visualize_results.py

# Jupyter notebook
jupyter notebook exploration.ipynb
```

## Résultats Attendus

Après `python evaluate.py`, vous obtiendrez:

```
Modèle                    Recall@10    NDCG@10      Precision@10    Temps (s)   
--------------------------------------------------------------------------------
MatrixFactorization       0.12XX       0.09XX       0.04XX          ~10-15s     
BPR                       0.13XX       0.10XX       0.04XX          ~15-20s     
EmbeddingRecommender      0.14XX       0.11XX       0.05XX          ~40-50s     
HybridEmbedding           0.15XX       0.11XX       0.05XX          ~45-55s     
```

## Personnalisation

### Changer de modèle BERT
Dans `models/embedding_recommender.py`, ligne 32:
```python
model_name='all-mpnet-base-v2'  # Plus puissant
# ou
model_name='all-MiniLM-L6-v2'   # Plus rapide (défaut)
```

### Ajuster le split train/test
Dans `evaluate.py`, ligne 159:
```python
prepare_cornac_dataset(ratings, item_texts, test_size=0.3)  # 70/30 au lieu de 80/20
```

### Modifier le ratio hybride
Dans `evaluate.py`, ligne 100:
```python
HybridEmbeddingRecommender(alpha=0.5)  # 50/50 au lieu de 70/30
```

## Dépannage

### Erreur GPU/CUDA
```bash
export CUDA_VISIBLE_DEVICES=""  # Forcer CPU
python evaluate.py
```

### Manque de mémoire
Réduire le batch_size dans `embedding_recommender.py` ligne 97:
```python
batch_size=16  # Au lieu de 32
```

### Téléchargement lent
Le premier lancement télécharge:
- Données MovieLens (~5MB)
- Modèle BERT (~80MB)
- Peut prendre quelques minutes

## Pour Aller Plus Loin

- Lire le [README.md](README.md) complet pour plus de détails
- Explorer le [notebook Jupyter](exploration.ipynb) pour comprendre les embeddings
- Modifier [evaluate.py](evaluate.py) pour ajouter vos propres modèles
- Consulter la [documentation Cornac](https://cornac.readthedocs.io/)

## Support

En cas de problème:
1. Vérifier que toutes les dépendances sont installées: `pip list`
2. Vérifier la version de Python: `python --version` (≥3.7 requis)
3. Consulter les logs d'erreur détaillés

---

**Temps total estimé pour compléter le projet: 10-15 minutes**
