# Système de Recommandation basé sur les Embeddings de Modèles de Langage

Projet de système de recommandation utilisant les embeddings de modèles de langage (Sentence-BERT) combinés à un **mécanisme d'attention** et un **apprentissage contrastif (BPR Loss)** pour améliorer les recommandations.

## Description du Projet

Ce projet implémente un système de recommandation innovant qui combine:
- **Embeddings de modèles de langage** (Sentence-BERT `all-MiniLM-L6-v2`) pour capturer la sémantique des items
- **Mécanisme d'Attention learnable**  pour pondérer l'importance des items dans le profil utilisateur
- **Apprentissage contrastif** avec BPR Loss pour optimiser le ranking
- **Comparaison avec Matrix Factorization** comme baseline

## Résultats

| Modèle | Recall@10 | NDCG@10 | Amélioration |
|--------|-----------|---------|--------------|
| Matrix Factorization | 0.0141 | 0.0438 | - |
| **ContrastiveAttention** | **0.0377** | **0.0960** | **+159% / +106%** |

### Architecture du Modèle Principal

#### `ContrastiveAttentionRecommender` 
1. **Encode les textes** (titre + genres) avec Sentence-BERT → 384 dimensions
2. **Mécanisme d'attention multi-head** pour pondérer les items du profil utilisateur
3. **Apprentissage par BPR Loss** : maximise le score des items positifs vs négatifs
4. **Negative Sampling** : 10 négatifs par positif avec pondération inverse à la popularité
5. **Interprétable** : `explain_recommendation()` pour visualiser les poids d'attention

## Objectifs

- Développer un système de recommandation basé sur les embeddings de modèles de langage
- Utiliser un mécanisme d'attention pour apprendre les poids des items
- Utiliser des données MovieLens 1M avec métadonnées textuelles (titre + genres)
- Comparer avec Matrix Factorization comme baseline
- Évaluer avec **Recall@10** et **NDCG@10**

## Dataset

- **Source**: MovieLens 1M (téléchargé automatiquement via Cornac)
- **Contenu**: ~1M ratings, 6040 utilisateurs, 3675 items avec texte
- **Texte**: Titre du film + genres (ex: "Toy Story (1995) | Animation | Children's | Comedy")
- **Split**: 80% train / 20% test (ratio-split)
- **Threshold**: Interactions positives si rating ≥ 3.5

## Architecture

```
système de recom/
├── models/                                # Modèles de recommandation
│   ├── __init__.py
│   ├── embedding_recommender.py           # Modèle baseline par embeddings
│   └── attention_recommender_v3.py        # Modèle principal avec attention + BPR
├── results/                               # Résultats d'évaluation (JSON)
├── data_loader.py                         # Chargement MovieLens 1M
├── evaluate.py                            # Script d'évaluation et comparaison
├── requirements.txt                       # Dépendances Python
└── README.md                              # Ce fichier
```

## Installation

### 1. Cloner le projet

```bash
git clone git@github.com:mohammedAtif09-ia/systeme-de-recommandation.git
cd systeme-de-recommandation
```

### 2. Créer un environnement virtuel (recommandé)

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

**Dépendances principales:**
- `cornac>=2.0.0` - Framework de recommandation
- `sentence-transformers>=2.2.2` - Embeddings BERT
- `torch>=2.0.0` - Backend PyTorch
- `numpy`, `pandas`, `scikit-learn` - Outils data science

## Utilisation

### Exécuter l'évaluation complète

```bash
python evaluate.py
```

Cette commande va:
1. Charger les données MovieLens 1M (~1M ratings)
2. Préparer le split train/test (80/20)
3. Entraîner 2 modèles:
   - Matrix Factorization (baseline)
   - ContrastiveAttentionRecommender (notre solution)
4. Évaluer avec Recall@10, NDCG@10, Precision@10
5. Afficher les résultats comparatifs
6. Sauvegarder les résultats dans `results/`

### Hyperparamètres du modèle

```python
ContrastiveAttentionRecommender(
    num_epochs=40,           # Epochs d'entraînement
    learning_rate=0.002,     # Learning rate Adam
    batch_size=512,          # Batch size
    num_negatives=10,        # Négatifs par positif
    max_history_items=30,    # Items max par profil
    embedding_dim=384,       # Dimension Sentence-BERT
    attention_heads=4,       # Têtes d'attention
)
```

## Métriques d'Évaluation

### Recall@10
Proportion des items pertinents (aimés par l'utilisateur) retrouvés dans le top-10 des recommandations.

$$\text{Recall@10} = \frac{\text{Nombre d'items pertinents dans top-10}}{\text{Nombre total d'items pertinents}}$$

### NDCG@10 (Normalized Discounted Cumulative Gain)
Mesure la qualité du classement en pénalisant les items pertinents mal classés.

$$\text{NDCG@k} = \frac{DCG@k}{IDCG@k}$$

où $DCG@k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i+1)}$

## Modèles Comparés

### 1. Matrix Factorization (Baseline)
- Modèle baseline classique de filtrage collaboratif
- Factorise la matrice user-item en matrices latentes
- Paramètres: k=50, learning_rate=0.01, max_iter=100

### 2. ContrastiveAttentionRecommender (Notre solution)
- **Sentence-BERT** (`all-MiniLM-L6-v2`) pour encoder les textes
- **Attention multi-head** pour pondérer les items du profil utilisateur
- **BPR Loss** + Margin Loss pour l'apprentissage contrastif
- **Negative Sampling** pondéré par l'inverse de la popularité
- **LR Scheduler** (StepLR) pour stabiliser l'entraînement
- Supporte GPU si disponible (détection automatique)

## Avantages de l'Approche

1. **Cold Start**: Peut recommander de nouveaux items avec texte (titre + genres)
2. **Interprétabilité**: `explain_recommendation()` montre quels films ont influencé la recommandation
3. **Performance**: +159% Recall@10 vs Matrix Factorization
4. **Apprentissage Contrastif**: BPR Loss optimise directement le ranking
5. **Attention Learnable**: Le modèle apprend quels items comptent le plus

## Exemple de Sortie

```
================================================================================
RÉSULTATS COMPARATIFS - MovieLens 1M
================================================================================

Modèle                          Recall@10    NDCG@10      Precision@10
--------------------------------------------------------------------------------
MatrixFactorization             0.0141       0.0438       0.0422
ContrastiveAttention            0.0377       0.0960       0.0826

================================================================================
 Meilleur modèle: ContrastiveAttention
   Recall@10: +159% vs MF | NDCG@10: +106% vs MF
================================================================================

Training Progress:
Epoch 1/40, Loss: 0.6749
Epoch 10/40, Loss: 0.6621
Epoch 20/40, Loss: 0.6545
Epoch 40/40, Loss: 0.6449
```

##  Configuration Avancée

### Changer de modèle d'embeddings

Dans [models/attention_recommender_v3.py](models/attention_recommender_v3.py), modifiez `model_name`:

```python
model = ContrastiveAttentionRecommender(
    model_name='sentence-transformers/all-mpnet-base-v2',  # Modèle plus puissant (768 dim)
    # ou
    model_name='all-MiniLM-L6-v2',  # Par défaut (384 dim, plus rapide)
)
```

### Ajuster les hyperparamètres

Dans [evaluate.py](evaluate.py):

```python
ContrastiveAttentionRecommender(
    num_epochs=60,          # Plus d'epochs
    learning_rate=0.001,    # LR plus faible
    num_negatives=15,       # Plus de négatifs
    batch_size=256,         # Batch plus petit
)
```

### Utiliser le GPU

Le modèle détecte automatiquement CUDA. Pour forcer le CPU:

```bash
export CUDA_VISIBLE_DEVICES=""
python evaluate.py
```

## Fichiers Principaux

| Fichier | Description |
|---------|-------------|
| [data_loader.py](data_loader.py) | Chargement MovieLens 1M, lecture directe des fichiers |
| [models/attention_recommender_v3.py](models/attention_recommender_v3.py) | Modèle principal avec attention + BPR Loss |
| [models/embedding_recommender.py](models/embedding_recommender.py) | Modèle baseline par embeddings simples |
| [evaluate.py](evaluate.py) | Script d'évaluation et comparaison |
| [requirements.txt](requirements.txt) | Dépendances Python |

##  Dépannage

### Erreur CUDA / GPU
Le modèle détecte automatiquement CUDA. Pour forcer CPU:
```bash
export CUDA_VISIBLE_DEVICES=""
python evaluate.py
```

### Manque de mémoire
Réduisez le batch size dans `evaluate.py`:
```python
ContrastiveAttentionRecommender(batch_size=128)
```

### Téléchargement des données échoue
Les données MovieLens sont téléchargées dans `~/.cornac/ml-1m/`. Vérifiez votre connexion.

##  Références

- [Cornac Framework](https://cornac.readthedocs.io/) - Framework de recommandation
- [Sentence-Transformers](https://www.sbert.net/) - Embeddings BERT
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/) - Dataset de ratings
- Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback" (2009)
- Vaswani et al., "Attention Is All You Need" (2017)

##  Auteur

Mohammed Atif - Projet de système de recommandation basé sur les embeddings de modèles de langage.

##  License

Ce projet est à usage éducatif.

---

**Note**: L'évaluation complète prend environ 15-20 minutes sur CPU (génération des embeddings + entraînement 40 epochs).
