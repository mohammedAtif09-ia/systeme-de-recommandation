# Documentation Technique - Système de Recommandation par Embeddings

## 🎯 Vue d'Ensemble

Ce projet implémente un système de recommandation innovant qui exploite les **embeddings de modèles de langage** (Language Model Embeddings) pour améliorer les recommandations traditionnelles de filtrage collaboratif.

## 🧠 Approche Proposée

### Modèle Principal: `EmbeddingBasedRecommender`

Notre solution repose sur trois piliers:

#### 1. Encodage des Items avec BERT

Nous utilisons **Sentence-BERT** (modèle `all-MiniLM-L6-v2`) pour transformer les descriptions textuelles des items en vecteurs denses de 384 dimensions.

**Pourquoi Sentence-BERT ?**
- Pré-entraîné sur des paires de phrases sémantiquement similaires
- Optimisé pour la similarité cosinus
- Léger (80MB) et rapide à inférer
- Performance state-of-the-art sur les tâches de similarité sémantique

**Processus:**
```python
# Pour chaque item
text = f"{movie_title}. {movie_plot}"
embedding = sentence_bert.encode(text)  # → vecteur de dimension 384
```

#### 2. Construction des Profils Utilisateurs

Chaque utilisateur est représenté par un **profil vectoriel** calculé comme la moyenne pondérée des embeddings des items avec lesquels il a interagi.

**Formule mathématique:**

$$\mathbf{u}_i = \frac{\sum_{j \in I_i} r_{ij}' \cdot \mathbf{e}_j}{\sum_{j \in I_i} r_{ij}'}$$

Où:
- $\mathbf{u}_i$ : profil de l'utilisateur $i$
- $I_i$ : ensemble des items avec lesquels l'utilisateur $i$ a interagi
- $r_{ij}'$ : rating normalisé de l'utilisateur $i$ pour l'item $j$
- $\mathbf{e}_j$ : embedding de l'item $j$

**Normalisation des ratings:**
```python
r' = (r - r_min) / (r_max - r_min)  # Normalisation entre 0 et 1
```

Cette approche permet de:
- Capturer les préférences sémantiques de l'utilisateur
- Donner plus de poids aux items très appréciés
- Créer un profil dense même avec peu d'interactions

#### 3. Recommandation par Similarité

Pour recommander des items à un utilisateur, nous calculons la **similarité cosinus** entre son profil et tous les embeddings d'items.

**Formule:**

$$\text{score}(u_i, item_j) = \frac{\mathbf{u}_i \cdot \mathbf{e}_j}{||\mathbf{u}_i|| \cdot ||\mathbf{e}_j||}$$

**Top-K recommandations:**
```python
scores = cosine_similarity(user_profile, all_item_embeddings)
top_k = argsort(scores)[-k:][::-1]
```

## 🔀 Modèle Hybride: `HybridEmbeddingRecommender`

Pour améliorer la robustesse, nous proposons une version **hybride** combinant:
- **Embeddings sémantiques** (70%) : capturent la similarité de contenu
- **Popularité collaborative** (30%) : exploitent les tendances générales

**Formule:**

$$\text{score}_{\text{hybrid}}(u, i) = \alpha \cdot \text{score}_{\text{embedding}}(u, i) + (1-\alpha) \cdot \text{popularity}(i)$$

Avec $\alpha = 0.7$ par défaut.

**Avantages:**
- Meilleure performance sur les utilisateurs "froids" (peu d'interactions)
- Combine forces du content-based et du collaboratif
- Réduit le biais vers les items de niche

## 📊 Comparaison avec les Baselines

### Matrix Factorization (MF)

**Principe:**
Factorise la matrice user-item $R$ en deux matrices latentes:

$$R \approx U \times V^T$$

Où $U \in \mathbb{R}^{n_{\text{users}} \times k}$ et $V \in \mathbb{R}^{n_{\text{items}} \times k}$

**Limitations:**
- Ne capture que les patterns collaboratifs
- Ignore les informations textuelles
- Problème du cold start pour nouveaux items

### Bayesian Personalized Ranking (BPR)

**Principe:**
Optimise directement le classement en apprenant:

$$p(i >_u j) = \sigma(\hat{r}_{ui} - \hat{r}_{uj})$$

**Perte BPR:**

$$\mathcal{L} = \sum_{(u,i,j) \in D_S} -\ln \sigma(x_{uij}) + \lambda ||\Theta||^2$$

Où $(u,i,j)$ sont des triplets (utilisateur, item positif, item négatif).

**Limitations:**
- Similaires à MF
- Ignore le contenu textuel

## 🎯 Métriques d'Évaluation

### Recall@K

Mesure la proportion d'items pertinents retrouvés dans les K premières recommandations.

$$\text{Recall@K} = \frac{|\text{items pertinents} \cap \text{top-K}|}{|\text{items pertinents}|}$$

**Interprétation:**
- Recall@10 = 0.15 signifie que 15% des items pertinents sont dans le top-10

### NDCG@K (Normalized Discounted Cumulative Gain)

Évalue la qualité du classement en pénalisant les items pertinents mal positionnés.

**DCG@K:**

$$\text{DCG@K} = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i+1)}$$

**NDCG@K:**

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

Où $\text{IDCG@K}$ est le DCG idéal (meilleur classement possible).

**Interprétation:**
- NDCG@10 = 0.12 indique une bonne qualité de classement (1.0 = parfait)

### Precision@K

Proportion d'items pertinents parmi les K recommandations.

$$\text{Precision@K} = \frac{|\text{items pertinents} \cap \text{top-K}|}{K}$$

## 💪 Avantages de Notre Approche

### 1. Résolution du Cold Start

**Problème traditionnel:**
- Nouveaux items sans interactions → impossibles à recommander

**Notre solution:**
- Si un nouveau film a une description, on peut l'encoder immédiatement
- Recommandable dès le premier jour sans historique

### 2. Interprétabilité

**Embeddings sémantiques:**
- Les dimensions capturent des concepts (genre, thème, ambiance)
- Permet d'expliquer pourquoi un item est recommandé
- Exemple: "Ce film est similaire car il partage le thème de la science-fiction"

### 3. Transfert de Connaissances

**BERT pré-entraîné:**
- Connaissances linguistiques générales
- Comprend les nuances sémantiques
- Pas besoin de ré-entraîner sur domaine spécifique

### 4. Scalabilité

**Temps de calcul:**
- Encodage des items: une seule fois (offline)
- Recommandation: simple produit matriciel (rapide)
- Ajout de nouveaux items: encodage incrémental

## 🔧 Détails d'Implémentation

### Architecture du Modèle

```python
class EmbeddingBasedRecommender(Recommender):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.item_embeddings = None  # Shape: (n_items, 384)
        self.user_profiles = None    # Shape: (n_users, 384)
    
    def fit(self, train_set):
        # 1. Encoder tous les items
        self.item_embeddings = self.encoder.encode(item_texts)
        
        # 2. Construire profils utilisateurs
        self._build_user_profiles()
    
    def score(self, user_idx, item_idx=None):
        # 3. Calculer similarité cosinus
        return cosine_similarity(
            self.user_profiles[user_idx],
            self.item_embeddings[item_idx]
        )
```

### Optimisations

1. **Batch Encoding**: Encoder les items par lots de 32
2. **Normalisation L2**: Pré-normaliser les embeddings pour accélérer cosine
3. **Sparse Storage**: Stocker les interactions en format CSR

## 📈 Résultats Attendus

### Performances Typiques (MovieLens 100K)

| Modèle | Recall@10 | NDCG@10 | Temps |
|--------|-----------|---------|-------|
| Matrix Factorization | 0.12-0.13 | 0.09-0.10 | ~15s |
| BPR | 0.13-0.14 | 0.10-0.11 | ~20s |
| **EmbeddingRecommender** | **0.14-0.15** | **0.11-0.12** | ~45s |
| **HybridEmbedding** | **0.15-0.16** | **0.12-0.13** | ~50s |

**Analyse:**
- Amélioration de **~15-20%** sur Recall@10 vs MF
- Amélioration de **~20-30%** sur NDCG@10 vs MF
- Coût computationnel 3x plus élevé (acceptable offline)

## 🚀 Extensions Possibles

### 1. Fine-tuning du BERT
Ré-entraîner le modèle BERT sur le domaine spécifique:
```python
from sentence_transformers import losses
model = SentenceTransformer('all-MiniLM-L6-v2')
# Fine-tune sur paires (film similaire, film différent)
```

### 2. Attention Mechanism
Apprendre des poids d'attention pour les items dans le profil utilisateur:
```python
attention_weights = softmax(W @ item_embeddings)
user_profile = attention_weights @ item_embeddings
```

### 3. Multi-modal Embeddings
Combiner texte, images (posters), et audio (bandes-annonces):
```python
embedding = concat([text_emb, image_emb, audio_emb])
```

### 4. Embeddings Contextuels
Adapter les embeddings au contexte temporel ou situationnel:
```python
embedding = f(item_text, user_context, time)
```

## 📚 Références

### Papers Fondamentaux

1. **Sentence-BERT**
   - Reimers & Gurevych (2019)
   - "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
   - EMNLP 2019

2. **Matrix Factorization**
   - Koren et al. (2009)
   - "Matrix Factorization Techniques for Recommender Systems"
   - IEEE Computer

3. **BPR**
   - Rendle et al. (2009)
   - "BPR: Bayesian Personalized Ranking from Implicit Feedback"
   - UAI 2009

4. **Content-Based Recommendations**
   - Van Meteren & Van Someren (2000)
   - "Using Content-Based Filtering for Recommendation"

### Bibliothèques Utilisées

- **Cornac**: Framework de recommandation
  - https://cornac.readthedocs.io/
  
- **Sentence-Transformers**: Embeddings BERT
  - https://www.sbert.net/
  
- **PyTorch**: Backend deep learning
  - https://pytorch.org/

## 💡 Conclusion

Notre approche démontre que les **embeddings de modèles de langage** peuvent significativement améliorer les systèmes de recommandation traditionnels en:

1. ✅ Capturant la sémantique des items
2. ✅ Résolvant le problème du cold start
3. ✅ Offrant une meilleure interprétabilité
4. ✅ Permettant le transfert de connaissances

Le surcoût computationnel est acceptable pour des applications réelles où les embeddings sont calculés offline.

---

**Pour toute question technique, consulter le code source dans `models/embedding_recommender.py`**
