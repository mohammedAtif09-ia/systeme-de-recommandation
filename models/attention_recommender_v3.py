"""
Modèle de recommandation AVANCÉ avec Attention Mechanism - Version 3
Corrigé avec: padding propre, negative sampling amélioré, masques, GPU support
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from cornac.models import Recommender
import random
from collections import Counter


class AttentionLayer(nn.Module):
    """
    Couche d'attention améliorée pour pondérer les items
    """
    def __init__(self, embedding_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim // 2, 1)
        )
    
    def forward(self, embeddings, mask=None):
        """
        Args:
            embeddings: (batch_size, num_items, embedding_dim)
            mask: (batch_size, num_items) - masque pour items valides (1=valide, 0=padding)
        
        Returns:
            weighted_embedding: (batch_size, embedding_dim)
            attention_weights: (batch_size, num_items)
        """
        attention_scores = self.attention(embeddings).squeeze(-1)
        
        if mask is not None:
            # Masquer les positions de padding avec -inf avant softmax
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(attention_scores / 0.5, dim=-1)# on prend un température de 0.5 pour des poids plus concentrés
        
        # Gérer le cas où tous les éléments sont masqués (éviter NaN)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        weighted_embedding = torch.sum(
            embeddings * attention_weights.unsqueeze(-1), 
            dim=1
        )
        
        return weighted_embedding, attention_weights


class UserEncoder(nn.Module):
    """
    Encodeur d'utilisateur avec attention sur l'historique
    """
    def __init__(self, embedding_dim):
        super(UserEncoder, self).__init__()
        self.attention = AttentionLayer(embedding_dim)
        self.transform = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, item_embeddings, mask=None):
        """
        Encode le profil utilisateur à partir de son historique
        """
        user_profile, attn_weights = self.attention(item_embeddings, mask)
        user_profile = self.transform(user_profile)
        return user_profile, attn_weights


class ContrastiveAttentionRecommender(Recommender):
    """
    Système de recommandation avec Contrastive Learning - Version corrigée
    
    Corrections appliquées:
    - Padding avec un vecteur zéro dédié (pas l'item 0)
    - Negative sampling pondéré par popularité
    - SentenceTransformer sur GPU si disponible
    - Masques corrects partout
    - Loss vectorisée pour performance
    - Régularisation des profils utilisateurs
    """
    
    def __init__(
        self,
        name="ContrastiveAttention",
        model_name='all-MiniLM-L6-v2',
        trainable=True,
        verbose=True,
        # learning_rate=0.001,  # Ancienne valeur
        learning_rate=0.002,  # Nouvelle valeur: convergence plus rapide
        # num_epochs=20,  # Ancienne valeur
        num_epochs=40,  # Nouvelle valeur: meilleure convergence
        # batch_size=256,  # Ancienne valeur
        batch_size=512,  # Nouvelle valeur: plus stable
        # num_negatives=5,  # Ancienne valeur
        num_negatives=10,  # Nouvelle valeur: meilleur contraste
        margin=0.5,
        reg_lambda=1e-4,  # Régularisation
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.margin = margin
        self.reg_lambda = reg_lambda
        self.encoder = None
        self.item_embeddings = None
        self.item_embeddings_tensor = None
        self.user_encoder = None
        self.user_profiles = None
        self.attention_weights_cache = {}  # Pour l'interprétabilité
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.padding_idx = None
        self.item_popularity = None  # Pour negative sampling pondéré
        self.global_mean_embedding = None  # Pour cold start
        
    def _compute_item_popularity(self):
        """
        Calcule la popularité de chaque item pour le negative sampling pondéré
        """
        item_counts = Counter()
        for user_idx in range(self.train_set.num_users):
            user_items, _ = self.train_set.user_data[user_idx]
            item_counts.update(user_items)
        
        # Convertir en probabilités (items populaires = plus de chance d'être négatifs)
        total = sum(item_counts.values())
        self.item_popularity = np.zeros(self.train_set.num_items)
        for item_idx, count in item_counts.items():
            # Popularité avec smoothing
            self.item_popularity[item_idx] = (count + 1) / (total + self.train_set.num_items)
        
        # Normaliser
        self.item_popularity = self.item_popularity / self.item_popularity.sum()
        
    def _sample_negatives_weighted(self, user_items_set, num_samples):
        """
        Échantillonne des items négatifs pondérés par popularité
        Les items populaires ont plus de chances d'être sélectionnés comme négatifs
        (hard negatives strategy)
        """
        negatives = []
        attempts = 0
        max_attempts = num_samples * 10
        
        while len(negatives) < num_samples and attempts < max_attempts:
            # Sampling pondéré par popularité
            neg = np.random.choice(
                self.train_set.num_items, 
                p=self.item_popularity
            )
            if neg not in user_items_set:
                negatives.append(neg)
            attempts += 1
        
        # Fallback si pas assez de négatifs
        while len(negatives) < num_samples:
            neg = random.randint(0, self.train_set.num_items - 1)
            if neg not in user_items_set:
                negatives.append(neg)
        
        return negatives
    
    def fit(self, train_set, val_set=None):
        """
        Entraîne le modèle avec contrastive learning
        """
        super().fit(train_set, val_set)
        
        if self.verbose:
            print(f"Initialisation du modèle d'embeddings: {self.model_name}")
            print(f"Device: {self.device}")
        
        # 1. Charger le modèle de sentence embeddings SUR LE BON DEVICE
        self.encoder = SentenceTransformer(self.model_name, device=str(self.device))
        
        # 2. Récupérer les textes des items
        if not hasattr(train_set, 'item_text') or train_set.item_text is None:
            raise ValueError("Le dataset doit contenir des textes d'items")
        
        item_texts = train_set.item_text
        num_items = self.train_set.num_items
        
        if self.verbose:
            print(f"Génération des embeddings pour {num_items} items...")
        
        # 3. Générer les embeddings pour tous les items
        item_texts_list = []
        for item_idx in range(num_items):
            text = item_texts.get(item_idx, "")
            item_texts_list.append(text if text else "unknown item")
        
        self.item_embeddings = self.encoder.encode(
            item_texts_list,
            show_progress_bar=self.verbose,
            convert_to_numpy=True,
            batch_size=32
        )
        
        embedding_dim = self.item_embeddings.shape[1]
        
        # 4. CORRECTION: Ajouter un embedding de padding (vecteur zéro)
        self.padding_idx = num_items  # Index du padding = après tous les items
        padding_embedding = np.zeros((1, embedding_dim))
        self.item_embeddings = np.vstack([self.item_embeddings, padding_embedding])
        
        # 5. Calculer l'embedding moyen global (pour cold start)
        self.global_mean_embedding = self.item_embeddings[:num_items].mean(axis=0)
        
        # Convertir en tensor
        self.item_embeddings_tensor = torch.FloatTensor(self.item_embeddings).to(self.device)
        
        # 6. Calculer la popularité des items pour le negative sampling
        if self.verbose:
            print("Calcul de la popularité des items...")
        self._compute_item_popularity()
        
        # 7. Initialiser l'encodeur utilisateur
        self.user_encoder = UserEncoder(embedding_dim).to(self.device)
        optimizer = torch.optim.Adam(
            self.user_encoder.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        # AMÉLIORATION: Learning rate scheduler (réduit le LR progressivement)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        
        if self.verbose:
            print(f"Entraînement contrastif ({self.num_epochs} epochs)...")
        
        # 8. Préparer les données d'entraînement
        training_data = []
        for user_idx in range(self.train_set.num_users):
            user_items, user_ratings = self.train_set.user_data[user_idx]
            if len(user_items) >= 3:
                user_ratings_arr = np.array(user_ratings)
                user_items_arr = np.array(user_items)
                
                # Trier par rating et prendre le meilleur comme positif
                sorted_indices = np.argsort(user_ratings_arr)[::-1]
                
                positive_item = user_items_arr[sorted_indices[0]]
                history_items = user_items_arr[sorted_indices[1:]].tolist()
                
                if len(history_items) >= 2:
                    training_data.append({
                        'user_idx': user_idx,
                        # 'history': history_items[:20],  # Ancienne valeur
                        'history': history_items[:30],  # Nouvelle valeur: plus de contexte
                        'positive': positive_item,
                        'all_items': set(user_items),  # Pour le negative sampling
                    })
        
        if self.verbose:
            print(f"Données d'entraînement: {len(training_data)} triplets")
        
        # 9. Entraînement
        self.user_encoder.train()
        
        for epoch in range(self.num_epochs):
            random.shuffle(training_data)
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(training_data), self.batch_size):
                batch = training_data[i:i+self.batch_size]
                
                max_hist_len = max(len(d['history']) for d in batch)
                
                batch_histories = []
                batch_masks = []
                batch_positives = []
                batch_negatives = []
                
                for d in batch:
                    hist = d['history']
                    # CORRECTION: Utiliser padding_idx au lieu de 0
                    padded_hist = hist + [self.padding_idx] * (max_hist_len - len(hist))
                    mask = [1.0] * len(hist) + [0.0] * (max_hist_len - len(hist))
                    
                    batch_histories.append(padded_hist)
                    batch_masks.append(mask)
                    batch_positives.append(d['positive'])
                    # CORRECTION: Negative sampling pondéré
                    batch_negatives.append(self._sample_negatives_weighted(
                        d['all_items'], self.num_negatives
                    ))
                
                # Convertir en tensors
                histories_idx = torch.LongTensor(batch_histories).to(self.device)
                masks = torch.FloatTensor(batch_masks).to(self.device)
                positive_idx = torch.LongTensor(batch_positives).to(self.device)
                negative_idx = torch.LongTensor(batch_negatives).to(self.device)
                
                # Obtenir les embeddings
                history_embeds = self.item_embeddings_tensor[histories_idx]
                positive_embeds = self.item_embeddings_tensor[positive_idx]
                negative_embeds = self.item_embeddings_tensor[negative_idx]
                
                # CORRECTION: Passer le masque à l'encodeur
                user_profiles, _ = self.user_encoder(history_embeds, masks)
                
                # Normaliser
                user_profiles = F.normalize(user_profiles, p=2, dim=-1)
                positive_embeds = F.normalize(positive_embeds, p=2, dim=-1)
                negative_embeds = F.normalize(negative_embeds, p=2, dim=-1)
                
                # Similarités
                pos_sim = torch.sum(user_profiles * positive_embeds, dim=-1)  # (batch,)
                neg_sim = torch.bmm(
                    negative_embeds, 
                    user_profiles.unsqueeze(-1)
                ).squeeze(-1)  # (batch, num_neg)
                
                # AMÉLIORATION: BPR Loss vectorisée (plus rapide)
                # pos_sim: (batch,) -> (batch, 1) pour broadcasting
                # neg_sim: (batch, num_neg)
                bpr_loss = -torch.mean(F.logsigmoid(pos_sim.unsqueeze(-1) - neg_sim))
                
                # Margin loss
                margin_loss = torch.mean(F.relu(self.margin - pos_sim.unsqueeze(-1) + neg_sim))
                
                # AMÉLIORATION: Régularisation des profils
                reg_loss = self.reg_lambda * torch.mean(torch.norm(user_profiles, p=2, dim=-1))
                
                # Loss totale
                loss = bpr_loss + 0.1 * margin_loss + reg_loss
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.user_encoder.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Mettre à jour le learning rate
            scheduler.step()
            
            if self.verbose and (epoch + 1) % 5 == 0:
                avg_loss = total_loss / max(num_batches, 1)
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        
        # 10. Construire les profils utilisateurs finaux
        if self.verbose:
            print("Construction des profils utilisateurs...")
        
        self._build_user_profiles()
        
        return self
    
    def _build_user_profiles(self):
        """
        Construit les profils utilisateurs avec l'encodeur entraîné
        CORRECTION: Utilise les masques correctement
        """
        self.user_encoder.eval()
        embedding_dim = self.item_embeddings.shape[1]
        self.user_profiles = np.zeros((self.train_set.num_users, embedding_dim))
        
        with torch.no_grad():
            for user_idx in tqdm(range(self.train_set.num_users), disable=not self.verbose):
                user_items, _ = self.train_set.user_data[user_idx]
                
                # AMÉLIORATION: Cold start - utiliser l'embedding moyen
                if len(user_items) == 0:
                    self.user_profiles[user_idx] = self.global_mean_embedding
                    continue
                
                # Limiter l'historique
                user_items = list(user_items)[:50]
                
                item_embeds = torch.FloatTensor(
                    self.item_embeddings[user_items]
                ).unsqueeze(0).to(self.device)
                
                # CORRECTION: Créer le masque (tous à 1 car pas de padding ici)
                mask = torch.ones(1, len(user_items)).to(self.device)
                
                user_profile, attn_weights = self.user_encoder(item_embeds, mask)
                user_profile = F.normalize(user_profile, p=2, dim=-1)
                
                self.user_profiles[user_idx] = user_profile.cpu().numpy().squeeze()
                
                # AMÉLIORATION: Stocker les poids d'attention pour l'interprétabilité
                self.attention_weights_cache[user_idx] = {
                    'items': user_items,
                    'weights': attn_weights.cpu().numpy().squeeze()
                }
    
    def score(self, user_idx, item_idx=None):
        """
        Calcule le score de recommandation
        """
        if self.user_profiles is None:
            raise ValueError("Le modèle n'a pas été entraîné")
        
        user_profile = self.user_profiles[user_idx]
        
        if item_idx is None:
            # Score pour tous les items (exclure le padding)
            item_embeds = self.item_embeddings[:self.train_set.num_items]
            item_embeds_norm = item_embeds / (
                np.linalg.norm(item_embeds, axis=1, keepdims=True) + 1e-10
            )
            scores = np.dot(item_embeds_norm, user_profile)
            return scores
        else:
            item_embed = self.item_embeddings[item_idx]
            item_embed_norm = item_embed / (np.linalg.norm(item_embed) + 1e-10)
            score = np.dot(item_embed_norm, user_profile)
            return score
    
    def explain_recommendation(self, user_idx, top_k=5):
        """
        AMÉLIORATION: Explique pourquoi un utilisateur reçoit certaines recommandations
        Retourne les items de l'historique les plus influents
        """
        if user_idx not in self.attention_weights_cache:
            return None
        
        cache = self.attention_weights_cache[user_idx]
        items = cache['items']
        weights = cache['weights']
        
        # Trier par poids d'attention
        sorted_indices = np.argsort(weights)[::-1][:top_k]
        
        explanations = []
        for idx in sorted_indices:
            explanations.append({
                'item_idx': items[idx],
                'attention_weight': float(weights[idx]),
            })
        
        return explanations


# Alias pour compatibilité
AttentionBasedRecommender = ContrastiveAttentionRecommender
