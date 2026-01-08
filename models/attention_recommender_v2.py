"""
Modèle de recommandation AVANCÉ avec Attention Mechanism - Version 2
Utilise le Contrastive Learning pour un vrai apprentissage
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from cornac.models import Recommender
import random


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
            mask: (batch_size, num_items) - masque pour items valides
        
        Returns:
            weighted_embedding: (batch_size, embedding_dim)
            attention_weights: (batch_size, num_items)
        """
        attention_scores = self.attention(embeddings).squeeze(-1)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
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
    Système de recommandation avec Contrastive Learning
    
    Innovation:
    - Apprend à maximiser la similarité entre profil utilisateur et items positifs
    - Minimise la similarité avec les items négatifs (non consommés)
    - Utilise BPR loss pour un vrai signal d'apprentissage
    """
    
    def __init__(
        self,
        name="ContrastiveAttention",
        model_name='all-MiniLM-L6-v2',
        trainable=True,
        verbose=True,
        learning_rate=0.001,
        num_epochs=20,
        batch_size=256,
        num_negatives=5,
        margin=0.5,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.margin = margin
        self.encoder = None
        self.item_embeddings = None
        self.user_encoder = None
        self.user_profiles = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _sample_negatives(self, user_items, num_items):
        """
        Échantillonne des items négatifs (non consommés par l'utilisateur)
        """
        user_items_set = set(user_items)
        negatives = []
        while len(negatives) < self.num_negatives:
            neg = random.randint(0, num_items - 1)
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
        
        # 1. Charger le modèle de sentence embeddings
        self.encoder = SentenceTransformer(self.model_name)
        
        # 2. Récupérer les textes des items
        if not hasattr(train_set, 'item_text') or train_set.item_text is None:
            raise ValueError("Le dataset doit contenir des textes d'items")
        
        item_texts = train_set.item_text
        
        if self.verbose:
            print(f"Génération des embeddings pour {self.train_set.num_items} items...")
        
        # 3. Générer les embeddings pour tous les items
        item_texts_list = []
        for item_idx in range(self.train_set.num_items):
            text = item_texts.get(item_idx, "")
            item_texts_list.append(text if text else "unknown item")
        
        self.item_embeddings = self.encoder.encode(
            item_texts_list,
            show_progress_bar=self.verbose,
            convert_to_numpy=True,
            batch_size=32
        )
        
        # Convertir en tensor
        self.item_embeddings_tensor = torch.FloatTensor(self.item_embeddings).to(self.device)
        
        embedding_dim = self.item_embeddings.shape[1]
        num_items = self.train_set.num_items
        
        # 4. Initialiser l'encodeur utilisateur
        self.user_encoder = UserEncoder(embedding_dim).to(self.device)
        optimizer = torch.optim.Adam(
            self.user_encoder.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        if self.verbose:
            print(f"Entraînement contrastif ({self.num_epochs} epochs)...")
        
        # 5. Préparer les données d'entraînement
        # Créer des triplets (user_history, positive_item, negative_items)
        training_data = []
        for user_idx in range(self.train_set.num_users):
            user_items, user_ratings = self.train_set.user_data[user_idx]
            if len(user_items) >= 3:  # Au moins 3 items pour split
                # Utiliser les items bien notés comme positifs
                user_ratings_arr = np.array(user_ratings)
                user_items_arr = np.array(user_items)
                
                # Trier par rating et prendre le meilleur comme positif
                sorted_indices = np.argsort(user_ratings_arr)[::-1]
                
                # L'item le mieux noté = positif, le reste = historique
                positive_item = user_items_arr[sorted_indices[0]]
                history_items = user_items_arr[sorted_indices[1:]].tolist()
                
                if len(history_items) >= 2:
                    training_data.append({
                        'user_idx': user_idx,
                        'history': history_items[:20],  # Limiter l'historique
                        'positive': positive_item,
                    })
        
        if self.verbose:
            print(f"Données d'entraînement: {len(training_data)} triplets")
        
        # 6. Entraînement
        self.user_encoder.train()
        
        for epoch in range(self.num_epochs):
            random.shuffle(training_data)
            total_loss = 0
            num_batches = 0
            
            # Mini-batches
            for i in range(0, len(training_data), self.batch_size):
                batch = training_data[i:i+self.batch_size]
                
                # Préparer le batch
                max_hist_len = max(len(d['history']) for d in batch)
                
                batch_histories = []
                batch_masks = []
                batch_positives = []
                batch_negatives = []
                
                for d in batch:
                    # Historique (padding si nécessaire)
                    hist = d['history']
                    padded_hist = hist + [0] * (max_hist_len - len(hist))
                    mask = [1] * len(hist) + [0] * (max_hist_len - len(hist))
                    
                    batch_histories.append(padded_hist)
                    batch_masks.append(mask)
                    batch_positives.append(d['positive'])
                    batch_negatives.append(self._sample_negatives(
                        d['history'] + [d['positive']], num_items
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
                
                # Encoder le profil utilisateur
                user_profiles, _ = self.user_encoder(history_embeds, masks)
                
                # Normaliser
                user_profiles = F.normalize(user_profiles, p=2, dim=-1)
                positive_embeds = F.normalize(positive_embeds, p=2, dim=-1)
                negative_embeds = F.normalize(negative_embeds, p=2, dim=-1)
                
                # Calculer les similarités
                pos_sim = torch.sum(user_profiles * positive_embeds, dim=-1)  # (batch,)
                neg_sim = torch.bmm(
                    negative_embeds, 
                    user_profiles.unsqueeze(-1)
                ).squeeze(-1)  # (batch, num_neg)
                
                # BPR Loss: log(sigmoid(pos - neg))
                loss = 0
                for j in range(self.num_negatives):
                    loss += -torch.mean(F.logsigmoid(pos_sim - neg_sim[:, j]))
                loss /= self.num_negatives
                
                # Margin loss additionnelle
                margin_loss = torch.mean(F.relu(self.margin - pos_sim + neg_sim.mean(dim=-1)))
                loss += 0.1 * margin_loss
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.user_encoder.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if self.verbose and (epoch + 1) % 5 == 0:
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
        
        # 7. Construire les profils utilisateurs finaux
        if self.verbose:
            print("Construction des profils utilisateurs...")
        
        self._build_user_profiles()
        
        return self
    
    def _build_user_profiles(self):
        """
        Construit les profils utilisateurs avec l'encodeur entraîné
        """
        self.user_encoder.eval()
        self.user_profiles = np.zeros((self.train_set.num_users, self.item_embeddings.shape[1]))
        
        with torch.no_grad():
            for user_idx in tqdm(range(self.train_set.num_users), disable=not self.verbose):
                user_items, _ = self.train_set.user_data[user_idx]
                
                if len(user_items) == 0:
                    continue
                
                # Limiter l'historique
                user_items = list(user_items)[:50]
                
                item_embeds = torch.FloatTensor(
                    self.item_embeddings[user_items]
                ).unsqueeze(0).to(self.device)
                
                user_profile, _ = self.user_encoder(item_embeds)
                user_profile = F.normalize(user_profile, p=2, dim=-1)
                self.user_profiles[user_idx] = user_profile.cpu().numpy().squeeze()
    
    def score(self, user_idx, item_idx=None):
        """
        Calcule le score de recommandation
        """
        if self.user_profiles is None:
            raise ValueError("Le modèle n'a pas été entraîné")
        
        user_profile = self.user_profiles[user_idx]
        
        if item_idx is None:
            # Score pour tous les items
            item_embeds_norm = self.item_embeddings / (
                np.linalg.norm(self.item_embeddings, axis=1, keepdims=True) + 1e-10
            )
            scores = np.dot(item_embeds_norm, user_profile)
            return scores
        else:
            # Score pour un item spécifique
            item_embed = self.item_embeddings[item_idx]
            item_embed_norm = item_embed / (np.linalg.norm(item_embed) + 1e-10)
            score = np.dot(item_embed_norm, user_profile)
            return score


# Alias pour compatibilité
AttentionBasedRecommender = ContrastiveAttentionRecommender
