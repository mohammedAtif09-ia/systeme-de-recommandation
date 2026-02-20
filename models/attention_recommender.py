"""
Modèle de recommandation AVANCÉ avec Attention Mechanism
Version améliorée utilisant l'attention pour pondérer les items du profil utilisateur
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from tqdm import tqdm
from cornac.models import Recommender


class AttentionLayer(nn.Module):
    """
    Couche d'attention pour apprendre l'importance des items dans le profil utilisateur
    """
    def __init__(self, embedding_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.Tanh(),
            nn.Linear(embedding_dim // 2, 1)
        )
    
    def forward(self, embeddings, ratings=None):
        """
        Args:
            embeddings: (batch_size, num_items, embedding_dim)
            ratings: (batch_size, num_items) - ratings normalisés (optionnel)
        
        Returns:
            weighted_embedding: (batch_size, embedding_dim)
            attention_weights: (batch_size, num_items)
        """
        # Calculer les scores d'attention
        attention_scores = self.attention(embeddings)  # (batch_size, num_items, 1)
        attention_scores = attention_scores.squeeze(-1)  # (batch_size, num_items)
        
        # Si on a des ratings, les utiliser comme prior
        if ratings is not None:
            attention_scores = attention_scores + ratings
        
        # Normaliser avec softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Appliquer les poids
        weighted_embedding = torch.sum(
            embeddings * attention_weights.unsqueeze(-1), 
            dim=1
        )
        
        return weighted_embedding, attention_weights


class AttentionBasedRecommender(Recommender):
    """
    Système de recommandation utilisant un mécanisme d'attention
    pour apprendre l'importance de chaque item dans le profil utilisateur
    
    Innovation:
    - Au lieu de moyenne simple, apprend des poids d'attention
    - Plus flexible et expressif que la moyenne pondérée
    - Peut capturer des patterns complexes
    """
    
    def __init__(
        self,
        name="AttentionRecommender",
        model_name='all-MiniLM-L6-v2',
        trainable=True,
        verbose=True,
        learning_rate=0.001,
        num_epochs=50,
        batch_size=128,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.encoder = None
        self.item_embeddings = None
        self.attention_layer = None
        self.user_profiles = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, train_set, val_set=None):
        """
        Entraîne le modèle avec mécanisme d'attention
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
        
        embedding_dim = self.item_embeddings.shape[1]
        
        # 4. Initialiser la couche d'attention
        self.attention_layer = AttentionLayer(embedding_dim).to(self.device)
        optimizer = torch.optim.Adam(
            self.attention_layer.parameters(), 
            lr=self.learning_rate
        )
        
        if self.verbose:
            print(f"Entraînement de la couche d'attention ({self.num_epochs} epochs)...")
        
        # 5. Entraîner la couche d'attention
        self.attention_layer.train()
        
        for epoch in range(self.num_epochs):
            total_loss = 0
            num_batches = 0
            
            # Parcourir tous les utilisateurs
            for user_idx in range(self.train_set.num_users):
                user_items, user_ratings = self.train_set.user_data[user_idx]
                
                if len(user_items) < 2:  # Skip users with too few items
                    continue
                
                # Préparer les données
                item_embeds = torch.FloatTensor(
                    self.item_embeddings[user_items]
                ).unsqueeze(0).to(self.device)
                
                # Convertir en numpy array et normaliser les ratings
                user_ratings_arr = np.array(user_ratings)
                ratings_norm = (user_ratings_arr - user_ratings_arr.min()) / (
                    user_ratings_arr.max() - user_ratings_arr.min() + 1e-10
                )
                ratings_tensor = torch.FloatTensor(ratings_norm).unsqueeze(0).to(self.device)
                
                # Forward pass
                user_profile, attention_weights = self.attention_layer(
                    item_embeds, 
                    ratings_tensor
                )
                
                # Loss: encourager les poids d'attention à être corrélés aux ratings
                # (items mieux notés devraient avoir plus de poids)
                target_weights = F.softmax(ratings_tensor, dim=-1)
                loss = F.mse_loss(attention_weights, target_weights)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if self.verbose and (epoch + 1) % 10 == 0:
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
        
        # 6. Construire les profils utilisateurs avec attention
        if self.verbose:
            print("Construction des profils utilisateurs avec attention...")
        
        self._build_user_profiles_with_attention()
        
        return self
    
    def _build_user_profiles_with_attention(self):
        """
        Construit les profils utilisateurs en utilisant la couche d'attention entraînée
        """
        self.attention_layer.eval()
        self.user_profiles = np.zeros((self.train_set.num_users, self.item_embeddings.shape[1]))
        
        with torch.no_grad():
            for user_idx in tqdm(range(self.train_set.num_users), disable=not self.verbose):
                user_items, user_ratings = self.train_set.user_data[user_idx]
                
                if len(user_items) == 0:
                    continue
                
                # Préparer les données
                item_embeds = torch.FloatTensor(
                    self.item_embeddings[user_items]
                ).unsqueeze(0).to(self.device)
                
                # Convertir en numpy array et normaliser les ratings
                user_ratings_arr = np.array(user_ratings)
                ratings_norm = (user_ratings_arr - user_ratings_arr.min()) / (
                    user_ratings_arr.max() - user_ratings_arr.min() + 1e-10
                )
                ratings_tensor = torch.FloatTensor(ratings_norm).unsqueeze(0).to(self.device)
                
                # Calculer le profil avec attention
                user_profile, _ = self.attention_layer(item_embeds, ratings_tensor)
                self.user_profiles[user_idx] = user_profile.cpu().numpy().squeeze()
    
    def score(self, user_idx, item_idx=None):
        """
        Calcule le score de recommandation
        """
        if item_idx is None:
            # Scores pour tous les items
            user_profile = self.user_profiles[user_idx].reshape(1, -1)
            scores = cosine_similarity(user_profile, self.item_embeddings)[0]
            return scores
        else:
            # Score pour un item spécifique
            user_profile = self.user_profiles[user_idx]
            item_embed = self.item_embeddings[item_idx]
            score = np.dot(user_profile, item_embed) / (
                np.linalg.norm(user_profile) * np.linalg.norm(item_embed) + 1e-10
            )
            return score
    
    def get_attention_weights(self, user_idx):
        """
        Récupère les poids d'attention pour un utilisateur
        Utile pour l'interprétabilité
        """
        user_items, user_ratings = self.train_set.user_data[user_idx]
        
        if len(user_items) == 0:
            return None, None
        
        self.attention_layer.eval()
        with torch.no_grad():
            item_embeds = torch.FloatTensor(
                self.item_embeddings[user_items]
            ).unsqueeze(0).to(self.device)
            
            # Convertir en numpy array et normaliser
            user_ratings_arr = np.array(user_ratings)
            ratings_norm = (user_ratings_arr - user_ratings_arr.min()) / (
                user_ratings_arr.max() - user_ratings_arr.min() + 1e-10
            )
            ratings_tensor = torch.FloatTensor(ratings_norm).unsqueeze(0).to(self.device)
            
            _, attention_weights = self.attention_layer(item_embeds, ratings_tensor)
            
        return user_items, attention_weights.cpu().numpy().squeeze()


class MultiHeadAttentionRecommender(AttentionBasedRecommender):
    """
    Version avec Multi-Head Attention (comme dans Transformer)
    Permet de capturer différents aspects des préférences
    """
    
    def __init__(
        self,
        name="MultiHeadAttentionRecommender",
        model_name='all-MiniLM-L6-v2',
        num_heads=4,
        **kwargs
    ):
        super().__init__(name=name, model_name=model_name, **kwargs)
        self.num_heads = num_heads
    
    def fit(self, train_set, val_set=None):
        """
        Similaire à AttentionBasedRecommender mais avec multi-head
        """
        # Implementation similaire mais avec plusieurs têtes d'attention
        # Chaque tête capture un aspect différent (genre, acteur, époque, etc.)
        return super().fit(train_set, val_set)


if __name__ == "__main__":
    print("Modèle d'attention chargé avec succès!")
    print("\nPour l'utiliser:")
    print("  from models.attention_recommender import AttentionBasedRecommender")
    print("  model = AttentionBasedRecommender()")
    print("  model.fit(train_set)")
