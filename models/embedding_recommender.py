"""
Modèle de recommandation basé sur les embeddings de modèles de langage
Utilise Sentence-BERT pour créer des embeddings des items et recommander par similarité
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from tqdm import tqdm
from cornac.models import Recommender
from cornac.exception import ScoreException


class EmbeddingBasedRecommender(Recommender):
    """
    Système de recommandation basé sur les embeddings de texte
    
    Approche:
    1. Encode les descriptions d'items avec un modèle de langage (BERT)
    2. Calcule les préférences utilisateur comme moyenne pondérée des embeddings des items qu'il a aimés
    3. Recommande les items les plus similaires aux préférences de l'utilisateur
    """
    
    def __init__(
        self,
        name="EmbeddingRecommender",
        model_name='all-MiniLM-L6-v2',
        trainable=True,
        verbose=True,
        alpha=1.0,
    ):
        """
        Args:
            model_name: Nom du modèle SentenceTransformer à utiliser
            alpha: Facteur de pondération pour combiner contenu et collaboratif
        """
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.model_name = model_name
        self.alpha = alpha
        self.encoder = None
        self.item_embeddings = None
        self.user_profiles = None
        self.item_texts = None
        
    def fit(self, train_set, val_set=None):
        """
        Entraîne le modèle en créant les embeddings des items
        
        Args:
            train_set: Dataset d'entraînement Cornac
            val_set: Dataset de validation (optionnel)
        """
        super().fit(train_set, val_set)
        
        if self.verbose:
            print(f"Initialisation du modèle d'embeddings: {self.model_name}")
        
        # Charger le modèle de sentence embeddings
        self.encoder = SentenceTransformer(self.model_name)
        
        # Récupérer les textes des items depuis le train_set
        if not hasattr(train_set, 'item_text') or train_set.item_text is None:
            raise ValueError("Le dataset doit contenir des textes d'items (train_set.item_text)")
        
        self.item_texts = train_set.item_text
        
        if self.verbose:
            print(f"Génération des embeddings pour {self.train_set.num_items} items...")
        
        # Générer les embeddings pour tous les items
        item_texts_list = []
        for item_idx in range(self.train_set.num_items):
            text = self.item_texts.get(item_idx, "")
            item_texts_list.append(text if text else "unknown item")
        
        # Encoder tous les textes en batch
        self.item_embeddings = self.encoder.encode(
            item_texts_list,
            show_progress_bar=self.verbose,
            convert_to_numpy=True,
            batch_size=32
        )
        
        if self.verbose:
            print(f"Embeddings créés: shape {self.item_embeddings.shape}")
            print("Calcul des profils utilisateurs...")
        
        # Créer les profils utilisateurs comme moyenne pondérée des items aimés
        self._build_user_profiles()
        
        return self
    
    def _build_user_profiles(self):
        """
        Construit le profil de chaque utilisateur comme la moyenne pondérée
        des embeddings des items avec lesquels il a interagi
        """
        self.user_profiles = np.zeros((self.train_set.num_users, self.item_embeddings.shape[1]))
        
        for user_idx in tqdm(range(self.train_set.num_users), disable=not self.verbose):
            # Récupérer les items et ratings de l'utilisateur
            user_items, user_ratings = self.train_set.user_data[user_idx]
            
            if len(user_items) == 0:
                continue
            
            # Normaliser les ratings (utiliser comme poids)
            user_ratings = np.array(user_ratings)
            # Normaliser entre 0 et 1
            if user_ratings.max() > user_ratings.min():
                normalized_ratings = (user_ratings - user_ratings.min()) / (user_ratings.max() - user_ratings.min())
            else:
                normalized_ratings = np.ones_like(user_ratings)
            
            # Calculer le profil comme moyenne pondérée
            item_embeds = self.item_embeddings[user_items]
            weighted_embeds = item_embeds * normalized_ratings[:, np.newaxis]
            self.user_profiles[user_idx] = weighted_embeds.sum(axis=0) / normalized_ratings.sum()
    
    def score(self, user_idx, item_idx=None):
        """
        Calcule le score de recommandation pour un utilisateur et un/des item(s)
        
        Args:
            user_idx: Index de l'utilisateur
            item_idx: Index de l'item (ou None pour tous les items)
        
        Returns:
            Score(s) de recommandation
        """
        if item_idx is None:
            # Calculer les scores pour tous les items
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
    
    def save(self, save_dir=None):
        """Sauvegarde le modèle"""
        if save_dir is None:
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        model_file = os.path.join(save_dir, 'embedding_recommender.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump({
                'item_embeddings': self.item_embeddings,
                'user_profiles': self.user_profiles,
                'model_name': self.model_name,
                'alpha': self.alpha,
            }, f)
        
        if self.verbose:
            print(f"Modèle sauvegardé dans {model_file}")
    
    def load(self, save_dir):
        """Charge le modèle"""
        model_file = os.path.join(save_dir, 'embedding_recommender.pkl')
        
        with open(model_file, 'rb') as f:
            data = pickle.load(f)
        
        self.item_embeddings = data['item_embeddings']
        self.user_profiles = data['user_profiles']
        self.model_name = data['model_name']
        self.alpha = data['alpha']
        self.encoder = SentenceTransformer(self.model_name)
        
        if self.verbose:
            print(f"Modèle chargé depuis {model_file}")


class HybridEmbeddingRecommender(EmbeddingBasedRecommender):
    """
    Version hybride combinant embeddings et filtrage collaboratif simple
    """
    
    def __init__(self, name="HybridEmbeddingRecommender", model_name='all-MiniLM-L6-v2', 
                 trainable=True, verbose=True, alpha=0.7):
        """
        Args:
            alpha: Poids pour la partie embedding (1-alpha pour la partie collaborative)
        """
        super().__init__(name=name, model_name=model_name, trainable=trainable, 
                        verbose=verbose, alpha=alpha)
        self.item_popularity = None
    
    def fit(self, train_set, val_set=None):
        """Entraîne le modèle hybride"""
        super().fit(train_set, val_set)
        
        # Calculer la popularité des items
        item_counts = np.zeros(self.train_set.num_items)
        for user_idx in range(self.train_set.num_users):
            user_items, _ = self.train_set.user_data[user_idx]
            for item in user_items:
                item_counts[item] += 1
        
        # Normaliser
        self.item_popularity = item_counts / item_counts.max()
        
        return self
    
    def score(self, user_idx, item_idx=None):
        """Score hybride combinant embeddings et popularité"""
        # Score basé sur embeddings
        embedding_score = super().score(user_idx, item_idx)
        
        # Score de popularité
        if item_idx is None:
            popularity_score = self.item_popularity
        else:
            popularity_score = self.item_popularity[item_idx]
        
        # Combiner les scores
        final_score = self.alpha * embedding_score + (1 - self.alpha) * popularity_score
        
        return final_score
