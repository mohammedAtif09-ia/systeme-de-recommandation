"""
Script de démonstration - Exemple d'utilisation du système de recommandation
"""
from data_loader import load_movielens_data, prepare_cornac_dataset
from models.embedding_recommender import EmbeddingBasedRecommender
import numpy as np


def demo_recommendations():
    """Démonstration des recommandations pour quelques utilisateurs"""
    
    print("="*80)
    print("DÉMONSTRATION DU SYSTÈME DE RECOMMANDATION")
    print("="*80)
    
    # Charger les données
    print("\n1. Chargement des données MovieLens...")
    ratings, item_texts = load_movielens_data('100K')
    train_set, test_set, aligned_texts = prepare_cornac_dataset(ratings, item_texts)
    
    # Ajouter les textes
    train_set.item_text = aligned_texts
    
    # Créer et entraîner le modèle
    print("\n2. Création et entraînement du modèle...")
    model = EmbeddingBasedRecommender(
        name="DemoModel",
        model_name='all-MiniLM-L6-v2',
        verbose=True
    )
    
    model.fit(train_set)
    
    # Faire des recommandations pour quelques utilisateurs
    print("\n3. Génération de recommandations pour des utilisateurs exemples...")
    
    num_demo_users = 3
    for user_idx in range(min(num_demo_users, train_set.num_users)):
        print("\n" + "-"*80)
        print(f"UTILISATEUR {user_idx}")
        print("-"*80)
        
        # Récupérer les items déjà consommés
        user_items, user_ratings = train_set.user_data[user_idx]
        
        print(f"\nItems déjà vus par l'utilisateur ({len(user_items)} films):")
        for i, (item_id, rating) in enumerate(zip(user_items[:5], user_ratings[:5])):
            if item_id in aligned_texts:
                title = aligned_texts[item_id].split('.')[0]  # Extraire le titre
                print(f"  {i+1}. {title[:60]}... (rating: {rating:.1f})")
        
        if len(user_items) > 5:
            print(f"  ... et {len(user_items) - 5} autres films")
        
        # Générer les recommandations
        scores = model.score(user_idx)
        
        # Masquer les items déjà vus
        scores[user_items] = -np.inf
        
        # Top 10 recommandations
        top_10_items = scores.argsort()[-10:][::-1]
        
        print(f"\nTop 10 recommandations:")
        for i, item_id in enumerate(top_10_items, 1):
            if item_id in aligned_texts:
                title = aligned_texts[item_id].split('.')[0]
                score = scores[item_id]
                print(f"  {i}. {title[:60]}... (score: {score:.4f})")
            else:
                print(f"  {i}. Item {item_id} (score: {scores[item_id]:.4f})")
    
    print("\n" + "="*80)
    print("Démonstration terminée!")
    print("="*80)


if __name__ == "__main__":
    demo_recommendations()
