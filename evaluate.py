"""
Script d'évaluation et de comparaison des modèles de recommandation
Compare le modèle basé sur embeddings avec Matrix Factorization
Métriques: Recall@10 et NDCG@10
"""
import numpy as np
import cornac
from cornac.models import MF, BPR, NMF
from cornac.eval_methods import RatioSplit
from cornac.metrics import Recall, NDCG, Precision, AUC
import time
import json
import os
from datetime import datetime

# Import local
from data_loader import load_movielens_data, prepare_cornac_dataset
from models.embedding_recommender import EmbeddingBasedRecommender, HybridEmbeddingRecommender
from models.attention_recommender_v3 import ContrastiveAttentionRecommender


class TextModality:
    """
    Classe helper pour passer les textes des items à Cornac
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, id_map, **kwargs):
        """Build the modality"""
        return self
        

def add_item_texts_to_dataset(train_set, test_set, item_texts):
    """
    Ajoute les textes des items aux datasets Cornac
    
    Args:
        train_set: Dataset d'entraînement
        test_set: Dataset de test
        item_texts: Dictionnaire des textes des items (avec IDs internes)
    
    Returns:
        train_set, test_set avec attribut item_text
    """
    train_set.item_text = item_texts
    test_set.item_text = item_texts
    
    return train_set, test_set


def evaluate_models(train_set, test_set, item_texts, save_results=True):
    """
    Évalue et compare différents modèles de recommandation
    
    Args:
        train_set: Dataset d'entraînement
        test_set: Dataset de test
        item_texts: Textes des items
        save_results: Si True, sauvegarde les résultats
    
    Returns:
        Dictionnaire contenant les résultats de tous les modèles
    """
    
    # Ajouter les textes aux datasets
    train_set, test_set = add_item_texts_to_dataset(train_set, test_set, item_texts)
    
    # Définir les métriques d'évaluation
    metrics = [
        Recall(k=10),
        NDCG(k=10),
        Precision(k=10),
        Recall(k=20),
        NDCG(k=20),
    ]
    
    # Définir les modèles à comparer
    models = [
        # Modèle baseline: Matrix Factorization
        MF(
            k=50,
            max_iter=100,
            learning_rate=0.01,
            lambda_reg=0.02,
            use_bias=True,
            verbose=True,
            seed=42,
            name="MatrixFactorization"
        ),
        
        # # Modèle baseline: BPR (Bayesian Personalized Ranking)
        # BPR(
        #     k=50,
        #     max_iter=100,
        #     learning_rate=0.01,
        #     lambda_reg=0.01,
        #     verbose=True,
        #     seed=42,
        #     name="BPR"
        # ),
        
        # # Notre modèle: Embedding-based Recommender
        # EmbeddingBasedRecommender(
        #     name="EmbeddingRecommender",
        #     model_name='all-MiniLM-L6-v2',  # Modèle BERT léger et performant
        #     verbose=True,
        # ),
        
        # # Variante hybride
        # HybridEmbeddingRecommender(
        #     name="HybridEmbedding",
        #     model_name='all-MiniLM-L6-v2',
        #     alpha=0.7,  # 70% embeddings, 30% popularité
        #     verbose=True,
        # ),
        
        # INNOVATION: Contrastive Attention avec BPR Loss
        ContrastiveAttentionRecommender(
            name="ContrastiveAttention",
            model_name='all-MiniLM-L6-v2',
            # num_epochs=20,  # Ancienne valeur
            num_epochs=40,  # Nouvelle valeur: meilleure convergence
            # num_negatives=5,  # Ancienne valeur
            num_negatives=10,  # Nouvelle valeur: meilleur contraste
            # learning_rate=0.001,  # Ancienne valeur
            learning_rate=0.002,  # Nouvelle valeur: convergence plus rapide
            verbose=True,
        ),
    ]
    
    print("="*80)
    print("ÉVALUATION DES MODÈLES DE RECOMMANDATION")
    print("="*80)
    print(f"\nDataset:")
    print(f"  - Train: {train_set.num_users} users, {train_set.num_items} items, {train_set.num_ratings} ratings")
    print(f"  - Test: {test_set.num_users} users, {test_set.num_items} items, {test_set.num_ratings} ratings")
    print(f"\nMétriques: Recall@10, NDCG@10, Precision@10, Recall@20, NDCG@20")
    print(f"Modèles: {len(models)}")
    
    results = {}
    
    # Évaluer chaque modèle
    for model in models:
        print("\n" + "="*80)
        print(f"Entraînement et évaluation: {model.name}")
        print("="*80)
        
        start_time = time.time()
        
        try:
            # Entraîner le modèle
            print(f"\n[{model.name}] Début de l'entraînement...")
            model.fit(train_set)
            training_time = time.time() - start_time
            print(f"[{model.name}] Entraînement terminé en {training_time:.2f}s")
            
            # Évaluer sur le test set
            print(f"[{model.name}] Évaluation en cours...")
            eval_start = time.time()
            
            # Utiliser la méthode d'évaluation correcte de Cornac
            from cornac.eval_methods import rating_eval, ranking_eval
            
            # Évaluation ranking (Recall, NDCG, Precision)
            model_results = {}
            
            # Calculer les métriques de ranking
            metric_values, _ = ranking_eval(
                model=model,
                metrics=metrics,
                train_set=train_set,
                test_set=test_set,
                verbose=False
            )
            
            # Stocker les résultats
            for i, metric in enumerate(metrics):
                model_results[metric.name] = metric_values[i]
            
            eval_time = time.time() - eval_start
            total_time = time.time() - start_time
            
            # Stocker les résultats
            results[model.name] = {
                'metrics': model_results,
                'training_time': training_time,
                'eval_time': eval_time,
                'total_time': total_time,
            }
            
            # Afficher les résultats
            print(f"\n[{model.name}] Résultats:")
            print(f"  - Temps d'entraînement: {training_time:.2f}s")
            print(f"  - Temps d'évaluation: {eval_time:.2f}s")
            print(f"  - Temps total: {total_time:.2f}s")
            print(f"\n  Métriques:")
            for metric_name, value in model_results.items():
                print(f"    - {metric_name}: {value:.4f}")
        
        except Exception as e:
            print(f"\n[ERREUR] Échec pour {model.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[model.name] = {'error': str(e)}
    
    # Afficher un tableau comparatif
    print("\n" + "="*80)
    print("RÉSULTATS COMPARATIFS")
    print("="*80)
    print(f"\n{'Modèle':<25} {'Recall@10':<12} {'NDCG@10':<12} {'Precision@10':<15} {'Temps (s)':<12}")
    print("-"*80)
    
    for model_name, result in results.items():
        if 'error' not in result:
            recall10 = result['metrics'].get('Recall@10', 0)
            ndcg10 = result['metrics'].get('NDCG@10', 0)
            prec10 = result['metrics'].get('Precision@10', 0)
            time_total = result['total_time']
            print(f"{model_name:<25} {recall10:<12.4f} {ndcg10:<12.4f} {prec10:<15.4f} {time_total:<12.2f}")
        else:
            print(f"{model_name:<25} {'ERREUR':<12} {'ERREUR':<12} {'ERREUR':<15} {'-':<12}")
    
    # Trouver le meilleur modèle
    successful_models = [(name, res['metrics'].get('Recall@10', 0)) for name, res in results.items() if 'error' not in res]
    
    if successful_models:
        best_model_recall = max(successful_models, key=lambda x: x[1])
        best_model_ndcg = max(
            [(name, res['metrics'].get('NDCG@10', 0)) for name, res in results.items() if 'error' not in res],
            key=lambda x: x[1]
        )
        
        print("\n" + "="*80)
        print(f" Meilleur modèle (Recall@10): {best_model_recall[0]} ({best_model_recall[1]:.4f})")
        print(f" Meilleur modèle (NDCG@10): {best_model_ndcg[0]} ({best_model_ndcg[1]:.4f})")
        print("="*80)
    else:
        print("\n" + "="*80)
        print(" ATTENTION: Aucun modèle n'a réussi l'évaluation!")
        print("="*80)
    
    # Sauvegarder les résultats
    if save_results:
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(results_dir, f'evaluation_results_{timestamp}.json')
        
        # Convertir en format JSON-serializable
        json_results = {}
        for model_name, result in results.items():
            if 'error' not in result:
                json_results[model_name] = {
                    'recall@10': float(result['metrics'].get('Recall@10', 0)),
                    'ndcg@10': float(result['metrics'].get('NDCG@10', 0)),
                    'precision@10': float(result['metrics'].get('Precision@10', 0)),
                    'recall@20': float(result['metrics'].get('Recall@20', 0)),
                    'ndcg@20': float(result['metrics'].get('NDCG@20', 0)),
                    'training_time': float(result['training_time']),
                    'eval_time': float(result['eval_time']),
                    'total_time': float(result['total_time']),
                }
            else:
                json_results[model_name] = {'error': result['error']}
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n Résultats sauvegardés dans: {results_file}")
    
    return results


def main():
    """Fonction principale"""
    print("Chargement des données...")
    
    # Charger les données MovieLens (1M car déjà téléchargé)
    ratings, item_texts = load_movielens_data('1M')
    
    # Préparer les datasets train/test
    train_set, test_set, aligned_item_texts = prepare_cornac_dataset(
        ratings, 
        item_texts,
        test_size=0.2,
        seed=42
    )
    
    # Évaluer les modèles
    results = evaluate_models(train_set, test_set, aligned_item_texts, save_results=True)
    
    print("\n Évaluation terminée!")
    
    return results


if __name__ == "__main__":
    results = main()
