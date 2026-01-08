"""
Script pour visualiser et comparer les résultats d'évaluation
"""
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime


def load_latest_results(results_dir='results'):
    """Charge le fichier de résultats le plus récent"""
    if not os.path.exists(results_dir):
        print(f"Le dossier {results_dir} n'existe pas encore.")
        return None
    
    # Trouver tous les fichiers de résultats
    result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    if not result_files:
        print(f"Aucun fichier de résultats trouvé dans {results_dir}")
        return None
    
    # Trier par date et prendre le plus récent
    result_files.sort(reverse=True)
    latest_file = os.path.join(results_dir, result_files[0])
    
    print(f"Chargement des résultats depuis: {latest_file}")
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    return results


def plot_comparison(results, output_dir='results'):
    """Crée des graphiques comparatifs des modèles"""
    
    if results is None:
        print("Aucun résultat à visualiser.")
        return
    
    # Préparer les données
    models = []
    recall_10 = []
    ndcg_10 = []
    precision_10 = []
    times = []
    
    for model_name, metrics in results.items():
        if 'error' not in metrics:
            models.append(model_name)
            recall_10.append(metrics.get('recall@10', 0))
            ndcg_10.append(metrics.get('ndcg@10', 0))
            precision_10.append(metrics.get('precision@10', 0))
            times.append(metrics.get('total_time', 0))
    
    if not models:
        print("Aucune donnée valide à visualiser.")
        return
    
    # Configurer le style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # Créer une figure avec 4 subplots
    fig, axes = plt.subplots(2, 2)
    fig.suptitle('Comparaison des Modèles de Recommandation', fontsize=16, fontweight='bold')
    
    # 1. Recall@10
    ax1 = axes[0, 0]
    bars1 = ax1.bar(models, recall_10, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax1.set_title('Recall@10', fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.set_ylim([0, max(recall_10) * 1.2])
    ax1.tick_params(axis='x', rotation=45)
    
    # Ajouter les valeurs sur les barres
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)
    
    # 2. NDCG@10
    ax2 = axes[0, 1]
    bars2 = ax2.bar(models, ndcg_10, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax2.set_title('NDCG@10', fontweight='bold')
    ax2.set_ylabel('Score')
    ax2.set_ylim([0, max(ndcg_10) * 1.2])
    ax2.tick_params(axis='x', rotation=45)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)
    
    # 3. Precision@10
    ax3 = axes[1, 0]
    bars3 = ax3.bar(models, precision_10, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax3.set_title('Precision@10', fontweight='bold')
    ax3.set_ylabel('Score')
    ax3.set_ylim([0, max(precision_10) * 1.2])
    ax3.tick_params(axis='x', rotation=45)
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)
    
    # 4. Temps d'exécution
    ax4 = axes[1, 1]
    bars4 = ax4.bar(models, times, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax4.set_title('Temps d\'exécution (secondes)', fontweight='bold')
    ax4.set_ylabel('Temps (s)')
    ax4.set_ylim([0, max(times) * 1.2])
    ax4.tick_params(axis='x', rotation=45)
    
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Sauvegarder la figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'comparison_{timestamp}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Graphique sauvegardé: {output_path}")
    
    plt.show()
    
    # Créer un graphique radar pour comparer les métriques
    fig2, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    # Normaliser les métriques entre 0 et 1 pour le radar
    max_recall = max(recall_10) if max(recall_10) > 0 else 1
    max_ndcg = max(ndcg_10) if max(ndcg_10) > 0 else 1
    max_prec = max(precision_10) if max(precision_10) > 0 else 1
    
    categories = ['Recall@10', 'NDCG@10', 'Precision@10']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Compléter le cercle
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, model in enumerate(models):
        values = [
            recall_10[i] / max_recall,
            ndcg_10[i] / max_ndcg,
            precision_10[i] / max_prec
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax.set_ylim(0, 1)
    ax.set_title('Comparaison Radar des Métriques (Normalisées)', 
                 size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    # Sauvegarder
    radar_path = os.path.join(output_dir, f'radar_comparison_{timestamp}.png')
    plt.savefig(radar_path, dpi=300, bbox_inches='tight')
    print(f"✅ Graphique radar sauvegardé: {radar_path}")
    
    plt.show()


def print_summary(results):
    """Affiche un résumé textuel des résultats"""
    if results is None:
        return
    
    print("\n" + "="*80)
    print("RÉSUMÉ DES RÉSULTATS")
    print("="*80)
    
    # Trouver les meilleurs modèles
    valid_models = {k: v for k, v in results.items() if 'error' not in v}
    
    if not valid_models:
        print("Aucun résultat valide.")
        return
    
    # Meilleurs modèles par métrique
    best_recall = max(valid_models.items(), key=lambda x: x[1].get('recall@10', 0))
    best_ndcg = max(valid_models.items(), key=lambda x: x[1].get('ndcg@10', 0))
    best_precision = max(valid_models.items(), key=lambda x: x[1].get('precision@10', 0))
    
    print(f"\n🏆 MEILLEURS MODÈLES:")
    print(f"  - Recall@10:    {best_recall[0]:<25} ({best_recall[1]['recall@10']:.4f})")
    print(f"  - NDCG@10:      {best_ndcg[0]:<25} ({best_ndcg[1]['ndcg@10']:.4f})")
    print(f"  - Precision@10: {best_precision[0]:<25} ({best_precision[1]['precision@10']:.4f})")
    
    # Comparaison avec baseline
    if 'MatrixFactorization' in valid_models and 'EmbeddingRecommender' in valid_models:
        mf = valid_models['MatrixFactorization']
        emb = valid_models['EmbeddingRecommender']
        
        recall_improvement = ((emb['recall@10'] - mf['recall@10']) / mf['recall@10']) * 100
        ndcg_improvement = ((emb['ndcg@10'] - mf['ndcg@10']) / mf['ndcg@10']) * 100
        
        print(f"\n📊 AMÉLIORATION vs Matrix Factorization:")
        print(f"  - Recall@10: {recall_improvement:+.2f}%")
        print(f"  - NDCG@10:   {ndcg_improvement:+.2f}%")
    
    print("\n" + "="*80)


def main():
    """Fonction principale"""
    print("Visualisation des résultats d'évaluation")
    print("="*80)
    
    # Charger les résultats
    results = load_latest_results()
    
    if results:
        # Afficher le résumé
        print_summary(results)
        
        # Créer les graphiques
        print("\nGénération des graphiques...")
        plot_comparison(results)
        
        print("\n✅ Visualisation terminée!")
    else:
        print("\n❌ Aucun résultat à visualiser.")
        print("Exécutez d'abord 'python evaluate.py' pour générer les résultats.")


if __name__ == "__main__":
    main()
