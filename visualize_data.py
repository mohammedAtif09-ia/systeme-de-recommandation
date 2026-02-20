"""
Script pour visualiser et explorer les données MovieLens
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_movielens_data, prepare_cornac_dataset

# Configuration
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)

print("="*80)
print("EXPLORATION DES DONNÉES MOVIELENS")
print("="*80)

# 1. Charger les données
print("\n1️⃣  Chargement des données...")
ratings, item_texts = load_movielens_data('100K')

print(f"\n📊 Statistiques générales:")
print(f"  - Nombre total d'interactions: {len(ratings)}")
print(f"  - Nombre d'items avec texte: {len(item_texts)}")

# 2. Afficher des exemples de textes
print("\n2️⃣  Exemples de descriptions de films:\n")
for i, (item_id, text) in enumerate(list(item_texts.items())[:5]):
    title = text.split('.')[0] if '.' in text else text[:50]
    description = text[len(title):200] if len(text) > len(title) else ""
    print(f"Film {i+1}: {title}")
    print(f"  Description: {description}...")
    print()

# 3. Préparer le dataset
print("\n3️⃣  Préparation du split train/test...")
train_set, test_set, aligned_texts = prepare_cornac_dataset(
    ratings, 
    item_texts,
    test_size=0.2,
    seed=42
)

# 4. Statistiques détaillées
print("\n4️⃣  Statistiques du dataset:\n")

# Stats train
print(f"📈 Train Set:")
print(f"  - Utilisateurs: {train_set.num_users}")
print(f"  - Items: {train_set.num_items}")
print(f"  - Interactions: {train_set.num_ratings}")
print(f"  - Densité: {(train_set.num_ratings / (train_set.num_users * train_set.num_items)) * 100:.2f}%")

# Stats test
print(f"\n📉 Test Set:")
print(f"  - Utilisateurs: {test_set.num_users}")
print(f"  - Items: {test_set.num_items}")
print(f"  - Interactions: {test_set.num_ratings}")
print(f"  - Densité: {(test_set.num_ratings / (test_set.num_users * test_set.num_items)) * 100:.2f}%")

# 5. Distribution des ratings par utilisateur
print("\n5️⃣  Calcul des distributions...")
user_ratings_count = []
user_avg_rating = []

for user_idx in range(train_set.num_users):
    user_items, user_ratings = train_set.user_data[user_idx]
    user_ratings_count.append(len(user_items))
    if len(user_ratings) > 0:
        user_avg_rating.append(np.mean(user_ratings))

print(f"\n👥 Statistiques Utilisateurs:")
print(f"  - Moyenne de ratings par utilisateur: {np.mean(user_ratings_count):.2f}")
print(f"  - Médiane: {np.median(user_ratings_count):.0f}")
print(f"  - Min: {np.min(user_ratings_count)}")
print(f"  - Max: {np.max(user_ratings_count)}")
print(f"  - Rating moyen: {np.mean(user_avg_rating):.2f}")

# 6. Distribution des ratings par item
item_ratings_count = [0] * train_set.num_items
item_avg_rating = {}

for user_idx in range(train_set.num_users):
    user_items, user_ratings = train_set.user_data[user_idx]
    for item, rating in zip(user_items, user_ratings):
        item_ratings_count[item] += 1
        if item not in item_avg_rating:
            item_avg_rating[item] = []
        item_avg_rating[item].append(rating)

item_ratings_count = [c for c in item_ratings_count if c > 0]
item_avg_rating_values = [np.mean(ratings) for ratings in item_avg_rating.values()]

print(f"\n🎬 Statistiques Items:")
print(f"  - Moyenne de ratings par item: {np.mean(item_ratings_count):.2f}")
print(f"  - Médiane: {np.median(item_ratings_count):.0f}")
print(f"  - Min: {np.min(item_ratings_count)}")
print(f"  - Max: {np.max(item_ratings_count)}")
print(f"  - Rating moyen: {np.mean(item_avg_rating_values):.2f}")

# 7. Top items les plus populaires
print("\n6️⃣  Top 10 films les plus populaires:")
item_popularity = [(item_id, len(ratings)) for item_id, ratings in item_avg_rating.items()]
item_popularity.sort(key=lambda x: x[1], reverse=True)

for rank, (item_id, count) in enumerate(item_popularity[:10], 1):
    if item_id in aligned_texts:
        title = aligned_texts[item_id].split('.')[0]
        avg_rating = np.mean(item_avg_rating[item_id])
        print(f"  {rank:2d}. {title[:50]:<50} ({count:3d} ratings, avg: {avg_rating:.2f})")

# 8. Visualisations
print("\n7️⃣  Génération des visualisations...")

fig = plt.figure(figsize=(16, 10))

# Distribution des ratings par utilisateur
ax1 = plt.subplot(2, 3, 1)
plt.hist(user_ratings_count, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('Nombre de ratings par utilisateur', fontsize=10)
plt.ylabel('Nombre d\'utilisateurs', fontsize=10)
plt.title('Distribution des ratings par utilisateur', fontsize=12, fontweight='bold')
plt.axvline(np.mean(user_ratings_count), color='red', linestyle='--', label=f'Moyenne: {np.mean(user_ratings_count):.1f}')
plt.legend()
plt.grid(alpha=0.3)

# Box plot utilisateurs
ax2 = plt.subplot(2, 3, 2)
plt.boxplot(user_ratings_count, vert=True)
plt.ylabel('Nombre de ratings', fontsize=10)
plt.title('Box plot - Ratings par utilisateur', fontsize=12, fontweight='bold')
plt.grid(alpha=0.3, axis='y')

# Distribution des ratings moyens par utilisateur
ax3 = plt.subplot(2, 3, 3)
plt.hist(user_avg_rating, bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
plt.xlabel('Rating moyen', fontsize=10)
plt.ylabel('Nombre d\'utilisateurs', fontsize=10)
plt.title('Distribution des ratings moyens (utilisateurs)', fontsize=12, fontweight='bold')
plt.axvline(np.mean(user_avg_rating), color='red', linestyle='--', label=f'Moyenne: {np.mean(user_avg_rating):.2f}')
plt.legend()
plt.grid(alpha=0.3)

# Distribution des ratings par item
ax4 = plt.subplot(2, 3, 4)
plt.hist(item_ratings_count, bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
plt.xlabel('Nombre de ratings par item', fontsize=10)
plt.ylabel('Nombre d\'items', fontsize=10)
plt.title('Distribution des ratings par item', fontsize=12, fontweight='bold')
plt.axvline(np.mean(item_ratings_count), color='red', linestyle='--', label=f'Moyenne: {np.mean(item_ratings_count):.1f}')
plt.legend()
plt.grid(alpha=0.3)

# Top 20 items les plus populaires
ax5 = plt.subplot(2, 3, 5)
top_20_items = item_popularity[:20]
top_20_counts = [count for _, count in top_20_items]
top_20_labels = [f"Item {item_id}" for item_id, _ in top_20_items]
plt.barh(range(20), top_20_counts, color='orange', alpha=0.7)
plt.xlabel('Nombre de ratings', fontsize=10)
plt.ylabel('Items', fontsize=10)
plt.title('Top 20 items les plus populaires', fontsize=12, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(alpha=0.3, axis='x')

# Heatmap de sparsité (échantillon)
ax6 = plt.subplot(2, 3, 6)
sample_size = min(100, train_set.num_users)
sample_matrix = np.zeros((sample_size, sample_size))
for user_idx in range(sample_size):
    user_items, user_ratings = train_set.user_data[user_idx]
    for item in user_items:
        if item < sample_size:
            sample_matrix[user_idx, item] = 1

plt.imshow(sample_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
plt.colorbar(label='Interaction')
plt.xlabel('Items (échantillon)', fontsize=10)
plt.ylabel('Utilisateurs (échantillon)', fontsize=10)
plt.title(f'Heatmap de sparsité ({sample_size}x{sample_size})', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('results/data_exploration.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Visualisations sauvegardées dans: results/data_exploration.png")
plt.show()

print("\n" + "="*80)
print("✅ EXPLORATION TERMINÉE")
print("="*80)
print("\nPour lancer l'évaluation complète des modèles, exécutez:")
print("  python evaluate.py")
