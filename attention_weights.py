from models.attention_recommender import AttentionBasedRecommender

model = AttentionBasedRecommender(
    num_epochs=50,
    learning_rate=0.001
)
model.fit(train_set)

# Voir les poids d'attention
items, weights = model.get_attention_weights(user_idx=0)
for item, w in zip(items, weights):
    print(f"Film {item}: importance {w:.3f}")