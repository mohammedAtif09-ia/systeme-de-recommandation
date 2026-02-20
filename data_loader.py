"""
Script pour charger et préparer les données MovieLens avec métadonnées textuelles
"""
import cornac
from cornac.datasets import movielens
import pandas as pd
import numpy as np
import os
import urllib.request
import zipfile
from typing import Tuple, Dict


def download_and_extract_movielens(variant='100K'):
    """
    Télécharge et extrait les données MovieLens si nécessaire
    
    Args:
        variant: '100K' ou '1M'
    
    Returns:
        Chemin vers le dossier extrait
    """
    cache_dir = os.path.expanduser('~/.cornac')
    
    if variant == '100K':
        folder_name = 'ml-100k'
        url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    else:
        folder_name = 'ml-1m'
        url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
    
    extract_path = os.path.join(cache_dir, folder_name)
    
    # Vérifier si les fichiers existent déjà
    if variant == '100K':
        check_file = os.path.join(extract_path, 'u.item')
    else:
        check_file = os.path.join(extract_path, 'movies.dat')
    
    if not os.path.exists(check_file):
        print(f"Téléchargement de MovieLens {variant}...")
        zip_path = os.path.join(cache_dir, f'{folder_name}.zip')
        os.makedirs(cache_dir, exist_ok=True)
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
        print("Extraction terminée!")
    
    return extract_path


def load_movies_metadata(variant='100K'):
    """
    Charge les métadonnées des films (titres, genres) depuis les fichiers MovieLens
    
    Args:
        variant: '100K' ou '1M'
    
    Returns:
        Dict mapping item_id -> texte descriptif
    """
    data_path = download_and_extract_movielens(variant)
    
    item_texts = {}
    
    if variant == '100K':
        # Format 100K: movie_id|title|release_date|video_release|imdb_url|genres...
        movies_file = os.path.join(data_path, 'u.item')
        with open(movies_file, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    movie_id = parts[0]
                    title = parts[1]
                    # Les colonnes 5+ sont les genres (binaire)
                    genres = []
                    genre_names = ['Unknown', 'Action', 'Adventure', 'Animation', 
                                   'Children', 'Comedy', 'Crime', 'Documentary',
                                   'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                                   'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                                   'Thriller', 'War', 'Western']
                    if len(parts) > 5:
                        for i, g in enumerate(parts[5:]):
                            if g == '1' and i < len(genre_names):
                                genres.append(genre_names[i])
                    
                    # Créer le texte descriptif
                    genre_text = ', '.join(genres) if genres else 'Unknown genre'
                    item_texts[movie_id] = f"{title}. Genres: {genre_text}"
    else:
        # Format 1M: MovieID::Title::Genres
        movies_file = os.path.join(data_path, 'movies.dat')
        with open(movies_file, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('::')
                if len(parts) >= 3:
                    movie_id = parts[0]
                    title = parts[1]
                    genres = parts[2].replace('|', ', ')
                    item_texts[movie_id] = f"{title}. Genres: {genres}"
    
    return item_texts


def load_movielens_data(variant='100K'):
    """
    Charge les données MovieLens avec les informations textuelles des films
    
    Args:
        variant: Version de MovieLens ('100K', '1M')
    
    Returns:
        Tuple contenant (ratings_data, item_texts)
    """
    print(f"Chargement de MovieLens {variant}...")
    
    # Charger les ratings via Cornac
    ml = movielens.load_feedback(variant=variant)
    
    # Charger les métadonnées des films directement
    item_texts = load_movies_metadata(variant)
    
    print(f"Données chargées: {len(ml)} interactions, {len(item_texts)} items avec texte")
    
    return ml, item_texts


def prepare_cornac_dataset(ratings_data, item_texts, test_size=0.2, seed=42):
    """
    Prépare les données pour Cornac avec split train/test
    
    Args:
        ratings_data: Données de ratings
        item_texts: Dictionnaire des textes des items
        test_size: Proportion du test set
        seed: Random seed pour la reproductibilité
    
    Returns:
        train_set, test_set, item_texts_dict
    """
    from cornac.eval_methods import RatioSplit
    
    print("Préparation du dataset avec split train/test...")
    
    # Créer le split train/test
    ratio_split = RatioSplit(
        data=ratings_data,
        test_size=test_size,
        rating_threshold=3.5,  # Considérer comme positive si rating >= 3.5
        exclude_unknowns=True,
        verbose=True,
        seed=seed
    )
    
    # Construire les datasets
    train_set = ratio_split.train_set
    test_set = ratio_split.test_set
    
    # Mapper les IDs des items avec leurs textes
    item_id_map = train_set.iid_map
    
    # Créer un dictionnaire texte aligné avec les IDs internes de Cornac
    aligned_item_texts = {}
    for external_id, internal_id in item_id_map.items():
        if external_id in item_texts:
            aligned_item_texts[internal_id] = item_texts[external_id]
    
    print(f"Train set: {train_set.num_users} users, {train_set.num_items} items")
    print(f"Test set: {test_set.num_users} users, {test_set.num_items} items")
    print(f"Items avec texte alignés: {len(aligned_item_texts)}")
    
    return train_set, test_set, aligned_item_texts


def get_item_features_matrix(item_texts_dict, max_items):
    """
    Retourne une matrice placeholder pour les features des items
    Cette fonction sera utilisée par le modèle d'embeddings
    
    Args:
        item_texts_dict: Dictionnaire des textes des items
        max_items: Nombre maximum d'items
    
    Returns:
        Dictionary mapping item_id -> text
    """
    item_features = {}
    for item_id in range(max_items):
        if item_id in item_texts_dict:
            item_features[item_id] = item_texts_dict[item_id]
        else:
            item_features[item_id] = ""  # Texte vide pour les items sans description
    
    return item_features


if __name__ == "__main__":
    # Test du chargement des données (1M car déjà téléchargé)
    ratings, texts = load_movielens_data('1M')
    
    # Debug: afficher les premiers IDs pour comprendre le format
    print("\n=== Debug: Format des IDs ===")
    print(f"Premiers IDs dans item_texts: {list(texts.keys())[:5]}")
    print(f"Type des IDs: {type(list(texts.keys())[0])}")
    print(f"Exemple de rating (user, item, rating): {ratings[0]}")
    print(f"Type de l'item ID dans rating: {type(ratings[0][1])}")
    print(f"Exemples de textes: ")
    for k, v in list(texts.items())[:2]:
        print(f"  ID {k}: {v[:80]}...")
    
    train, test, aligned_texts = prepare_cornac_dataset(ratings, texts)
    
    print("\n=== Exemple de données ===")
    print(f"Nombre total d'interactions train: {train.num_ratings}")
    print(f"Nombre total d'interactions test: {test.num_ratings}")
    
    # Afficher quelques exemples de textes
    print("\n=== Exemples de textes d'items ===")
    for i, (item_id, text) in enumerate(list(aligned_texts.items())[:3]):
        print(f"\nItem {item_id}:")
        print(f"  {text[:200]}...")
