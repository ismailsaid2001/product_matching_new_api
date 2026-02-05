import pandas as pd
import os

def filter_labeled_products():
    """
    Filtre labeled_products.csv pour EXCLURE les paires (description_cleaned, nature_product)
    où le nature_product n'existe QUE dans validation_set.csv (pas ailleurs dans labeled_products)
    """
    
    # Vérifier l'existence des fichiers
    labeled_file = "..\\data\\labeled_products.csv"
    validation_file = "..\\data\\validation_set.csv"
    
    if not os.path.exists(labeled_file):
        print(f"Fichier {labeled_file} non trouvé")
        return
    
    if not os.path.exists(validation_file):
        print(f"Fichier {validation_file} non trouvé")
        return
    
    print("Chargement des fichiers...")
    
    # Charger validation_set pour identifier les IDs et nature_products "exclusifs"
    validation_df = pd.read_csv(validation_file)
    
    # Charger nature_product.csv pour les correspondances
    print("Chargement de nature_product.csv...")
    try:
        nature_product_df = pd.read_csv('..\\data\\nature_product.csv')
        # Joindre validation_set avec nature_product pour obtenir les nature_products
        validation_with_nature = validation_df.merge(
            nature_product_df[['id', 'nature_product']], 
            left_on='nature_product_id', 
            right_on='id', 
            how='left'
        )
    except Exception as e:
        print(f"Erreur lors du chargement de nature_product.csv: {e}")
        return
    
    # Obtenir les IDs présents dans validation_set
    validation_ids = set(validation_df['id'].astype(str))
    print(f"Validation set chargé: {len(validation_df)} lignes, {len(validation_ids)} IDs uniques")
    
    # Charger labeled_products
    labeled_df = pd.read_csv(labeled_file)
    print(f"Labeled products chargé: {len(labeled_df)} lignes")
    print(f"Colonnes disponibles: {list(labeled_df.columns)}")
    
    # Identifier les nature_products qui n'existent QUE dans validation_set
    # 1. Nature_products présents dans validation_set
    nature_products_in_validation = set(validation_with_nature['nature_product'].dropna())
    
    # 2. Nature_products présents dans labeled_products HORS validation_set
    labeled_df['id'] = labeled_df['id'].astype(str)
    labeled_outside_validation = labeled_df[~labeled_df['id'].isin(validation_ids)]
    nature_products_outside_validation = set(labeled_outside_validation['nature_product'].dropna())
    
    # 3. Nature_products EXCLUSIFS à validation_set (présents dans validation mais PAS ailleurs)
    exclusive_nature_products = nature_products_in_validation - nature_products_outside_validation
    
    print(f"Nature_products dans validation_set: {len(nature_products_in_validation)}")
    print(f"Nature_products hors validation_set: {len(nature_products_outside_validation)}")
    print(f"Nature_products EXCLUSIFS à validation_set: {len(exclusive_nature_products)}")
    print(f"Exemples d'exclusifs: {list(exclusive_nature_products)[:5]}")
    
    # Vérifier la présence des colonnes nécessaires
    required_columns = ['description_cleaned', 'nature_product']
    missing_columns = [col for col in required_columns if col not in labeled_df.columns]
    
    if missing_columns:
        print(f"Colonnes manquantes dans labeled_products.csv: {missing_columns}")
        return
    
    if 'id' not in labeled_df.columns:
        print("Colonne 'id' manquante dans labeled_products.csv")
        return
    
    # EXCLURE les nature_products qui sont exclusifs à validation_set
    filtered_df = labeled_df[~labeled_df['nature_product'].isin(exclusive_nature_products)].copy()
    print(f"Après exclusion des nature_products exclusifs: {len(filtered_df)} lignes conservées")
    print(f"Nature_products uniques conservés: {filtered_df['nature_product'].nunique()}")
    
    # Garder seulement les colonnes spécifiées
    final_df = filtered_df[['description_cleaned', 'nature_product']].copy()
    
    # Nettoyer les données (supprimer les valeurs nulles ou vides)
    initial_count = len(final_df)
    final_df = final_df.dropna()
    final_df = final_df[final_df['description_cleaned'].str.strip() != '']
    final_df = final_df[final_df['nature_product'].str.strip() != '']
    final_count = len(final_df)
    
    print(f"Après nettoyage: {final_count} lignes (supprimé {initial_count - final_count} lignes vides)")
    
    # Supprimer les doublons
    final_df = final_df.drop_duplicates()
    dedupe_count = len(final_df)
    print(f"Après déduplication: {dedupe_count} lignes uniques")
    
    # Sauvegarder le résultat
    output_file = "labeled_products_filtered.csv"
    final_df.to_csv(output_file, index=False)
    print(f"Fichier sauvegardé: {output_file}")
    
    # Afficher quelques statistiques
    print("\nSTATISTIQUES:")
    print(f"Total lignes filtrées: {len(final_df)}")
    print(f"Nature_products uniques: {final_df['nature_product'].nunique()}")
    
    # Afficher un aperçu
    print("\nAPERÇU DU FICHIER FILTRÉ:")
    print(final_df.head(10).to_string())
    
    return final_df

if __name__ == "__main__":
    result = filter_labeled_products()