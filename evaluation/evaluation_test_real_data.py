import asyncio
import aiohttp
import pandas as pd
import time
from datetime import datetime
import json
from rouge_score import rouge_scorer
import matplotlib
matplotlib.use('Agg')  # Backend sans interface graphique pour Docker
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration API
API_BASE_URL = "http://localhost:8000"

async def classify_single_product(session, product_data):
    """Classify a single product via API"""
    try:
        payload = {
            "designation": product_data["description_cleaned"],
            "product_id": product_data["id"]
        }
        
        start_time = time.time()
        async with session.post(f"{API_BASE_URL}/classify", json=payload) as response:
            if response.status == 200:
                result = await response.json()
                processing_time_ms = (time.time() - start_time) * 1000
                
                return {
                    'id': product_data["id"],
                    'description_cleaned': product_data["description_cleaned"],
                    'expected_nature_product_id': product_data["nature_product_id"], 
                    'expected_nature_product': product_data["nature_product"],
                    'expected_category': product_data.get("category", "Unknown"),
                    'expected_sub_category': product_data.get("sub_category", "Unknown"),
                    'predicted_nature_product': result.get('final_label', 'ERROR'),
                    'database_confidence': result.get('database_confidence', 0.0),
                    'database_prediction': result.get('database_prediction', ''),
                    't5_confidence': result.get('t5_confidence', 0.0),
                    't5_prediction': result.get('t5_prediction', ''),
                    'confidence': result.get('confidence', 0.0),
                    'path_taken': result.get('path_taken', ''),
                    'decision_node': extract_decision_node(result.get('path_taken', '')),
                    'cost_usd': result.get('cost_usd', 0.0),
                    'processing_time_ms': processing_time_ms,
                    'status': 'success'
                }
            else:
                error_text = await response.text()
                return {
                    'id': product_data["id"],
                    'description_cleaned': product_data["description_cleaned"],
                    'expected_nature_product_id': product_data["nature_product_id"],
                    'expected_nature_product': product_data["nature_product"],
                    'predicted_nature_product': 'ERROR',
                    'confidence': 0.0,
                    'status': f'error_http_{response.status}',
                    'error': error_text,
                    'processing_time_ms': 0,
                    'cost_usd': 0.0
                }
    except Exception as e:
        return {
            'id': product_data["id"],
            'description_cleaned': product_data["description_cleaned"],
            'expected_nature_product_id': product_data["nature_product_id"],
            'expected_nature_product': product_data["nature_product"],
            'predicted_nature_product': 'ERROR',
            'confidence': 0.0,
            'status': 'error_exception',
            'error': str(e),
            'processing_time_ms': 0,
            'cost_usd': 0.0
        }

def extract_decision_node(path_taken):
    """Extract which node made the final decision from path_taken"""
    if isinstance(path_taken, str):
        path_str = path_taken
    else:
        path_str = str(path_taken)
    
    # Si DB a trouvé un match direct
    if "db_match_found" in path_str:
        return "database"
    # Si T5 a pris la décision
    elif "t5_pred_" in path_str and "gpt" not in path_str.lower():
        return "t5"
    # Si GPT/Orchestrator a pris la décision
    elif any(keyword in path_str.lower() for keyword in ["gpt", "orchestrator", "arbitrage"]):
        return "llm"
    # Si incertain, probablement passé par T5 puis LLM
    elif "db_uncertain" in path_str and "t5_pred_" in path_str:
        # Si on voit gpt après t5, c'est LLM qui a décidé, sinon c'est T5
        return "llm" if "gpt" in path_str.lower() else "t5"
    
    else:
        return "unknown"

def load_labeled_products_filtered():
    """Charge le fichier labeled_products_filtered.csv pour classification identification vs création"""
    try:
        df = pd.read_csv('labeled_products_filtered.csv')
        # Normaliser les nature_products (lowercase, strip)
        existing_products = set(df['nature_product'].str.lower().str.strip().unique())
        print(f"labeled_products_filtered.csv chargé: {len(existing_products)} nature_products uniques")
        return existing_products
    except Exception as e:
        print(f"ATTENTION: Impossible de charger labeled_products_filtered.csv: {e}")
        return set()

def load_and_prepare_data(validation_file, nature_product_file, sample_size=None):
    """Load and prepare real data with join between validation and nature_product"""
    print("Chargement des données réelles...")
    
    # Charger validation_set.csv
    validation_df = pd.read_csv(validation_file)
    print(f"Validation set chargé: {len(validation_df)} produits")
    
    # Charger nature_product.csv  
    nature_product_df = pd.read_csv(nature_product_file)
    print(f"Nature products chargé: {len(nature_product_df)} nature_products")
    
    # Faire la jointure sur l'ID
    # Attention: nature_product_id dans validation peut être string, id dans nature_product peut être int
    validation_df['nature_product_id'] = pd.to_numeric(validation_df['nature_product_id'], errors='coerce')
    nature_product_df['id'] = pd.to_numeric(nature_product_df['id'], errors='coerce')
    
    # Jointure
    joined_df = validation_df.merge(
        nature_product_df[['id', 'nature_product', 'nature_product_group', 'category', 'sub_category']], 
        left_on='nature_product_id', 
        right_on='id', 
        how='left',
        suffixes=('', '_nature')
    )
    
    print(f"✅ Jointure effectuée: {len(joined_df)} produits")
    
    # Filtrer les produits sans nature_product valide
    joined_df = joined_df[joined_df['nature_product'].notna()]
    joined_df = joined_df[joined_df['nature_product'] != 'None']
    joined_df = joined_df[joined_df['description_cleaned'].notna()]
    joined_df = joined_df[joined_df['description_cleaned'].str.strip() != '']
    
    print(f"✅ Après filtrage: {len(joined_df)} produits valides")
    
    # Échantillonnage si demandé
    if sample_size and sample_size < len(joined_df):
        joined_df = joined_df.sample(n=sample_size, random_state=74)
        print(f"Échantillon sélectionné: {sample_size} produits")
    
    return joined_df

def calculate_accuracy_metrics(df, existing_products=None):
    """Calculate various accuracy metrics including Exact Match and ROUGE-L with identification vs création"""
    successful_predictions = df[df['status'] == 'success'].copy()
    if len(successful_predictions) == 0:
        metrics = {
            'total_products': len(df),
            'successful_predictions': 0,
            'failed_predictions': len(df),
            'exact_match_count': 0,
            'exact_match_accuracy': 0,
            'rouge_l_mean': 0,
            'rouge_l_scores': [],
            'identification_count': 0,
            'creation_count': 0,
            'identification_accuracy': 0,
            'creation_accuracy': 0
        }
        return metrics, successful_predictions
    
    # Calcul de l'exactitude (exact match)
    successful_predictions['is_exact_match'] = (
        successful_predictions['predicted_nature_product'].str.lower().str.strip() == 
        successful_predictions['expected_nature_product'].str.lower().str.strip()
    )
    
    # Classification identification vs création
    if existing_products:
        successful_predictions['is_identification'] = (
            successful_predictions['predicted_nature_product'].str.lower().str.strip().isin(existing_products)
        )
        successful_predictions['prediction_type'] = successful_predictions['is_identification'].map({
            True: 'identification',
            False: 'creation'
        })
    else:
        successful_predictions['is_identification'] = False
        successful_predictions['prediction_type'] = 'unknown'
    
    # Calcul ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    rouge_l_scores = []
    
    for _, row in successful_predictions.iterrows():
        predicted = str(row['predicted_nature_product']).lower().strip()
        expected = str(row['expected_nature_product']).lower().strip()
        
        # Calculer ROUGE-L
        scores = scorer.score(expected, predicted)
        rouge_l_f1 = scores['rougeL'].fmeasure
        rouge_l_scores.append(rouge_l_f1)
    
    successful_predictions['rouge_l_score'] = rouge_l_scores
    
    # Calcul de moyennes ROUGE-L par seuil
    rouge_l_mean = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0
    rouge_l_threshold_70 = sum(1 for score in rouge_l_scores if score >= 0.7) / len(rouge_l_scores) * 100 if rouge_l_scores else 0
    rouge_l_threshold_80 = sum(1 for score in rouge_l_scores if score >= 0.8) / len(rouge_l_scores) * 100 if rouge_l_scores else 0
    rouge_l_threshold_90 = sum(1 for score in rouge_l_scores if score >= 0.9) / len(rouge_l_scores) * 100 if rouge_l_scores else 0
    
    # Calcul des métriques identification vs création
    if existing_products and len(successful_predictions) > 0:
        identifications = successful_predictions[successful_predictions['prediction_type'] == 'identification']
        creations = successful_predictions[successful_predictions['prediction_type'] == 'creation']
        
        identification_count = len(identifications)
        creation_count = len(creations)
        identification_accuracy = (identifications['is_exact_match'].sum() / len(identifications) * 100) if len(identifications) > 0 else 0
        creation_accuracy = (creations['is_exact_match'].sum() / len(creations) * 100) if len(creations) > 0 else 0
    else:
        identification_count = 0
        creation_count = 0
        identification_accuracy = 0
        creation_accuracy = 0
    
    metrics = {
        'total_products': len(df),
        'successful_predictions': len(successful_predictions),
        'failed_predictions': len(df) - len(successful_predictions),
        'exact_match_count': successful_predictions['is_exact_match'].sum(),
        'exact_match_accuracy': (successful_predictions['is_exact_match'].sum() / len(successful_predictions) * 100) if len(successful_predictions) > 0 else 0,
        'rouge_l_mean': rouge_l_mean,
        'rouge_l_threshold_70': rouge_l_threshold_70,
        'rouge_l_threshold_80': rouge_l_threshold_80, 
        'rouge_l_threshold_90': rouge_l_threshold_90,
        'rouge_l_scores': rouge_l_scores,
        'identification_count': identification_count,
        'creation_count': creation_count,
        'identification_accuracy': identification_accuracy,
        'creation_accuracy': creation_accuracy
    }
    
    return metrics, successful_predictions

async def run_real_data_evaluation(validation_file="..\\data\\validation_set.csv", 
                                  nature_product_file="..\\data\\nature_product.csv",
                                  sample_size=None, 
                                  max_concurrent=5):
    """Run evaluation on real data"""
    
    print("ÉVALUATION SYSTÈME - DONNÉES RÉELLES")
    print("=" * 60)
    
    # Vérification de l'API
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{API_BASE_URL}/health") as response:
                if response.status != 200:
                    raise Exception(f"API non disponible: {response.status}")
        print("API disponible")
    except Exception as e:
        print(f"Erreur API: {e}")
        return
    
    # Charger et préparer les données
    try:
        data_df = load_and_prepare_data(validation_file, nature_product_file, sample_size)
        if len(data_df) == 0:
            print("Aucune donnée valide trouvée")
            return
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        return
    
    print(f"\nTest sur {len(data_df)} produits réels")
    print(f"Concurrence maximum: {max_concurrent} requêtes simultanées")
    print("=" * 60)
    
    # Charger labeled_products_filtered pour classification identification vs création
    existing_products = load_labeled_products_filtered()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.time()
    results = []
    
    # Traitement avec limitation de concurrence
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(session, product_data):
        async with semaphore:
            return await classify_single_product(session, product_data)
    
    async with aiohttp.ClientSession() as session:
        # Convertir DataFrame en liste de dictionnaires
        products_data = data_df.to_dict('records')
        
        # Traiter tous les produits
        tasks = [process_with_semaphore(session, product_data) for product_data in products_data]
        results = await asyncio.gather(*tasks)
    
    # Calculer les totaux
    total_time = time.time() - start_time
    total_cost = sum((r.get('cost_usd') or 0.0) for r in results)
    total_processing_time = sum((r.get('processing_time_ms') or 0.0) for r in results)
    
    # Créer DataFrame des résultats
    results_df = pd.DataFrame(results)
    
    # Nettoyer les colonnes numériques (gérer les colonnes manquantes)
    numeric_columns = ['cost_usd', 'processing_time_ms', 'confidence', 'database_confidence', 't5_confidence']
    for col in numeric_columns:
        if col in results_df.columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce').fillna(0.0)
        else:
            results_df[col] = 0.0
    
    # Calculer les métriques de précision
    metrics, successful_df = calculate_accuracy_metrics(results_df, existing_products)
    
    # Ajouter les colonnes de classification au results_df principal
    if len(successful_df) > 0 and 'prediction_type' in successful_df.columns:
        # Merger les colonnes de classification avec results_df
        classification_cols = successful_df[['id', 'prediction_type', 'is_identification']].copy()
        results_df = results_df.merge(classification_cols, on='id', how='left')
        # Remplir les valeurs manquantes pour les échecs
        results_df['prediction_type'] = results_df['prediction_type'].fillna('unknown')
        results_df['is_identification'] = results_df['is_identification'].fillna(False)
    else:
        results_df['prediction_type'] = 'unknown'
        results_df['is_identification'] = False
    
    # Statistiques par noeud de décision
    if len(successful_df) > 0:
        stats_by_node = successful_df.groupby('decision_node').agg({
            'is_exact_match': ['count', 'sum', 'mean'],
            'rouge_l_score': ['mean', 'std'],
            'confidence': ['mean', 'std'],
            'processing_time_ms': ['mean', 'std'],
            'cost_usd': 'sum'
        }).round(4)
        
        # Statistiques par catégorie
        if 'expected_category' in successful_df.columns:
            stats_by_category = successful_df.groupby('expected_category').agg({
                'is_exact_match': ['count', 'sum', 'mean'],
                'rouge_l_score': ['mean', 'std']
            }).round(4)
    
    # Affichage des résultats
    print("\nRÉSULTATS DE L'ÉVALUATION")
    print("=" * 60)
    print(f"Produits traités: {metrics['total_products']}")
    print(f"Prédictions réussies: {metrics['successful_predictions']}")
    print(f"Prédictions échouées: {metrics['failed_predictions']}")
    print(f"Exact Match: {metrics['exact_match_accuracy']:.2f}% ({metrics['exact_match_count']}/{metrics['successful_predictions']})") 
    print(f"ROUGE-L Moyen: {metrics['rouge_l_mean']:.3f}")
    print(f"ROUGE-L >= 0.7: {metrics['rouge_l_threshold_70']:.1f}% des prédictions")
    print(f"ROUGE-L >= 0.8: {metrics['rouge_l_threshold_80']:.1f}% des prédictions")
    print(f"ROUGE-L >= 0.9: {metrics['rouge_l_threshold_90']:.1f}% des prédictions")
    print(f"")
    print(f"IDENTIFICATION vs CRÉATION")
    print(f"Identifications: {metrics['identification_count']} ({metrics['identification_count']/metrics['successful_predictions']*100:.1f}% des prédictions) - Exact: {metrics['identification_accuracy']:.1f}%")
    print(f"Créations: {metrics['creation_count']} ({metrics['creation_count']/metrics['successful_predictions']*100:.1f}% des prédictions) - Exact: {metrics['creation_accuracy']:.1f}%")
    print(f"Coût total: ${total_cost:.4f}")
    print(f"⏱️  Temps total: {total_time:.2f}s")
    print(f"⚡ Temps moyen par produit: {(total_processing_time/len(results)):.0f}ms")
    
    if len(successful_df) > 0:
        print(f"Confiance moyenne: {successful_df['confidence'].mean():.3f}")
        
        # Calculer les moyennes seulement sur les échantillons pertinents
        db_samples = successful_df[successful_df['database_confidence'] > 0]
        t5_samples = successful_df[successful_df['t5_confidence'] > 0]
        
        if len(db_samples) > 0:
            print(f"Confiance DB moyenne: {db_samples['database_confidence'].mean():.3f} (sur {len(db_samples)} échantillons)")
        
        if len(t5_samples) > 0:
            print(f"Confiance T5 moyenne: {t5_samples['t5_confidence'].mean():.3f} (sur {len(t5_samples)} échantillons)")
        else:
            print(f"Confiance T5 moyenne: N/A (aucun échantillon traité par T5)")
    
    # Sauvegarde des résultats détaillés
    detailed_filename = f"evaluation_real_data_detailed_{timestamp}.csv"
    results_df.to_csv(detailed_filename, index=False)
    print(f"\n💾 Résultats détaillés sauvegardés: {detailed_filename}")
    
    # Sauvegarder les fichiers séparés pour identification vs création
    if 'prediction_type' in results_df.columns:
        # Filtrer les succès seulement pour les fichiers séparés
        successful_only = results_df[results_df['status'] == 'success'].copy()
        
        if len(successful_only) > 0:
            identifications = successful_only[successful_only['prediction_type'] == 'identification']
            creations = successful_only[successful_only['prediction_type'] == 'creation']
            
            if len(identifications) > 0:
                identification_filename = f"evaluation_identifications_{timestamp}.csv"
                identifications.to_csv(identification_filename, index=False)
                print(f"🔍 Identifications sauvegardées: {identification_filename} ({len(identifications)} lignes)")
            
            if len(creations) > 0:
                creation_filename = f"evaluation_creations_{timestamp}.csv"
                creations.to_csv(creation_filename, index=False)
                print(f"🆕 Créations sauvegardées: {creation_filename} ({len(creations)} lignes)")
        else:
            identifications = creations = pd.DataFrame()
    else:
        successful_only = identifications = creations = pd.DataFrame()
    
    # Sauvegarder les fichiers séparés pour identification vs création
    if 'prediction_type' in results_df.columns:
        # Filtrer les succès seulement pour les fichiers séparés
        successful_only = results_df[results_df['status'] == 'success'].copy()
        
        if len(successful_only) > 0:
            identifications = successful_only[successful_only['prediction_type'] == 'identification']
            creations = successful_only[successful_only['prediction_type'] == 'creation']
            
            if len(identifications) > 0:
                identification_filename = f"evaluation_identifications_{timestamp}.csv"
                identifications.to_csv(identification_filename, index=False)
                print(f"🔍 Identifications sauvegardées: {identification_filename} ({len(identifications)} lignes)")
            
            if len(creations) > 0:
                creation_filename = f"evaluation_creations_{timestamp}.csv"
                creations.to_csv(creation_filename, index=False)
                print(f"🆕 Créations sauvegardées: {creation_filename} ({len(creations)} lignes)")
    
    # Sauvegarde du résumé des performances par noeud
    if len(successful_df) > 0:
        summary_filename = f"evaluation_real_data_summary_{timestamp}.csv"
        stats_by_node.to_csv(summary_filename)
        print(f"📊 Résumé des performances sauvegardé: {summary_filename}")
    
    print(f"\n📂 FICHIERS GÉNÉRÉS:")
    print(f"   📄 Tout: {detailed_filename}")
    if 'prediction_type' in results_df.columns and len(successful_only) > 0:
        if len(identifications) > 0:
            print(f"   🔍 Identifications: {identification_filename}")
        if len(creations) > 0:
            print(f"   🆕 Créations: {creation_filename}")
    if len(successful_df) > 0:
        print(f"   📊 Résumé: {summary_filename}")
    
    print(f"\n📂 FICHIERS GÉNÉRÉS:")
    print(f"   📄 Tout: {detailed_filename}")
    if 'prediction_type' in results_df.columns and len(successful_only) > 0:
        if len(identifications) > 0:
            print(f"   🔍 Identifications: {identification_filename}")
        if len(creations) > 0:
            print(f"   🆕 Créations: {creation_filename}")
    if len(successful_df) > 0:
        print(f"   📊 Résumé: {summary_filename}")
    
    print("\n🎯 ANALYSE PAR NOEUD DE DÉCISION")
    print("=" * 60)
    if len(successful_df) > 0:
        for node in successful_df['decision_node'].unique():
            node_data = successful_df[successful_df['decision_node'] == node]
            exact_acc = (node_data['is_exact_match'].sum() / len(node_data)) * 100
            rouge_l_mean = node_data['rouge_l_score'].mean()
            
            # Stats par type de prédiction
            if 'prediction_type' in node_data.columns:
                identifications = node_data[node_data['prediction_type'] == 'identification']
                creations = node_data[node_data['prediction_type'] == 'creation']
                
                print(f"{node.upper():>15}: {len(node_data):>4} total | Exact: {exact_acc:>5.1f}% | ROUGE-L: {rouge_l_mean:>5.3f}")
                if len(identifications) > 0:
                    ident_acc = (identifications['is_exact_match'].sum() / len(identifications)) * 100
                    print(f"{'':>15}  ├─🔍 Ident: {len(identifications):>4} ({len(identifications)/len(node_data)*100:>4.1f}%) | Exact: {ident_acc:>5.1f}%")
                if len(creations) > 0:
                    creat_acc = (creations['is_exact_match'].sum() / len(creations)) * 100
                    print(f"{'':>15}  └─🆕 Créat: {len(creations):>4} ({len(creations)/len(node_data)*100:>4.1f}%) | Exact: {creat_acc:>5.1f}%")
            else:
                print(f"{node.upper():>15}: {len(node_data):>4} predictions | Exact: {exact_acc:>5.1f}% | ROUGE-L: {rouge_l_mean:>5.3f}")
    
    # Analyse des performances brutes du nœud DATABASE
    print("\n📊 ANALYSE PERFORMANCES BRUTES DU NOEUD DATABASE")
    print("=" * 60)
    if len(successful_df) > 0:
        # Filtrer les échantillons où DATABASE a fait une prédiction
        db_predictions = successful_df[
            (successful_df['database_prediction'].notna()) & 
            (successful_df['database_prediction'].str.strip() != '') &
            (successful_df['database_prediction'] != 'ERROR')
        ].copy()
        
        if len(db_predictions) > 0:
            # Calculer Exact Match pour DATABASE
            db_predictions['db_is_exact_match'] = (
                db_predictions['database_prediction'].str.lower().str.strip() == 
                db_predictions['expected_nature_product'].str.lower().str.strip()
            )
            
            # Calculer ROUGE-L pour DATABASE
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
            db_rouge_l_scores = []
            
            for _, row in db_predictions.iterrows():
                predicted = str(row['database_prediction']).lower().strip()
                expected = str(row['expected_nature_product']).lower().strip()
                
                scores = scorer.score(expected, predicted)
                rouge_l_f1 = scores['rougeL'].fmeasure
                db_rouge_l_scores.append(rouge_l_f1)
            
            db_predictions['db_rouge_l_score'] = db_rouge_l_scores
            
            # Métriques globales DATABASE
            db_exact_acc = (db_predictions['db_is_exact_match'].sum() / len(db_predictions)) * 100
            db_rouge_l_mean = sum(db_rouge_l_scores) / len(db_rouge_l_scores)
            
            print(f"🔍 DATABASE (toutes prédictions): {len(db_predictions):>4} échantillons | Exact: {db_exact_acc:>5.1f}% | ROUGE-L: {db_rouge_l_mean:>5.3f}")
            
            # Répartition par confiance DATABASE
            high_conf_db = db_predictions[db_predictions['database_confidence'] >= 0.8]
            med_conf_db = db_predictions[(db_predictions['database_confidence'] >= 0.5) & (db_predictions['database_confidence'] < 0.8)]
            low_conf_db = db_predictions[db_predictions['database_confidence'] < 0.5]
            
            if len(high_conf_db) > 0:
                high_exact = (high_conf_db['db_is_exact_match'].sum() / len(high_conf_db)) * 100
                high_rouge = high_conf_db['db_rouge_l_score'].mean()
                print(f"  📈 DATABASE (conf ≥ 0.8):       {len(high_conf_db):>4} échantillons | Exact: {high_exact:>5.1f}% | ROUGE-L: {high_rouge:>5.3f}")
            
            if len(med_conf_db) > 0:
                med_exact = (med_conf_db['db_is_exact_match'].sum() / len(med_conf_db)) * 100
                med_rouge = med_conf_db['db_rouge_l_score'].mean()
                print(f"  📊 DATABASE (0.5 ≤ conf < 0.8):  {len(med_conf_db):>4} échantillons | Exact: {med_exact:>5.1f}% | ROUGE-L: {med_rouge:>5.3f}")
            
            if len(low_conf_db) > 0:
                low_exact = (low_conf_db['db_is_exact_match'].sum() / len(low_conf_db)) * 100
                low_rouge = low_conf_db['db_rouge_l_score'].mean()
                print(f"  📉 DATABASE (conf < 0.5):        {len(low_conf_db):>4} échantillons | Exact: {low_exact:>5.1f}% | ROUGE-L: {low_rouge:>5.3f}")
        else:
            print("❌ Aucune prédiction DATABASE valide trouvée")
    
    # Analyse similaire pour T5 si disponible
    if len(successful_df) > 0:
        t5_predictions = successful_df[
            (successful_df['t5_prediction'].notna()) & 
            (successful_df['t5_prediction'].str.strip() != '') &
            (successful_df['t5_prediction'] != 'ERROR')
        ].copy()
        
        if len(t5_predictions) > 0:
            print(f"\n📊 ANALYSE PERFORMANCES BRUTES DU NOEUD T5")
            print("=" * 60)
            
            # Calculer Exact Match pour T5
            t5_predictions['t5_is_exact_match'] = (
                t5_predictions['t5_prediction'].str.lower().str.strip() == 
                t5_predictions['expected_nature_product'].str.lower().str.strip()
            )
            
            # Calculer ROUGE-L pour T5
            t5_rouge_l_scores = []
            
            for _, row in t5_predictions.iterrows():
                predicted = str(row['t5_prediction']).lower().strip()
                expected = str(row['expected_nature_product']).lower().strip()
                
                scores = scorer.score(expected, predicted)
                rouge_l_f1 = scores['rougeL'].fmeasure
                t5_rouge_l_scores.append(rouge_l_f1)
            
            t5_predictions['t5_rouge_l_score'] = t5_rouge_l_scores
            
            # Métriques globales T5
            t5_exact_acc = (t5_predictions['t5_is_exact_match'].sum() / len(t5_predictions)) * 100
            t5_rouge_l_mean = sum(t5_rouge_l_scores) / len(t5_rouge_l_scores)
            
            print(f"🔍 T5 (toutes prédictions):     {len(t5_predictions):>4} échantillons | Exact: {t5_exact_acc:>5.1f}% | ROUGE-L: {t5_rouge_l_mean:>5.3f}")
    
    # Exemples de prédictions incorrectes pour analyse
    if len(successful_df) > 0:
        incorrect_exact = successful_df[successful_df['is_exact_match'] == False]
        if len(incorrect_exact) > 0:
            print(f"\n🔍 EXEMPLES DE PRÉDICTIONS NON-EXACTES (sur {len(incorrect_exact)} total)")
            print("=" * 80)
            
            # Exemples d'identifications incorrectes
            incorrect_identifications = incorrect_exact[incorrect_exact['prediction_type'] == 'identification']
            if len(incorrect_identifications) > 0:
                print(f"\n❌ IDENTIFICATIONS INCORRECTES ({len(incorrect_identifications)} cas):")
                for i, row in incorrect_identifications.head(5).iterrows():
                    print(f"📝 Description: {row['description_cleaned'][:60]}...")
                    print(f"✅ Attendu: {row['expected_nature_product']}")
                    print(f"🔍 Identifié comme: {row['predicted_nature_product']} (conf: {row['confidence']:.3f})")
                    print(f"📏 ROUGE-L: {row['rouge_l_score']:.3f} | Nœud: {row['decision_node']}")
                    print("-" * 60)
            
            # Exemples de créations incorrectes  
            incorrect_creations = incorrect_exact[incorrect_exact['prediction_type'] == 'creation']
            if len(incorrect_creations) > 0:
                print(f"\n🆕 CRÉATIONS INCORRECTES ({len(incorrect_creations)} cas):")
                for i, row in incorrect_creations.head(5).iterrows():
                    print(f"📝 Description: {row['description_cleaned'][:60]}...")
                    print(f"✅ Attendu: {row['expected_nature_product']}")
                    print(f"🆕 Créé comme: {row['predicted_nature_product']} (conf: {row['confidence']:.3f})")
                    print(f"📏 ROUGE-L: {row['rouge_l_score']:.3f} | Nœud: {row['decision_node']}")
                    print("-" * 60)
    
    # Analyse des seuils optimaux DATABASE
    if len(successful_df) > 0:
        analyze_database_thresholds(successful_df, timestamp)
        analyze_t5_thresholds(successful_df, timestamp)
    
    return results_df, metrics

def analyze_database_thresholds(successful_df, timestamp):
    """Analyze DATABASE threshold optimization through ROUGE-L distribution"""
    print("\n📊 ANALYSE DES SEUILS OPTIMAUX DATABASE")
    print("=" * 60)
    
    # Filtrer les échantillons où DATABASE a fait une prédiction
    db_predictions = successful_df[
        (successful_df['database_prediction'].notna()) & 
        (successful_df['database_prediction'].str.strip() != '') &
        (successful_df['database_prediction'] != 'ERROR')
    ].copy()
    
    if len(db_predictions) == 0:
        print("❌ Aucune prédiction DATABASE valide pour l'analyse")
        return
    
    # Calculer ROUGE-L pour toutes les prédictions DATABASE
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    db_rouge_scores = []
    db_exact_matches = []
    
    for _, row in db_predictions.iterrows():
        predicted = str(row['database_prediction']).lower().strip()
        expected = str(row['expected_nature_product']).lower().strip()
        
        # ROUGE-L score
        scores = scorer.score(expected, predicted)
        rouge_score = scores['rougeL'].fmeasure
        db_rouge_scores.append(rouge_score)
        
        # Exact match
        is_exact = (predicted == expected)
        db_exact_matches.append(is_exact)
    
    db_predictions['db_rouge_score'] = db_rouge_scores
    db_predictions['db_exact_match'] = db_exact_matches
    
    # Séparer successful vs non-successful
    successful_db = db_predictions[db_predictions['db_exact_match'] == True]
    unsuccessful_db = db_predictions[db_predictions['db_exact_match'] == False]
    
    print(f"📈 Prédictions DATABASE exactes: {len(successful_db)} ({len(successful_db)/len(db_predictions)*100:.1f}%)")
    print(f"📉 Prédictions DATABASE inexactes: {len(unsuccessful_db)} ({len(unsuccessful_db)/len(db_predictions)*100:.1f}%)")
    
    # Créer la visualisation
    plt.figure(figsize=(12, 8))
    
    # Distribution des scores ROUGE-L
    plt.subplot(2, 2, 1)
    if len(successful_db) > 0:
        plt.hist(successful_db['db_rouge_score'], bins=20, alpha=0.7, label=f'Exact Match (n={len(successful_db)})', color='green')
    if len(unsuccessful_db) > 0:
        plt.hist(unsuccessful_db['db_rouge_score'], bins=20, alpha=0.7, label=f'Non-Exact (n={len(unsuccessful_db)})', color='red')
    plt.xlabel('ROUGE-L Score DATABASE')
    plt.ylabel('Fréquence')
    plt.title('Distribution ROUGE-L: Exact vs Non-Exact')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Distribution des confidences DATABASE
    plt.subplot(2, 2, 2)
    if len(successful_db) > 0:
        plt.hist(successful_db['database_confidence'], bins=20, alpha=0.7, label=f'Exact Match (n={len(successful_db)})', color='green')
    if len(unsuccessful_db) > 0:
        plt.hist(unsuccessful_db['database_confidence'], bins=20, alpha=0.7, label=f'Non-Exact (n={len(unsuccessful_db)})', color='red')
    plt.xlabel('Confiance DATABASE')
    plt.ylabel('Fréquence')
    plt.title('Distribution Confiance: Exact vs Non-Exact')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Scatter plot: Confiance vs ROUGE-L
    plt.subplot(2, 2, 3)
    if len(successful_db) > 0:
        plt.scatter(successful_db['database_confidence'], successful_db['db_rouge_score'], 
                   alpha=0.6, label=f'Exact Match (n={len(successful_db)})', color='green', s=20)
    if len(unsuccessful_db) > 0:
        plt.scatter(unsuccessful_db['database_confidence'], unsuccessful_db['db_rouge_score'], 
                   alpha=0.6, label=f'Non-Exact (n={len(unsuccessful_db)})', color='red', s=20)
    plt.xlabel('Confiance DATABASE')
    plt.ylabel('ROUGE-L Score DATABASE')
    plt.title('Confiance vs ROUGE-L')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Analyse des seuils optimaux
    plt.subplot(2, 2, 4)
    thresholds = np.arange(0.1, 1.0, 0.05)
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for threshold in thresholds:
        # Prédictions considérées comme "acceptées" par DATABASE avec ce seuil
        accepted = db_predictions[db_predictions['database_confidence'] >= threshold]
        
        if len(accepted) > 0:
            # Précision: parmi les acceptées, combien sont exactes
            precision = len(accepted[accepted['db_exact_match'] == True]) / len(accepted)
            # Rappel: parmi les exactes, combien sont acceptées
            recall = len(accepted[accepted['db_exact_match'] == True]) / len(successful_db) if len(successful_db) > 0 else 0
            # F1 score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = 0
            recall = 0
            f1 = 0
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    
    plt.plot(thresholds, precision_scores, label='Précision', marker='o', markersize=3)
    plt.plot(thresholds, recall_scores, label='Rappel', marker='s', markersize=3)
    plt.plot(thresholds, f1_scores, label='F1-Score', marker='^', markersize=3)
    plt.axvline(x=0.8, color='black', linestyle='--', alpha=0.5, label='Seuil actuel (0.8)')
    plt.xlabel('Seuil de Confiance DATABASE')
    plt.ylabel('Score')
    plt.title('Optimisation du Seuil DATABASE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Trouver le seuil optimal (meilleur F1)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    plt.tight_layout()
    
    # Sauvegarder la figure
    fig_filename = f"database_threshold_analysis_{timestamp}.png"
    plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 Figure sauvegardée: {fig_filename}")
    
    # Afficher les recommandations
    print(f"\n🎯 RECOMMANDATIONS SEUILS DATABASE:")
    print(f"📈 Seuil optimal (F1): {optimal_threshold:.2f} (F1={optimal_f1:.3f})")
    print(f"📊 Seuil actuel: 0.80")
    
    # Statistiques à différents seuils
    current_threshold_stats = db_predictions[db_predictions['database_confidence'] >= 0.8]
    optimal_threshold_stats = db_predictions[db_predictions['database_confidence'] >= optimal_threshold]
    
    if len(current_threshold_stats) > 0:
        current_precision = len(current_threshold_stats[current_threshold_stats['db_exact_match'] == True]) / len(current_threshold_stats)
        print(f"⚡ Seuil actuel (0.8): {len(current_threshold_stats)} échantillons, précision: {current_precision:.3f}")
    
    if len(optimal_threshold_stats) > 0:
        optimal_precision = len(optimal_threshold_stats[optimal_threshold_stats['db_exact_match'] == True]) / len(optimal_threshold_stats)
        print(f"🔥 Seuil optimal ({optimal_threshold:.2f}): {len(optimal_threshold_stats)} échantillons, précision: {optimal_precision:.3f}")
    
    # Sauvegarder les données d'analyse
    analysis_data = pd.DataFrame({
        'threshold': thresholds,
        'precision': precision_scores,
        'recall': recall_scores,
        'f1_score': f1_scores
    })
    analysis_filename = f"database_threshold_optimization_{timestamp}.csv"
    analysis_data.to_csv(analysis_filename, index=False)
    print(f"💾 Données d'optimisation sauvegardées: {analysis_filename}")
    
    # Ne pas afficher la figure dans le conteneur (pas d'interface graphique)
    print(f"📈 Graphiques générés et sauvegardés dans {fig_filename}")

def analyze_t5_thresholds(successful_df, timestamp):
    """Analyze T5 threshold optimization through ROUGE-L distribution"""
    print("\n📊 ANALYSE DES SEUILS OPTIMAUX T5")
    print("=" * 60)
    
    # Filtrer les échantillons où T5 a fait une prédiction
    t5_predictions = successful_df[
        (successful_df['t5_prediction'].notna()) & 
        (successful_df['t5_prediction'].str.strip() != '') &
        (successful_df['t5_prediction'] != 'ERROR')
    ].copy()
    
    if len(t5_predictions) == 0:
        print("❌ Aucune prédiction T5 valide pour l'analyse")
        return
    
    # Calculer ROUGE-L pour toutes les prédictions T5
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    t5_rouge_scores = []
    t5_exact_matches = []
    
    for _, row in t5_predictions.iterrows():
        predicted = str(row['t5_prediction']).lower().strip()
        expected = str(row['expected_nature_product']).lower().strip()
        
        # ROUGE-L score
        scores = scorer.score(expected, predicted)
        rouge_score = scores['rougeL'].fmeasure
        t5_rouge_scores.append(rouge_score)

        
        # Exact match
        is_exact = (predicted == expected)
        t5_exact_matches.append(is_exact)
    
    t5_predictions['t5_rouge_score'] = t5_rouge_scores
    t5_predictions['t5_exact_match'] = t5_exact_matches
    
    # Séparer successful vs non-successful
    successful_t5 = t5_predictions[t5_predictions['t5_exact_match'] == True]
    unsuccessful_t5 = t5_predictions[t5_predictions['t5_exact_match'] == False]
    
    print(f"📈 Prédictions T5 exactes: {len(successful_t5)} ({len(successful_t5)/len(t5_predictions)*100:.1f}%)")
    print(f"📉 Prédictions T5 inexactes: {len(unsuccessful_t5)} ({len(unsuccessful_t5)/len(t5_predictions)*100:.1f}%)")
    
    # Créer la visualisation
    plt.figure(figsize=(12, 8))
    
    # Distribution des scores ROUGE-L
    plt.subplot(2, 2, 1)
    if len(successful_t5) > 0:
        plt.hist(successful_t5['t5_rouge_score'], bins=15, alpha=0.7, label=f'Exact Match (n={len(successful_t5)})', color='blue')
    if len(unsuccessful_t5) > 0:
        plt.hist(unsuccessful_t5['t5_rouge_score'], bins=15, alpha=0.7, label=f'Non-Exact (n={len(unsuccessful_t5)})', color='orange')
    plt.xlabel('ROUGE-L Score T5')
    plt.ylabel('Fréquence')
    plt.title('Distribution ROUGE-L T5: Exact vs Non-Exact')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Distribution des confidences T5
    plt.subplot(2, 2, 2)
    if len(successful_t5) > 0:
        plt.hist(successful_t5['t5_confidence'], bins=15, alpha=0.7, label=f'Exact Match (n={len(successful_t5)})', color='blue')
    if len(unsuccessful_t5) > 0:
        plt.hist(unsuccessful_t5['t5_confidence'], bins=15, alpha=0.7, label=f'Non-Exact (n={len(unsuccessful_t5)})', color='orange')
    plt.xlabel('Confiance T5')
    plt.ylabel('Fréquence')
    plt.title('Distribution Confiance T5: Exact vs Non-Exact')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Scatter plot: Confiance vs ROUGE-L
    plt.subplot(2, 2, 3)
    if len(successful_t5) > 0:
        plt.scatter(successful_t5['t5_confidence'], successful_t5['t5_rouge_score'], 
                   alpha=0.6, label=f'Exact Match (n={len(successful_t5)})', color='blue', s=20)
    if len(unsuccessful_t5) > 0:
        plt.scatter(unsuccessful_t5['t5_confidence'], unsuccessful_t5['t5_rouge_score'], 
                   alpha=0.6, label=f'Non-Exact (n={len(unsuccessful_t5)})', color='orange', s=20)
    plt.xlabel('Confiance T5')
    plt.ylabel('ROUGE-L Score T5')
    plt.title('Confiance T5 vs ROUGE-L')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Analyse des seuils optimaux pour T5 → LLM
    plt.subplot(2, 2, 4)
    thresholds = np.arange(0.1, 1.0, 0.05)
    precision_scores = []
    recall_scores = []
    f1_scores = []
    should_pass_to_llm_scores = []  # % qui devrait passer à LLM
    
    for threshold in thresholds:
        # Prédictions T5 "acceptées" avec ce seuil (ne passent PAS à LLM)
        t5_accepted = t5_predictions[t5_predictions['t5_confidence'] >= threshold]
        # Prédictions T5 "rejetées" avec ce seuil (passent à LLM)
        t5_rejected = t5_predictions[t5_predictions['t5_confidence'] < threshold]
        
        if len(t5_accepted) > 0:
            # Précision: parmi les acceptées par T5, combien sont exactes
            precision = len(t5_accepted[t5_accepted['t5_exact_match'] == True]) / len(t5_accepted)
            # Rappel: parmi les exactes, combien sont acceptées par T5
            recall = len(t5_accepted[t5_accepted['t5_exact_match'] == True]) / len(successful_t5) if len(successful_t5) > 0 else 0
            # F1 score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = 0
            recall = 0
            f1 = 0
        
        # Pourcentage qui devrait passer à LLM (rejets)
        should_pass_pct = len(t5_rejected) / len(t5_predictions) * 100 if len(t5_predictions) > 0 else 0
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        should_pass_to_llm_scores.append(should_pass_pct)
    
    plt.plot(thresholds, precision_scores, label='Précision T5', marker='o', markersize=3)
    plt.plot(thresholds, recall_scores, label='Rappel T5', marker='s', markersize=3)
    plt.plot(thresholds, f1_scores, label='F1-Score T5', marker='^', markersize=3)
    plt.axvline(x=0.7, color='black', linestyle='--', alpha=0.5, label='Seuil actuel T5 (0.7)')
    plt.xlabel('Seuil de Confiance T5')
    plt.ylabel('Score')
    plt.title('Optimisation du Seuil T5 → LLM')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Trouver le seuil optimal (meilleur F1)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    plt.tight_layout()
    
    # Sauvegarder la figure
    fig_filename = f"t5_threshold_analysis_{timestamp}.png"
    plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 Figure T5 sauvegardée: {fig_filename}")
    
    # Afficher les recommandations
    print(f"\n🎯 RECOMMANDATIONS SEUILS T5:")
    print(f"📈 Seuil optimal (F1): {optimal_threshold:.2f} (F1={optimal_f1:.3f})")
    print(f"📊 Seuil actuel: 0.70")
    
    # Statistiques à différents seuils
    current_threshold_stats = t5_predictions[t5_predictions['t5_confidence'] >= 0.7]
    optimal_threshold_stats = t5_predictions[t5_predictions['t5_confidence'] >= optimal_threshold]
    
    if len(current_threshold_stats) > 0:
        current_precision = len(current_threshold_stats[current_threshold_stats['t5_exact_match'] == True]) / len(current_threshold_stats)
        current_pass_to_llm = len(t5_predictions[t5_predictions['t5_confidence'] < 0.7]) / len(t5_predictions) * 100
        print(f"⚡ Seuil actuel (0.7): {len(current_threshold_stats)} acceptés par T5, précision: {current_precision:.3f}")
        print(f"   → {current_pass_to_llm:.1f}% des cas passent à LLM")
    
    if len(optimal_threshold_stats) > 0:
        optimal_precision = len(optimal_threshold_stats[optimal_threshold_stats['t5_exact_match'] == True]) / len(optimal_threshold_stats)
        optimal_pass_to_llm = len(t5_predictions[t5_predictions['t5_confidence'] < optimal_threshold]) / len(t5_predictions) * 100
        print(f"🔥 Seuil optimal ({optimal_threshold:.2f}): {len(optimal_threshold_stats)} acceptés par T5, précision: {optimal_precision:.3f}")
        print(f"   → {optimal_pass_to_llm:.1f}% des cas passeraient à LLM")
    
    # Impact sur les coûts (estimation)
    current_llm_cases = len(t5_predictions[t5_predictions['t5_confidence'] < 0.7])
    optimal_llm_cases = len(t5_predictions[t5_predictions['t5_confidence'] < optimal_threshold])
    cost_impact = ((optimal_llm_cases - current_llm_cases) / current_llm_cases * 100) if current_llm_cases > 0 else 0
    
    print(f"💰 Impact coût estimé: {cost_impact:+.1f}% d'appels LLM")
    
    # Sauvegarder les données d'analyse T5
    analysis_data = pd.DataFrame({
        'threshold': thresholds,
        'precision': precision_scores,
        'recall': recall_scores,
        'f1_score': f1_scores,
        'pass_to_llm_pct': should_pass_to_llm_scores
    })
    analysis_filename = f"t5_threshold_optimization_{timestamp}.csv"
    analysis_data.to_csv(analysis_filename, index=False)
    print(f"💾 Données d'optimisation T5 sauvegardées: {analysis_filename}")
    
    print(f"📈 Graphiques T5 générés et sauvegardés dans {fig_filename}")

if __name__ == "__main__":
    # Configuration par défaut
    VALIDATION_FILE = "./data/validation_set.csv"
    NATURE_PRODUCT_FILE = "./data/nature_product.csv" 
    SAMPLE_SIZE = 20  # Commencer par un échantillon pour tester
    MAX_CONCURRENT = 5  # Éviter de surcharger l'API
    
    print("🚀 Démarrage de l'évaluation sur données réelles...")
    print(f"📁 Fichier validation: {VALIDATION_FILE}")
    print(f"📁 Fichier nature_product: {NATURE_PRODUCT_FILE}")
    print(f"📊 Taille échantillon: {SAMPLE_SIZE if SAMPLE_SIZE else 'Toutes les données'}")
    
    asyncio.run(run_real_data_evaluation(
        validation_file=VALIDATION_FILE,
        nature_product_file=NATURE_PRODUCT_FILE, 
        sample_size=SAMPLE_SIZE,
        max_concurrent=MAX_CONCURRENT
    ))