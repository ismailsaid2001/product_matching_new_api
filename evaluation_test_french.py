import asyncio
import aiohttp
import pandas as pd
import time
from datetime import datetime

# Test products - 150 produits entièrement en français pour restaurants
TEST_PRODUCTS = [
    # === PRODUITS ALIMENTAIRES (75 items) ===
    
    # Fromages (15 produits)
    {"product_id": "FR001", "description": "Emmental râpé Président 200g sachet refermable", "expected_label": "emmental rape"},
    {"product_id": "FR002", "description": "Camembert de Normandie AOP Président 250g", "expected_label": "camembert"},
    {"product_id": "FR003", "description": "Roquefort AOP Société 100g portion", "expected_label": "roquefort"},
    {"product_id": "FR004", "description": "Chèvre frais ail et fines herbes Chavroux 150g", "expected_label": "fromage chevre"},
    {"product_id": "FR005", "description": "Gruyère France râpé 200g sachet", "expected_label": "gruyere rape"},
    {"product_id": "FR006", "description": "Brie de Meaux AOP 200g portion", "expected_label": "brie"},
    {"product_id": "FR007", "description": "Comté AOP 12 mois affinage 200g", "expected_label": "comte"},
    {"product_id": "FR008", "description": "Mozzarella fraîche di bufala 125g", "expected_label": "mozzarella"},
    {"product_id": "FR009", "description": "Cantal entre-deux AOP 200g portion", "expected_label": "cantal"},
    {"product_id": "FR010", "description": "Saint-Nectaire AOP fermier 200g", "expected_label": "saint nectaire"},
    {"product_id": "FR011", "description": "Bleu d'Auvergne AOP 150g portion", "expected_label": "bleu auvergne"},
    {"product_id": "FR012", "description": "Chaource AOP coeur de Champagne 250g", "expected_label": "chaource"},
    {"product_id": "FR013", "description": "Maroilles AOP quart 200g", "expected_label": "maroilles"},
    {"product_id": "FR014", "description": "Reblochon de Savoie AOP 450g", "expected_label": "reblochon"},
    {"product_id": "FR015", "description": "Munster AOP fermier Alsace 200g", "expected_label": "munster"},

    # Charcuterie (15 produits)
    {"product_id": "CH001", "description": "Jambon cuit supérieur Fleury Michon 4 tranches", "expected_label": "jambon cuit"},
    {"product_id": "CH002", "description": "Jambon de Bayonne AOP 18 mois 100g tranches", "expected_label": "jambon cru tranche"},
    {"product_id": "CH003", "description": "Saucisson sec pur porc traditionnel 200g", "expected_label": "saucisson sec"},
    {"product_id": "CH004", "description": "Rillettes du Mans pur porc 220g terrine", "expected_label": "rillettes"},
    {"product_id": "CH005", "description": "Pâté de campagne pur porc 200g", "expected_label": "pate campagne"},
    {"product_id": "CH006", "description": "Chorizo fort Espagne tranches 100g", "expected_label": "chorizo"},
    {"product_id": "CH007", "description": "Coppa di Parma tranches 80g", "expected_label": "coppa"},
    {"product_id": "CH008", "description": "Boudin noir aux pommes 2 pièces 200g", "expected_label": "boudin noir"},
    {"product_id": "CH009", "description": "Andouille de Vire véritable 200g", "expected_label": "andouille"},
    {"product_id": "CH010", "description": "Merguez pur boeuf x6 300g", "expected_label": "merguez"},
    {"product_id": "CH011", "description": "Saucisse de Toulouse fraîche x4 300g", "expected_label": "saucisse toulouse"},
    {"product_id": "CH012", "description": "Lardons fumés nature 200g barquette", "expected_label": "lardon fume"},
    {"product_id": "CH013", "description": "Pancetta italienne tranches 100g", "expected_label": "pancetta"},
    {"product_id": "CH014", "description": "Mortadelle di Bologna IGP 100g", "expected_label": "mortadelle"},
    {"product_id": "CH015", "description": "Bresaola della Valtellina IGP 80g", "expected_label": "bresaola"},

    # Poissons et Fruits de mer (15 produits)
    {"product_id": "PS001", "description": "Filet de saumon frais Ecosse portion 150g", "expected_label": "filet saumon"},
    {"product_id": "PS002", "description": "Saumon fumé Ecosse tranches 100g", "expected_label": "saumon fume tranche"},
    {"product_id": "PS003", "description": "Crevettes cuites décortiquées 300g", "expected_label": "crevette"},
    {"product_id": "PS004", "description": "Filet de bar ligne portion 200g", "expected_label": "filet bar"},
    {"product_id": "PS005", "description": "Pavé de cabillaud frais 150g", "expected_label": "pave cabillaud"},
    {"product_id": "PS006", "description": "Truite fumée France entière 200g", "expected_label": "truite fumee"},
    {"product_id": "PS007", "description": "Noix de Saint-Jacques fraîches x8", "expected_label": "saint jacques"},
    {"product_id": "PS008", "description": "Moules de Bouchot France 1kg", "expected_label": "moule"},
    {"product_id": "PS009", "description": "Huîtres fines de claires n°3 bourriche 12", "expected_label": "huitre"},
    {"product_id": "PS010", "description": "Thon rouge frais portion 150g", "expected_label": "thon rouge"},
    {"product_id": "PS011", "description": "Sole meunière portion 200g", "expected_label": "sole"},
    {"product_id": "PS012", "description": "Sardines fraîches 500g", "expected_label": "sardine"},
    {"product_id": "PS013", "description": "Maquereau fumé filets 150g", "expected_label": "maquereau fume"},
    {"product_id": "PS014", "description": "Langoustines vivantes 500g", "expected_label": "langoustine"},
    {"product_id": "PS015", "description": "Bulots cuits France 300g", "expected_label": "bulot"},

    # Viandes (15 produits)
    {"product_id": "VI001", "description": "Escalope de dinde fermière blanc 4 pièces", "expected_label": "escalope dinde"},
    {"product_id": "VI002", "description": "Filet de boeuf Angus portion 200g", "expected_label": "filet boeuf"},
    {"product_id": "VI003", "description": "Côte de porc échine épaisse 2cm", "expected_label": "cote porc"},
    {"product_id": "VI004", "description": "Gigot d'agneau désossé 1kg", "expected_label": "gigot agneau"},
    {"product_id": "VI005", "description": "Magret de canard fermier 300g", "expected_label": "magret canard"},
    {"product_id": "VI006", "description": "Entrecôte de boeuf maturée 21 jours 250g", "expected_label": "entrecote"},
    {"product_id": "VI007", "description": "Filet de porc mignon entier 400g", "expected_label": "filet porc"},
    {"product_id": "VI008", "description": "Cuisse de poulet fermier Label Rouge x2", "expected_label": "cuisse poulet"},
    {"product_id": "VI009", "description": "Pavé de rumsteck boeuf 180g", "expected_label": "rumsteck"},
    {"product_id": "VI010", "description": "Carré d'agneau 8 côtes 600g", "expected_label": "carre agneau"},
    {"product_id": "VI011", "description": "Bavette à l'échalote portion 180g", "expected_label": "bavette"},
    {"product_id": "VI012", "description": "Blanquette de veau 500g", "expected_label": "blanquette veau"},
    {"product_id": "VI013", "description": "Confit de canard 4 cuisses 800g", "expected_label": "confit canard"},
    {"product_id": "VI014", "description": "Rôti de porc aux herbes 600g", "expected_label": "roti porc"},
    {"product_id": "VI015", "description": "Lapin fermier entier découpé 1.2kg", "expected_label": "lapin"},

    # Épicerie salée (15 produits)
    {"product_id": "EP001", "description": "Huile d'olive extra vierge Puget 500ml", "expected_label": "huile olive extra"},
    {"product_id": "EP002", "description": "Vinaigre balsamique Modène IGP 250ml", "expected_label": "vinaigre balsamique"},
    {"product_id": "EP003", "description": "Moutarde de Dijon forte Fallot 210g", "expected_label": "moutarde dijon"},
    {"product_id": "EP004", "description": "Cornichons extra fins français 220g", "expected_label": "cornichon"},
    {"product_id": "EP005", "description": "Câpres au vinaigre Pantelleria 100g", "expected_label": "capre"},
    {"product_id": "EP006", "description": "Olives noires Kalamata 200g", "expected_label": "olive noire"},
    {"product_id": "EP007", "description": "Tapenade d'olives noires 100g", "expected_label": "tapenade"},
    {"product_id": "EP008", "description": "Pesto au basilic Barilla 190g", "expected_label": "pesto"},
    {"product_id": "EP009", "description": "Tomates séchées à l'huile 200g", "expected_label": "tomate sechee"},
    {"product_id": "EP010", "description": "Artichauts à l'huile 280g", "expected_label": "artichaut huile"},
    {"product_id": "EP011", "description": "Anchois salés boîte 50g", "expected_label": "anchois"},
    {"product_id": "EP012", "description": "Sauce soja sucrée 250ml", "expected_label": "sauce soja"},
    {"product_id": "EP013", "description": "Nuoc-mâm Vietnam 200ml", "expected_label": "nuoc mam"},
    {"product_id": "EP014", "description": "Harissa forte tube 70g", "expected_label": "harissa"},
    {"product_id": "EP015", "description": "Wasabi en poudre 25g", "expected_label": "wasabi"},

    # === BOISSONS (45 items) ===

    # Vins (20 produits)
    {"product_id": "VN001", "description": "Bordeaux Saint-Émilion Grand Cru 2020 75cl", "expected_label": "vin rouge"},
    {"product_id": "VN002", "description": "Châteauneuf-du-Pape rouge 2019 75cl", "expected_label": "vin rouge"},
    {"product_id": "VN003", "description": "Sancerre blanc Loire Valley 2022 75cl", "expected_label": "vin blanc"},
    {"product_id": "VN004", "description": "Chablis Premier Cru 2021 75cl", "expected_label": "vin blanc"},
    {"product_id": "VN005", "description": "Côtes du Rhône Villages rouge 2020 75cl", "expected_label": "vin rouge"},
    {"product_id": "VN006", "description": "Muscadet Sèvre-et-Maine sur lie 75cl", "expected_label": "vin blanc"},
    {"product_id": "VN007", "description": "Beaujolais Villages Primeur 2023 75cl", "expected_label": "vin rouge"},
    {"product_id": "VN008", "description": "Pouilly-Fumé Loire 2022 75cl", "expected_label": "vin blanc"},
    {"product_id": "VN009", "description": "Côte de Beaune blanc 2021 75cl", "expected_label": "vin blanc"},
    {"product_id": "VN010", "description": "Bandol rouge Provence 2019 75cl", "expected_label": "vin rouge"},
    {"product_id": "VN011", "description": "Alsace Riesling 2022 75cl", "expected_label": "vin blanc"},
    {"product_id": "VN012", "description": "Cahors Malbec 2020 75cl", "expected_label": "vin rouge"},
    {"product_id": "VN013", "description": "Jurançon moelleux 2021 50cl", "expected_label": "vin blanc"},
    {"product_id": "VN014", "description": "Côtes de Provence rosé 2023 75cl", "expected_label": "vin rose"},
    {"product_id": "VN015", "description": "Chinon rouge Loire 2021 75cl", "expected_label": "vin rouge"},
    {"product_id": "VN016", "description": "Gewurztraminer Alsace 2022 75cl", "expected_label": "vin blanc"},
    {"product_id": "VN017", "description": "Médoc rouge Bordeaux 2020 75cl", "expected_label": "vin rouge"},
    {"product_id": "VN018", "description": "Entre-Deux-Mers blanc 2023 75cl", "expected_label": "vin blanc"},
    {"product_id": "VN019", "description": "Morgon Beaujolais 2022 75cl", "expected_label": "vin rouge"},
    {"product_id": "VN020", "description": "Saumur-Champigny rouge 2021 75cl", "expected_label": "vin rouge"},

    # Champagnes et Crémants (10 produits)
    {"product_id": "CP001", "description": "Champagne Veuve Clicquot Brut 75cl", "expected_label": "champagne"},
    {"product_id": "CP002", "description": "Champagne Moët & Chandon Impérial 75cl", "expected_label": "champagne"},
    {"product_id": "CP003", "description": "Crémant de Loire Bouvet Ladubay 75cl", "expected_label": "cremant"},
    {"product_id": "CP004", "description": "Champagne Dom Pérignon Vintage 2015 75cl", "expected_label": "champagne"},
    {"product_id": "CP005", "description": "Crémant d'Alsace blanc de blancs 75cl", "expected_label": "cremant"},
    {"product_id": "CP006", "description": "Champagne Laurent-Perrier Brut 75cl", "expected_label": "champagne"},
    {"product_id": "CP007", "description": "Crémant de Bourgogne rosé 75cl", "expected_label": "cremant"},
    {"product_id": "CP008", "description": "Champagne Pol Roger Réserve 75cl", "expected_label": "champagne"},
    {"product_id": "CP009", "description": "Blanquette de Limoux Sieur d'Arques 75cl", "expected_label": "blanquette"},
    {"product_id": "CP010", "description": "Champagne Billecart-Salmon Brut 75cl", "expected_label": "champagne"},

    # Bières (10 produits)
    {"product_id": "BR001", "description": "Kronenbourg 1664 blonde 6x25cl", "expected_label": "biere 1664"},
    {"product_id": "BR002", "description": "Leffe blonde abbaye 6x33cl bouteilles", "expected_label": "biere"},
    {"product_id": "BR003", "description": "Grimbergen blonde 6x25cl", "expected_label": "biere"},
    {"product_id": "BR004", "description": "Stella Artois premium 4x50cl", "expected_label": "biere"},
    {"product_id": "BR005", "description": "Heineken blonde 12x33cl canettes", "expected_label": "biere"},
    {"product_id": "BR006", "description": "Chimay rouge trappiste 33cl", "expected_label": "biere"},
    {"product_id": "BR007", "description": "Hoegaarden blanche 6x25cl", "expected_label": "biere"},
    {"product_id": "BR008", "description": "Desperados tequila 3x33cl", "expected_label": "biere"},
    {"product_id": "BR009", "description": "Guinness stout 4x44cl canettes", "expected_label": "biere"},
    {"product_id": "BR010", "description": "Carlsberg pilsner 8x50cl", "expected_label": "biere"},

    # Spiritueux (5 produits)
    {"product_id": "SP001", "description": "Cognac Hennessy VS 70cl", "expected_label": "cognac"},
    {"product_id": "SP002", "description": "Armagnac Bas-Armagnac VSOP 70cl", "expected_label": "armagnac"},
    {"product_id": "SP003", "description": "Calvados du Pays d'Auge 70cl", "expected_label": "calvados"},
    {"product_id": "SP004", "description": "Pastis de Marseille Ricard 100cl", "expected_label": "pastis"},
    {"product_id": "SP005", "description": "Whisky français Rozelieures 70cl", "expected_label": "whisky"},

    # === PRODUITS D'ENTRETIEN (15 items) ===
    {"product_id": "CL001", "description": "Liquide vaisselle professionnel Ecover 5L", "expected_label": "liquide vaisselle"},
    {"product_id": "CL002", "description": "Dégraissant cuisine professionnel 750ml vaporisateur", "expected_label": "degraissant"},
    {"product_id": "CL003", "description": "Nettoyant désinfectant surfaces Sanytol 1L", "expected_label": "desinfectant"},
    {"product_id": "CL004", "description": "Tablettes lave-vaisselle Finish Quantum 50 unités", "expected_label": "tablette lave vaisselle"},
    {"product_id": "CL005", "description": "Lingettes désinfectantes Javel Spontex 100 unités", "expected_label": "lingette desinfectante"},
    {"product_id": "CL006", "description": "Détartrant machine à café professionnel 1L", "expected_label": "detartrant"},
    {"product_id": "CL007", "description": "Nettoyant vitres professionnel Mr Propre 5L", "expected_label": "nettoyant vitre"},
    {"product_id": "CL008", "description": "Savon liquide mains antibactérien 500ml distributeur", "expected_label": "savon main"},
    {"product_id": "CL009", "description": "Eau de Javel concentrée La Croix 12° 1L", "expected_label": "javel"},
    {"product_id": "CL010", "description": "Essuie-tout professionnel Lotus 6 rouleaux", "expected_label": "essuie tout"},
    {"product_id": "CL011", "description": "Papier toilette professionnel Tork 12 rouleaux", "expected_label": "papier toilette"},
    {"product_id": "CL012", "description": "Sacs poubelle 50L renforcés noir 20 unités", "expected_label": "sac poubelle"},
    {"product_id": "CL013", "description": "Désodorisant professionnel Febreze 300ml", "expected_label": "desodorisant"},
    {"product_id": "CL014", "description": "Nettoyant sol restaurant concentré 5L", "expected_label": "nettoyant sol"},
    {"product_id": "CL015", "description": "Film plastique alimentaire 300m x 30cm", "expected_label": "film plastique"},

    # === MATÉRIEL DE CUISINE (15 items) ===
    {"product_id": "EQ001", "description": "Planche à découper polyéthylène 40x30cm blanche", "expected_label": "planche decouper"},
    {"product_id": "EQ002", "description": "Couteau de chef professionnel 20cm lame acier", "expected_label": "couteau chef"},
    {"product_id": "EQ003", "description": "Bac gastronorme GN 1/1 inox hauteur 20mm", "expected_label": "bac gastronome"},
    {"product_id": "EQ004", "description": "Thermomètre sonde digital cuisine -50°C +200°C", "expected_label": "thermometre sonde"},
    {"product_id": "EQ005", "description": "Fouet professionnel inox 30cm manche plastique", "expected_label": "fouet"},
    {"product_id": "EQ006", "description": "Mandoline légumes réglable inox avec protection", "expected_label": "mandoline"},
    {"product_id": "EQ007", "description": "Balance de cuisine digitale 5kg précision 1g", "expected_label": "balance cuisine"},
    {"product_id": "EQ008", "description": "Mixeur plongeant professionnel 400W", "expected_label": "mixeur plongeant"},
    {"product_id": "EQ009", "description": "Casserole inox 24cm fond triple épaisseur", "expected_label": "casserole"},
    {"product_id": "EQ010", "description": "Poêle anti-adhésive professionnelle 28cm", "expected_label": "poele"},
    {"product_id": "EQ011", "description": "Spatule inox coudée 25cm manche isolant", "expected_label": "spatule"},
    {"product_id": "EQ012", "description": "Louche inox 12cl manche long 35cm", "expected_label": "louche"},
    {"product_id": "EQ013", "description": "Passoire fine inox 20cm avec manche", "expected_label": "passoire"},
    {"product_id": "EQ014", "description": "Récipient hermétique 2L transparent", "expected_label": "boite hermetique"},
    {"product_id": "EQ015", "description": "Tablier de cuisine plastifié 90cm", "expected_label": "tablier"},
]

async def classify_single_product(session, product):
    """Classify a single product via API"""
    try:
        async with session.post(
            'http://localhost:8000/classify',
            json={"designation": product["description"], "product_id": product["product_id"]},
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            if response.status == 200:
                result = await response.json()
                
                # Extract the decision stage/node from path_taken
                decision_node = extract_decision_node(result['path_taken'])
                
                # Get expected label
                expected_label = product.get("expected_label", "unknown")
                
                # Check if prediction is correct (case insensitive)
                predicted = result['final_label'].lower().strip()
                expected = expected_label.lower().strip()
                is_correct = (predicted == expected)
                
                return {
                    'product_id': product["product_id"],
                    'description': product["description"],
                    'predicted_label': result['final_label'],
                    'expected_label': expected_label,
                    'is_correct': is_correct,
                    'confidence': result['confidence'],
                    'processing_time_ms': result['processing_time_ms'],
                    'cost_usd': result.get('cost_usd', 0.0),
                    'path_taken': str(result['path_taken']),
                    'decision_node': decision_node,
                    'status': 'success',
                    'category': get_category_from_product_id(product["product_id"])
                }
            else:
                error_text = await response.text()
                return {
                    'product_id': product["product_id"],
                    'description': product["description"],
                    'predicted_label': f'ERROR_{response.status}',
                    'expected_label': product.get("expected_label", "unknown"),
                    'is_correct': False,
                    'confidence': 0.0,
                    'processing_time_ms': 0.0,
                    'cost_usd': 0.0,
                    'path_taken': f'Error: {error_text}',
                    'decision_node': 'error',
                    'status': 'error',
                    'category': get_category_from_product_id(product["product_id"])
                }
    except Exception as e:
        return {
            'product_id': product["product_id"],
            'description': product["description"],
            'predicted_label': f'EXCEPTION',
            'expected_label': product.get("expected_label", "unknown"),
            'is_correct': False,
            'confidence': 0.0,
            'processing_time_ms': 0.0,
            'cost_usd': 0.0,
            'path_taken': f'Exception: {str(e)}',
            'decision_node': 'exception',
            'status': 'exception',
            'category': get_category_from_product_id(product["product_id"])
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
    
    # Si T5 était confiant et a terminé
    elif "t5_pred_" in path_str and "gpt" not in path_str.lower():
        return "t5"
    
    # Si on est allé jusqu'au GPT/orchestrator
    elif any(keyword in path_str.lower() for keyword in ["gpt", "orchestrator", "arbitrage"]):
        return "llm"
    
    # Analyse plus fine du path_taken
    elif "db_uncertain" in path_str and "t5_pred_" in path_str:
        # Si on a T5 mais pas de continuation vers GPT
        return "t5" if "gpt" not in path_str.lower() else "llm"
    
    else:
        return "unknown"

def get_category_from_product_id(product_id):
    """Extract category from product ID"""
    if product_id.startswith(("FR", "CH", "PS", "VI", "EP")):
        return "Alimentaire"
    elif product_id.startswith(("VN", "CP", "BR", "SP")):
        return "Boissons"
    elif product_id.startswith("CL"):
        return "Entretien"
    elif product_id.startswith("EQ"):
        return "Matériel"
    else:
        return "Autre"

async def run_evaluation():
    """Run the complete evaluation test - French products only"""
    print("🇫🇷 Évaluation Système de Classification - Produits 100% Français")
    print(f"📊 Test de {len(TEST_PRODUCTS)} produits pour restaurants français")
    print("🍷 Focus: Vins français, fromages, charcuterie, produits du terroir")
    print("🎯 Labels de vérité terrain inclus pour calcul de précision")
    print("=" * 80)
    
    # Check API health
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/health', timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    print("❌ API Health check failed!")
                    return
                print("✅ API est opérationnelle")
    except Exception as e:
        print(f"❌ Impossible de se connecter à l'API: {e}")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.time()
    results = []
    
    # Process products with concurrency limit to avoid overwhelming the API
    semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
    
    async def process_with_semaphore(session, product):
        async with semaphore:
            return await classify_single_product(session, product)
    
    async with aiohttp.ClientSession() as session:
        tasks = [process_with_semaphore(session, product) for product in TEST_PRODUCTS]
        
        # Show progress
        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1
            category = result['category']
            accuracy_symbol = "✅" if result['is_correct'] else "❌"
            print(f"📈 Progression: {completed}/{len(TEST_PRODUCTS)} ({(completed/len(TEST_PRODUCTS)*100):.1f}%) - {category}: {result['product_id']} {accuracy_symbol}")
    
    # Calculate totals
    total_time = time.time() - start_time
    total_cost = sum((r.get('cost_usd') or 0.0) for r in results)
    total_processing_time = sum((r.get('processing_time_ms') or 0.0) for r in results)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Clean numeric columns
    df['cost_usd'] = pd.to_numeric(df['cost_usd'], errors='coerce').fillna(0.0)
    df['processing_time_ms'] = pd.to_numeric(df['processing_time_ms'], errors='coerce').fillna(0.0)

    # Calculate accuracy metrics
    successful_predictions = df[df['status'] == 'success']
    total_correct = len(successful_predictions[successful_predictions['is_correct'] == True])
    overall_accuracy = (total_correct / len(successful_predictions) * 100) if len(successful_predictions) > 0 else 0
    
    # Accuracy by decision node
    accuracy_by_node = successful_predictions.groupby('decision_node')['is_correct'].agg(['count', 'sum', 'mean']).round(4)
    accuracy_by_node.columns = ['total_predictions', 'correct_predictions', 'accuracy_rate']
    
    # Accuracy by category
    accuracy_by_category = successful_predictions.groupby('category')['is_correct'].agg(['count', 'sum', 'mean']).round(4)
    accuracy_by_category.columns = ['total_predictions', 'correct_predictions', 'accuracy_rate']

    # Create summary statistics
    summary_stats = {
        'timestamp': timestamp,
        'total_products': len(TEST_PRODUCTS),
        'successful_classifications': len(successful_predictions),
        'failed_classifications': len(df[df['status'] != 'success']),
        'success_rate_pct': (len(successful_predictions) / len(TEST_PRODUCTS) * 100),
        'overall_accuracy_pct': overall_accuracy,
        'correct_predictions': total_correct,
        'total_evaluation_time_sec': total_time,
        'total_api_processing_time_ms': total_processing_time,
        'total_cost_usd': total_cost,
        'average_processing_time_ms': successful_predictions['processing_time_ms'].mean() or 0.0,
        'average_cost_per_product_usd': (total_cost / len(TEST_PRODUCTS)) if len(TEST_PRODUCTS) > 0 else 0.0,
        'average_confidence': successful_predictions['confidence'].mean() or 0.0
    }
    
    # Add node distribution to summary
    summary_stats.update({
        'database_decisions': len(df[df['decision_node'] == 'database']),
        't5_decisions': len(df[df['decision_node'] == 't5']),
        'llm_decisions': len(df[df['decision_node'] == 'llm']),
        'database_percentage': len(df[df['decision_node'] == 'database']) / len(df) * 100,
        't5_percentage': len(df[df['decision_node'] == 't5']) / len(df) * 100,
        'llm_percentage': len(df[df['decision_node'] == 'llm']) / len(df) * 100,
    })
    
    # Category analysis
    print("\n" + "=" * 80)
    print("📋 RÉSUMÉ DE L'ÉVALUATION - PRODUITS FRANÇAIS")
    print("=" * 80)
    print(f"📊 Produits testés: {summary_stats['total_products']}")
    print(f"✅ Classifications réussies: {summary_stats['successful_classifications']}")
    print(f"❌ Classifications échouées: {summary_stats['failed_classifications']}")
    print(f"📈 Taux de succès API: {summary_stats['success_rate_pct']:.1f}%")
    print(f"🎯 Précision des prédictions: {summary_stats['overall_accuracy_pct']:.1f}% ({total_correct}/{len(successful_predictions)})")
    print(f"⏱️  Temps total d'évaluation: {summary_stats['total_evaluation_time_sec']:.1f} secondes")
    print(f"⚡ Temps de traitement moyen: {summary_stats['average_processing_time_ms']:.1f} ms")
    print(f"💰 Coût total: ${summary_stats['total_cost_usd']:.4f}")
    print(f"💵 Coût moyen par produit: ${summary_stats['average_cost_per_product_usd']:.4f}")
    print(f"🎯 Confiance moyenne: {summary_stats['average_confidence']:.3f}")

    print(f"\n🎯 PRÉCISION PAR NOEUD DE DÉCISION:")
    print(accuracy_by_node)
    
    print(f"\n🎯 PRÉCISION PAR CATÉGORIE:")
    print(accuracy_by_category)
    
    # Count by category
    category_counts = df['category'].value_counts()
    print(f"\n📊 DISTRIBUTION DES PRODUITS:")
    for category, count in category_counts.items():
        print(f"   • {category}: {count} produits")
    
    # Show some prediction examples
    print(f"\n🔍 EXEMPLES DE PRÉDICTIONS:")
    correct_samples = successful_predictions[successful_predictions['is_correct'] == True].head(5)
    incorrect_samples = successful_predictions[successful_predictions['is_correct'] == False].head(5)
    
    if not correct_samples.empty:
        print("\n✅ PRÉDICTIONS CORRECTES:")
        for _, row in correct_samples.iterrows():
            print(f"   • {row['product_id']}: '{row['predicted_label']}' = '{row['expected_label']}' ✓")
    
    if not incorrect_samples.empty:
        print("\n❌ PRÉDICTIONS INCORRECTES:")
        for _, row in incorrect_samples.iterrows():
            print(f"   • {row['product_id']}: '{row['predicted_label']}' ≠ '{row['expected_label']}' ✗")
    
    # Save results
    summary_filename = f"evaluation_french_{timestamp}.csv"
    detailed_filename = f"evaluation_french_detailed_{timestamp}.csv"
    df.to_csv(detailed_filename, index=False)
    pd.DataFrame([summary_stats]).to_csv(summary_filename, index=False)
    
    print(f"\n💾 Résultats sauvegardés:")
    print(f"   • Détaillé: {detailed_filename}")
    print(f"   • Résumé: {summary_filename}")
    
    # Cost analysis by decision node
    print(f"\n💰 ANALYSE DES COÛTS PAR NOEUD:")
    cost_by_node = df.groupby('decision_node')['cost_usd'].agg(['count', 'sum', 'mean']).round(6)
    print(cost_by_node)
    
    print(f"\n💡 INSIGHTS COÛTS:")
    llm_products = len(df[df['decision_node'] == 'llm'])
    total_llm_cost = df[df['decision_node'] == 'llm']['cost_usd'].sum()
    db_products = len(df[df['decision_node'] == 'database']) 
    t5_products = len(df[df['decision_node'] == 't5'])
    
    print(f"   • Base de données: {db_products} produits → $0.000 (gratuit)")
    print(f"   • Modèle T5: {t5_products} produits → $0.000 (gratuit)")
    print(f"   • Arbitrage LLM: {llm_products} produits → ${total_llm_cost:.6f}")
    print(f"   • Coût moyen par appel LLM: ${(total_llm_cost/llm_products if llm_products > 0 else 0):.6f}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())