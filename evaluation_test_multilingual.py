# evaluation_test.py
import asyncio
import aiohttp
import pandas as pd
import time
from datetime import datetime

# Test products focused on restaurant/food service business - 150 examples
TEST_PRODUCTS = [
    # === FOOD PRODUCTS (50 items) ===
    
    # French Food Products
    {"product_id": "FR001", "description": "Emmental râpé Président 200g sachet refermable", "expected_label": "emmental rape"},
    {"product_id": "FR002", "description": "Filet de saumon frais Ecosse portion 150g", "expected_label": "filet saumon"},
    {"product_id": "FR003", "description": "Jambon cuit supérieur Fleury Michon 4 tranches", "expected_label": "jambon cuit"},
    {"product_id": "FR004", "description": "Beurre demi-sel Président plaquette 250g", "expected_label": "beurre sel 250g"},
    {"product_id": "FR005", "description": "Pâtes tagliatelles fraîches aux œufs 400g", "expected_label": "pate tagliatelle"},
    {"product_id": "FR006", "description": "Huile d'olive extra vierge Puget 500ml", "expected_label": "huile olive extra"},
    {"product_id": "FR007", "description": "Tomates cerises grappe origine France 250g", "expected_label": "tomate cerise"},
    {"product_id": "FR008", "description": "Pain de mie complet Harry's 14 tranches", "expected_label": "pain mie"},
    {"product_id": "FR009", "description": "Crème fraîche épaisse 30% Elle & Vire 200ml", "expected_label": "creme fraiche"},
    {"product_id": "FR010", "description": "Escalope de dinde fermière blanc 4 pièces", "expected_label": "escalope dinde"},
    
    # Italian Food Products  
    {"product_id": "IT001", "description": "Parmigiano Reggiano DOP stagionato 24 mesi 200g", "expected_label": "parmesan"},
    {"product_id": "IT002", "description": "Prosciutto di Parma DOP affettato 80g", "expected_label": "jambon cru tranche"},
    {"product_id": "IT003", "description": "Pasta spaghetti Barilla n°5 500g", "expected_label": "spaghetti"},
    {"product_id": "IT004", "description": "Pomodori San Marzano DOP pelati 400g", "expected_label": "tomate pelees"},
    {"product_id": "IT005", "description": "Olio extravergine Colavita 750ml bottiglia", "expected_label": "huile olive extra"},
    {"product_id": "IT006", "description": "Mozzarella di bufala Campana DOP 125g", "expected_label": "mozzarella"},
    {"product_id": "IT007", "description": "Risotto Carnaroli Scotti 1kg sacchetto", "expected_label": "riz carnaroli"},
    {"product_id": "IT008", "description": "Bresaola della Valtellina IGP 70g", "expected_label": "bresaola"},
    {"product_id": "IT009", "description": "Basilico fresco in vaso 40g", "expected_label": "basilic frais"},
    {"product_id": "IT010", "description": "Gorgonzola dolce DOP 200g portion", "expected_label": "gorgonzola"},
    
    # Spanish Food Products
    {"product_id": "ES001", "description": "Jamón ibérico bellota Cinco Jotas 80g lonchas", "expected_label": "jambon cru tranche"},
    {"product_id": "ES002", "description": "Aceite oliva virgen extra Carbonell 1L", "expected_label": "huile olive extra"},
    {"product_id": "ES003", "description": "Chorizo dulce Extra Casademont 200g", "expected_label": "chorizo"},
    {"product_id": "ES004", "description": "Arroz bomba Calasparra DO 1kg", "expected_label": "riz"},
    {"product_id": "ES005", "description": "Queso manchego curado 12 meses 250g", "expected_label": "manchego"},
    {"product_id": "ES006", "description": "Pimientos del piquillo Navarrico 290g lata", "expected_label": "poivron rouge"},
    {"product_id": "ES007", "description": "Merluza lomos congelados 400g bolsa", "expected_label": "filet colin"},
    {"product_id": "ES008", "description": "Gambas rojas Dénia 500g bandeja", "expected_label": "crevette"},
    {"product_id": "ES009", "description": "Tomate frito Orlando 350g brick", "expected_label": "tomate concassee"},
    {"product_id": "ES010", "description": "Lomo embuchado ibérico lonchas 100g", "expected_label": "longe porc"},
    
    # English Food Products
    {"product_id": "EN001", "description": "Aberdeen Angus beef fillet 250g steak", "expected_label": "filet boeuf"},
    {"product_id": "EN002", "description": "Smoked salmon Scottish premium 100g sliced", "expected_label": "saumon fume tranche"},
    {"product_id": "EN003", "description": "Cheddar cheese mature 12 months 200g", "expected_label": "cheddar"},
    {"product_id": "EN004", "description": "Free-range eggs large size 12 pack", "expected_label": "oeuf frais"},
    {"product_id": "EN005", "description": "Extra virgin olive oil Filippo Berio 500ml", "expected_label": "huile olive extra"},
    {"product_id": "EN006", "description": "Organic chicken breast fillets 400g pack", "expected_label": "filet poulet"},
    {"product_id": "EN007", "description": "Wholemeal bread loaf sliced 800g", "expected_label": "pain complet"},
    {"product_id": "EN008", "description": "Greek yogurt natural thick 500g pot", "expected_label": "yaourt grec"},
    {"product_id": "EN009", "description": "Wild sea bass fillets fresh 300g", "expected_label": "filet bar"},
    {"product_id": "EN010", "description": "Organic honey wildflower 340g jar", "expected_label": "miel"},
    
    # German Food Products
    {"product_id": "DE001", "description": "Schwarzwälder Schinken geräuchert 150g", "expected_label": "jambon foret noire"},
    {"product_id": "DE002", "description": "Leberwurst fein Rügenwalder 125g", "expected_label": "pate foie"},
    {"product_id": "DE003", "description": "Sauerkraut Hengstenberg 680g Glas", "expected_label": "choucroute"},
    {"product_id": "DE004", "description": "Pumpernickel Vollkorn Mestemacher 500g", "expected_label": "pain noir"},
    {"product_id": "DE005", "description": "Weißwurst München original 4 Stück", "expected_label": "saucisse blanche"},
    {"product_id": "DE006", "description": "Spätzle schwäbisch Bürger 500g", "expected_label": "spatzle"},
    {"product_id": "DE007", "description": "Gouda jung Holland 200g Scheiben", "expected_label": "gouda"},
    {"product_id": "DE008", "description": "Rinderfilet Argentinien 300g Steak", "expected_label": "filet boeuf"},
    {"product_id": "DE009", "description": "Forelle geräuchert ganz 250g", "expected_label": "truite fumee"},
    {"product_id": "DE010", "description": "Senf scharf Löwensenf 200ml Tube", "expected_label": "moutarde"},
    
    # === BEVERAGES - ALCOHOLIC (60 items) ===
    
    # Beer - French Brands
    {"product_id": "BE001", "description": "Kronenbourg 1664 blonde 6x25cl pack bouteilles", "expected_label": "biere 1664"},
    {"product_id": "BE002", "description": "Heineken blonde 12x33cl canettes pack", "expected_label": "biere"},
    {"product_id": "BE003", "description": "Stella Artois premium lager 4x50cl bouteilles", "expected_label": "biere"},
    {"product_id": "BE004", "description": "Leffe blonde abbaye 6x33cl bouteilles belge", "expected_label": "biere"},
    {"product_id": "BE005", "description": "Guinness stout 4x44cl canettes irlandaise", "expected_label": "biere"},
    {"product_id": "BE006", "description": "Corona Extra lime 6x35.5cl bouteilles mexicaine", "expected_label": "biere"},
    {"product_id": "BE007", "description": "Hoegaarden blanche 6x25cl bouteilles belge", "expected_label": "biere"},
    {"product_id": "BE008", "description": "Desperados tequila beer 3x33cl bouteilles", "expected_label": "biere"},
    {"product_id": "BE009", "description": "Carlsberg pilsner 8x50cl canettes danoise", "expected_label": "biere"},
    {"product_id": "BE010", "description": "Chimay rouge trappiste 33cl bouteille belge", "expected_label": "biere"},
    
    # Beer - Italian Market
    {"product_id": "BE011", "description": "Peroni Nastro Azzurro 6x33cl lattine premium", "expected_label": "biere"},
    {"product_id": "BE012", "description": "Moretti birra italiana 12x33cl bottiglie", "expected_label": "biere"},
    {"product_id": "BE013", "description": "Beck's pils tedesca 6x33cl bottiglie verdi", "expected_label": "biere"},
    {"product_id": "BE014", "description": "Menabrea birra piemontese 6x33cl bottiglie", "expected_label": "biere"},
    {"product_id": "BE015", "description": "Ichnusa birra sarda 6x33cl bottiglie", "expected_label": "biere"},
    
    # Beer - Spanish Market  
    {"product_id": "BE016", "description": "Mahou Cinco Estrellas 6x25cl botellines", "expected_label": "biere"},
    {"product_id": "BE017", "description": "Estrella Damm lager 12x33cl latas", "expected_label": "biere"},
    {"product_id": "BE018", "description": "San Miguel especial 6x25cl botellas", "expected_label": "biere"},
    {"product_id": "BE019", "description": "Alhambra reserva 1925 6x33cl botellas", "expected_label": "biere"},
    {"product_id": "BE020", "description": "Cruzcampo pilsen 8x33cl latas pack", "expected_label": "biere"},
    
    # Beer - German Brands
    {"product_id": "BE021", "description": "Löwenbräu Original 6x33cl Flaschen", "expected_label": "biere"},
    {"product_id": "BE022", "description": "Warsteiner Premium Verum 12x33cl Dosen", "expected_label": "biere"},
    {"product_id": "BE023", "description": "Augustiner Lagerbier Hell 6x50cl Flaschen", "expected_label": "biere"},
    {"product_id": "BE024", "description": "Erdinger Weissbier 6x50cl Flaschen", "expected_label": "biere"},
    {"product_id": "BE025", "description": "Spaten Münchner Hell 8x50cl Dosen", "expected_label": "biere"},
    
    # Champagne & Sparkling Wine
    {"product_id": "CH001", "description": "Champagne Veuve Clicquot Brut 75cl coffret", "expected_label": "champagne"},
    {"product_id": "CH002", "description": "Dom Pérignon Vintage 2015 75cl bouteille", "expected_label": "champagne"},
    {"product_id": "CH003", "description": "Champagne Moët & Chandon Impérial 75cl", "expected_label": "champagne"},
    {"product_id": "CH004", "description": "Laurent-Perrier Brut 75cl champagne", "expected_label": "champagne"},
    {"product_id": "CH005", "description": "Taittinger Comtes de Champagne Blanc 75cl", "expected_label": "champagne"},
    {"product_id": "CH006", "description": "Krug Grande Cuvée 75cl champagne prestige", "expected_label": "champagne"},
    {"product_id": "CH007", "description": "Champagne Pol Roger Brut Réserve 75cl", "expected_label": "champagne"},
    {"product_id": "CH008", "description": "Bollinger Special Cuvée 75cl champagne", "expected_label": "champagne"},
    {"product_id": "CH009", "description": "Perrier-Jouët Belle Epoque 75cl champagne", "expected_label": "champagne"},
    {"product_id": "CH010", "description": "Prosecco Valdobbiadene DOCG 75cl Italie", "expected_label": "prosecco"},
    {"product_id": "CH011", "description": "Cava Freixenet Cordon Negro 75cl Espagne", "expected_label": "cava"},
    {"product_id": "CH012", "description": "Crémant de Loire Bouvet Ladubay 75cl", "expected_label": "cremant"},
    
    # Tequila & Mexican Spirits
    {"product_id": "TE001", "description": "Tequila José Cuervo Especial Gold 70cl", "expected_label": "tequila"},
    {"product_id": "TE002", "description": "Patron Silver tequila premium 100% agave 70cl", "expected_label": "tequila"},
    {"product_id": "TE003", "description": "Don Julio Blanco tequila 70cl Mexico", "expected_label": "tequila"},
    {"product_id": "TE004", "description": "Herradura Silver tequila 100% agave 70cl", "expected_label": "tequila"},
    {"product_id": "TE005", "description": "Sauza Gold tequila mixto 70cl bouteille", "expected_label": "tequila"},
    {"product_id": "TE006", "description": "Olmeca Altos Plata tequila 70cl premium", "expected_label": "tequila"},
    {"product_id": "TE007", "description": "Espolòn Tequila Blanco 70cl 100% agave", "expected_label": "tequila"},
    {"product_id": "TE008", "description": "1800 Silver tequila premium 70cl Mexico", "expected_label": "tequila"},
    
    # Wine & Spirits
    {"product_id": "WI001", "description": "Bordeaux Saint-Émilion Grand Cru 75cl 2020", "expected_label": "vin rouge"},
    {"product_id": "WI002", "description": "Chianti Classico DOCG Riserva 75cl 2019", "expected_label": "vin rouge"},
    {"product_id": "WI003", "description": "Rioja Crianza Tempranillo 75cl 2021", "expected_label": "vin rouge"},
    {"product_id": "WI004", "description": "Sancerre Loire Valley 75cl 2022", "expected_label": "vin blanc"},
    {"product_id": "WI005", "description": "Whisky Johnnie Walker Black Label 70cl", "expected_label": "whisky"},
    {"product_id": "WI006", "description": "Cognac Hennessy VS 70cl bouteille", "expected_label": "cognac"},
    {"product_id": "WI007", "description": "Vodka Grey Goose premium 70cl France", "expected_label": "vodka"},
    {"product_id": "WI008", "description": "Gin Hendrick's premium 70cl Ecosse", "expected_label": "gin"},
    
    # === CLEANING SUPPLIES (20 items) ===
    {"product_id": "CL001", "description": "Liquide vaisselle professionnel Ecover 5L", "expected_label": "liquide vaisselle"},
    {"product_id": "CL002", "description": "Dégraissant cuisine professionnel 750ml spray", "expected_label": "degraissant"},
    {"product_id": "CL003", "description": "Nettoyant désinfectant surfaces Sanytol 1L", "expected_label": "desinfectant"},
    {"product_id": "CL004", "description": "Tablettes lave-vaisselle Finish Quantum 50 unités", "expected_label": "tablette lave vaisselle"},
    {"product_id": "CL005", "description": "Lingettes désinfectantes Javel 100 unités", "expected_label": "lingette desinfectante"},
    {"product_id": "CL006", "description": "Produit nettoyant four décapant 500ml", "expected_label": "nettoyant four"},
    {"product_id": "CL007", "description": "Détartrant machine café professionnel 1L", "expected_label": "detartrant"},
    {"product_id": "CL008", "description": "Nettoyant vitres professionnel 5L bidon", "expected_label": "nettoyant vitre"},
    {"product_id": "CL009", "description": "Savon main antibactérien distributeur 500ml", "expected_label": "savon main"},
    {"product_id": "CL010", "description": "Eau de Javel concentrée 12° 1L", "expected_label": "javel"},
    {"product_id": "CL011", "description": "Essuie-tout professionnel 6 rouleaux", "expected_label": "essuie tout"},
    {"product_id": "CL012", "description": "Gants jetables nitrile bleu taille L 100 unités", "expected_label": "gant jetable"},
    {"product_id": "CL013", "description": "Sacs poubelle 50L noir 20 unités", "expected_label": "sac poubelle"},
    {"product_id": "CL014", "description": "Produit sol restaurant 5L concentré", "expected_label": "nettoyant sol"},
    {"product_id": "CL015", "description": "Désodorisant professionnel Febreze 300ml", "expected_label": "desodorisant"},
    {"product_id": "CL016", "description": "Pastilles désinfectantes Sterifree 100 unités", "expected_label": "pastille desinfectante"},
    {"product_id": "CL017", "description": "Nettoyant inox professionnel 750ml", "expected_label": "nettoyant inox"},
    {"product_id": "CL018", "description": "Papier toilette professionnel Tork 12 rouleaux", "expected_label": "papier toilette"},
    {"product_id": "CL019", "description": "Film plastique alimentaire 300m x 30cm", "expected_label": "film plastique"},
    {"product_id": "CL020", "description": "Papier cuisson professionnel 50m rouleau", "expected_label": "papier cuisson"},
    
    # === RESTAURANT EQUIPMENT (20 items) ===
    {"product_id": "EQ001", "description": "Planche à découper polyéthylène 40x30cm blanche", "expected_label": "planche decouper"},
    {"product_id": "EQ002", "description": "Couteau chef professionnel 20cm lame acier", "expected_label": "couteau chef"},
    {"product_id": "EQ003", "description": "Bac gastronorme GN 1/1 inox 20mm hauteur", "expected_label": "bac gastronome"},
    {"product_id": "EQ004", "description": "Thermomètre sonde digital cuisine -50°C +200°C", "expected_label": "thermometre sonde"},
    {"product_id": "EQ005", "description": "Fouet professionnel inox 30cm manche plastique", "expected_label": "fouet"},
    {"product_id": "EQ006", "description": "Mandoline légumes réglable inox protection", "expected_label": "mandoline"},
    {"product_id": "EQ007", "description": "Balance cuisine digitale 5kg précision 1g", "expected_label": "balance cuisine"},
    {"product_id": "EQ008", "description": "Mixer plongeant professionnel 400W", "expected_label": "mixeur plongeant"},
    {"product_id": "EQ009", "description": "Casserole inox 24cm fond triple épaisseur", "expected_label": "casserole"},
    {"product_id": "EQ010", "description": "Poêle anti-adhésive professionnelle 28cm", "expected_label": "poele"},
    {"product_id": "EQ011", "description": "Spatule inox coudée 25cm manche plastique", "expected_label": "spatule"},
    {"product_id": "EQ012", "description": "Louche inox 12cl manche long 35cm", "expected_label": "louche"},
    {"product_id": "EQ013", "description": "Passoire fine inox 20cm manche", "expected_label": "passoire"},
    {"product_id": "EQ014", "description": "Minuteur digital cuisine 99min 59sec", "expected_label": "minuteur"},
    {"product_id": "EQ015", "description": "Gants anti-chaleur silicone jusqu'à 250°C", "expected_label": "gant chaleur"},
    {"product_id": "EQ016", "description": "Pince universelle inox 30cm cuisine", "expected_label": "pince cuisine"},
    {"product_id": "EQ017", "description": "Récipient hermétique 2L transparent", "expected_label": "boite hermetique"},
    {"product_id": "EQ018", "description": "Tablier cuisine plastifié 90cm lavable", "expected_label": "tablier"},
    {"product_id": "EQ019", "description": "Distributeur film plastique mural 45cm", "expected_label": "derouleur film"},
    {"product_id": "EQ020", "description": "Étiquettes alimentaires amovibles 500 unités", "expected_label": "etiquette alimentaire"},
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
                    'category': get_category_from_product_id(product["product_id"]),
                    'language': get_language_from_description(product["description"])
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
                    'category': get_category_from_product_id(product["product_id"]),
                    'language': get_language_from_description(product["description"])
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
            'category': get_category_from_product_id(product["product_id"]),
            'language': get_language_from_description(product["description"])
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
    if product_id.startswith(("FR", "IT", "ES", "EN", "DE")) and any(char.isdigit() for char in product_id):
        return "Food"
    elif product_id.startswith(("BE", "CH", "TE", "WI")):
        return "Beverages"
    elif product_id.startswith("CL"):
        return "Cleaning"
    elif product_id.startswith("EQ"):
        return "Equipment"
    else:
        return "Other"

def get_language_from_description(description):
    """Detect language from description keywords"""
    if any(word in description.lower() for word in ['le', 'la', 'les', 'du', 'de', 'avec', 'sans', 'pour']):
        return "French"
    elif any(word in description.lower() for word in ['della', 'con', 'senza', 'per', 'di', 'da']):
        return "Italian" 
    elif any(word in description.lower() for word in ['del', 'con', 'sin', 'para', 'de', 'y']):
        return "Spanish"
    elif any(word in description.lower() for word in ['with', 'without', 'for', 'and', 'the', 'of']):
        return "English"
    elif any(word in description.lower() for word in ['mit', 'ohne', 'für', 'und', 'der', 'die', 'das']):
        return "German"
    else:
        return "Mixed"

async def run_evaluation():
    """Run the complete evaluation test"""
    print("🚀 Starting Restaurant Product Classification Evaluation")
    print(f"📊 Testing {len(TEST_PRODUCTS)} products across multiple categories")
    print("🍺 Focus: Beverages (Beer, Tequila, Champagne), Food, Cleaning, Equipment")
    print("🎯 Ground truth labels included for accuracy calculation")
    print("=" * 70)
    
    # Check API health
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/health', timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    print("❌ API Health check failed!")
                    return
                print("✅ API is healthy")
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
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
            print(f"📈 Progress: {completed}/{len(TEST_PRODUCTS)} ({(completed/len(TEST_PRODUCTS)*100):.1f}%) - {category}: {result['product_id']} {accuracy_symbol}")
    
    # Calculate totals - Fix for cost calculation
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

    # Create summary statistics with accuracy
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
    
    # Node breakdown
    node_stats = df.groupby('decision_node').agg({
        'status': lambda x: (x == 'success').sum(),
        'confidence': 'mean',
        'processing_time_ms': 'mean',
        'cost_usd': 'sum'
    }).round(4)

    # Node breakdown by category
    node_category_stats = df.groupby(['category', 'decision_node']).agg({
        'status': lambda x: (x == 'success').sum(),
        'confidence': 'mean',
        'processing_time_ms': 'mean',
        'cost_usd': 'sum'
    }).round(4)
    
    # Language breakdown
    language_stats = df.groupby('language').agg({
        'status': lambda x: (x == 'success').sum(),
        'confidence': 'mean',
        'processing_time_ms': 'mean',
        'cost_usd': 'sum'
    }).round(4)
    
    # Category breakdown
    category_stats = df.groupby('category').agg({
        'status': lambda x: (x == 'success').sum(),
        'confidence': 'mean',
        'processing_time_ms': 'mean',
        'cost_usd': 'sum'
    }).round(4)
    
    # Category analysis
    print("\n" + "=" * 70)
    print("📋 RESTAURANT BUSINESS EVALUATION SUMMARY")
    print("=" * 70)
    print(f"📊 Total Products Tested: {summary_stats['total_products']}")
    print(f"✅ Successful Classifications: {summary_stats['successful_classifications']}")
    print(f"❌ Failed Classifications: {summary_stats['failed_classifications']}")
    print(f"📈 API Success Rate: {summary_stats['success_rate_pct']:.1f}%")
    print(f"🎯 Prediction Accuracy: {summary_stats['overall_accuracy_pct']:.1f}% ({total_correct}/{len(successful_predictions)})")
    print(f"⏱️  Total Evaluation Time: {summary_stats['total_evaluation_time_sec']:.1f} seconds")
    print(f"⚡ Average Processing Time: {summary_stats['average_processing_time_ms']:.1f} ms")
    print(f"💰 Total Cost: ${summary_stats['total_cost_usd']:.4f}")
    print(f"💵 Average Cost per Product: ${summary_stats['average_cost_per_product_usd']:.4f}")
    print(f"🎯 Average Confidence: {summary_stats['average_confidence']:.3f}")

    print(f"\n🎯 ACCURACY BY DECISION NODE:")
    print(accuracy_by_node)
    
    print(f"\n🎯 ACCURACY BY CATEGORY:")
    print(accuracy_by_category)
    
    print(f"\n🗂️ CATEGORY BREAKDOWN:")
    print(category_stats)
    
    print(f"\n🌍 LANGUAGE BREAKDOWN:")
    print(language_stats)
    
    # Count by category
    category_counts = df['category'].value_counts()
    print(f"\n📊 PRODUCT DISTRIBUTION:")
    for category, count in category_counts.items():
        print(f"   • {category}: {count} products")
    
    # Show some prediction examples
    print(f"\n🔍 SAMPLE PREDICTIONS:")
    correct_samples = successful_predictions[successful_predictions['is_correct'] == True].head(5)
    incorrect_samples = successful_predictions[successful_predictions['is_correct'] == False].head(5)
    
    if not correct_samples.empty:
        print("\n✅ CORRECT PREDICTIONS:")
        for _, row in correct_samples.iterrows():
            print(f"   • {row['product_id']}: '{row['predicted_label']}' = '{row['expected_label']}' ✓")
    
    if not incorrect_samples.empty:
        print("\n❌ INCORRECT PREDICTIONS:")
        for _, row in incorrect_samples.iterrows():
            print(f"   • {row['product_id']}: '{row['predicted_label']}' ≠ '{row['expected_label']}' ✗")
    
    # Save summary
    summary_filename = f"evaluation_summary_{timestamp}.csv"
    detailed_filename = f"evaluation_detailed_{timestamp}.csv"
    df.to_csv(detailed_filename, index=False)
    pd.DataFrame([summary_stats]).to_csv(summary_filename, index=False)
    
    print(f"\n💾 Results saved to:")
    print(f"   • Detailed: {detailed_filename}")
    print(f"   • Summary: {summary_filename}")
    
    print(f"\n🎯 DECISION NODE BREAKDOWN:")
    print(node_stats)

    print(f"\n🎯 NODE BY CATEGORY:")
    print(node_category_stats)

    # Pourcentage par node
    node_counts = df['decision_node'].value_counts()
    print(f"\n📈 NODE DISTRIBUTION:")
    for node, count in node_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   • {node}: {count} decisions ({percentage:.1f}%)")

    # Add cost analysis by decision node
    print(f"\n💰 COST ANALYSIS BY DECISION NODE:")
    cost_by_node = df.groupby('decision_node')['cost_usd'].agg(['count', 'sum', 'mean']).round(6)
    print(cost_by_node)
    
    print(f"\n💡 COST INSIGHTS:")
    llm_products = len(df[df['decision_node'] == 'llm'])
    total_llm_cost = df[df['decision_node'] == 'llm']['cost_usd'].sum()
    db_products = len(df[df['decision_node'] == 'database']) 
    t5_products = len(df[df['decision_node'] == 't5'])
    
    print(f"   • Database decisions: {db_products} products → $0.000 (free)")
    print(f"   • T5 model decisions: {t5_products} products → $0.000 (free)")
    print(f"   • LLM arbitration: {llm_products} products → ${total_llm_cost:.6f}")
    print(f"   • Average cost per LLM call: ${(total_llm_cost/llm_products if llm_products > 0 else 0):.6f}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())