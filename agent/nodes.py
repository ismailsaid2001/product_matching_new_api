# agent/nodes.py
from services.database_service import get_database_suggestions
from services.t5_service import T5ModelService
from services.llm_service import OrchestratorService
from agent.state import AgentState
from config import THRESHOLD_DATABASE, THRESHOLD_T5_CONF

def database_node(state: AgentState):
    print("--- ÉTAPE 1 : RECHERCHE BASE DE DONNÉES ---")
    suggestions = get_database_suggestions(state["description"])
    
    # Store database confidence and prediction regardless of threshold
    database_confidence = suggestions[0]['similarity_score'] if suggestions else 0.0
    database_prediction = suggestions[0]['nature_product'] if suggestions else None
    
    # Check the best score
    if suggestions and suggestions[0]['similarity_score'] >= THRESHOLD_DATABASE:
        similarity_score = suggestions[0]['similarity_score']
        return {
            "final_label": suggestions[0]['nature_product'],
            "confidence": similarity_score,  # Confidence = similarity score, not 1
            "database_confidence": database_confidence,
            "database_prediction": database_prediction,
            "api_suggestions": suggestions,
            "step_history": ["db_match_found"]
        }
    
    return {
        "database_confidence": database_confidence,
        "database_prediction": database_prediction,
        "api_suggestions": suggestions,
        "step_history": ["db_uncertain_calling_t5"]
    }

def t5_node(state: AgentState):
    print("--- ÉTAPE 2 : GÉNÉRATION LOCALE T5 ---")
    
    t5_service = T5ModelService.get_instance()
    prediction, confidence = t5_service.predict(state["description"])
    
    # Check if T5 is confident enough
    is_confident = confidence >= THRESHOLD_T5_CONF
    
    # If T5 is confident, prepare final label
    # Otherwise, leave state as is for GPT orchestrator intervention
    update = {
        "t5_prediction": prediction,
        "t5_confidence": confidence,
        "step_history": state["step_history"] + [f"t5_pred_{prediction}_conf_{confidence:.2f}"]
    }
    
    if is_confident:
        update["final_label"] = prediction
        update["confidence"] = confidence  # Set general confidence to T5 confidence
        
    return update


def orchestrator_node(state: AgentState):
    print("--- ÉTAPE 3 : ARBITRAGE GPT-5 & WEB SEARCH ---")
    
    # Fallback if orchestrator unavailable (missing API key, etc.)
    try:
        service = OrchestratorService()
    except Exception as e:
        print(f"Orchestrator indisponible, repli local: {e}")
        # Choice: if API has good suggestion, else T5, else empty
        fallback_label = None
        if state.get("api_suggestions"):
            fallback_label = state["api_suggestions"][0].get("nature_product")
        if not fallback_label and state.get("t5_prediction"):
            fallback_label = state["t5_prediction"]
        return {
            "final_label": fallback_label or "",
            "web_context": None,
            "step_history": state["step_history"] + ["orchestrator_unavailable_fallback"],
            "cost_info": None
        }

    web_info = None

    # GPT renders its verdict
    cost_info = None
    try:
        final_decision, cost_info = service.arbitrate(
            description=state["description"],
            t5_suggestion=state.get("t5_prediction"),
            t5_confidence=state.get("t5_confidence", 0.0),
            api_suggestions=state.get("api_suggestions", []),
            web_context=web_info,
        )
        
    except Exception as e:
        print(f"Arbitration failed, local fallback: {e}")
        final_decision = (state.get("api_suggestions") or [{}])[0].get("nature_product") or state.get("t5_prediction") or ""

    return {
        "final_label": final_decision,
        "web_context": web_info,
        "step_history": state["step_history"] + ["gpt_arbitration_completed"],
        "cost_info": cost_info
    }


