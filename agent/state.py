from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    description: str               # original description
    api_suggestions: List[dict]    
    t5_prediction: Optional[str]
    t5_confidence: Optional[float]
    database_confidence: Optional[float]  # Database step confidence
    database_prediction: Optional[str]    # Database step prediction
    confidence: Optional[float]    # General confidence score (from database or T5)
    web_context: Optional[str]
    final_label: str                #nature_product_predicted
    step_history: List[str]         #Pour the debug
    cost_info: Optional[dict]       # Cost tracking information