from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import database_node, t5_node, orchestrator_node
from config import THRESHOLD_DATABASE, THRESHOLD_T5_CONF

def create_app():
    workflow = StateGraph(AgentState)

    # 1. Ajout des Nœuds
    workflow.add_node("check_db", database_node)
    workflow.add_node("t5_gen", t5_node)
    workflow.add_node("gpt_arbitrator", orchestrator_node)

    # 2. Définition du point d'entrée
    workflow.set_entry_point("check_db")

    # 3. Logique de routage après la DB
    def router_after_db(state):
        if state.get("final_label"): # Si un match parfait a été trouvé
            return "end"
        return "t5"

    # 4. Logique de routage après T5
    def router_after_t5(state):
        if state.get("final_label"): # Si T5 est confiant
            return "end"
        return "gpt"

    # 5. Liens entre les nœuds
    workflow.add_conditional_edges(
        "check_db",
        router_after_db,
        {"end": END, "t5": "t5_gen"}
    )

    workflow.add_conditional_edges(
        "t5_gen",
        router_after_t5,
        {"end": END, "gpt": "gpt_arbitrator"}
    )

    workflow.add_edge("gpt_arbitrator", END)

    return workflow.compile()

# Compilation de l'application LangGraph
app_langgraph = create_app()