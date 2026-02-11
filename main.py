from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from agent.graph import app_langgraph

app = FastAPI(title="Product Classification API")

# TODO: might need to tune this number based on server specs
thread_pool = ThreadPoolExecutor(max_workers=4)

class ClassificationRequest(BaseModel):
    designation: str
    product_id: Optional[str] = None

class ClassificationResponse(BaseModel):
    final_label: str
    confidence: float
    database_confidence: Optional[float] = None
    database_prediction: Optional[str] = None
    t5_confidence: Optional[float] = None
    t5_prediction: Optional[str] = None
    path_taken: list
    processing_time_ms: float
    cost_usd: Optional[float] = None
    product_id: Optional[str] = None

class BatchClassificationRequest(BaseModel):
    products: List[ClassificationRequest]

class BatchClassificationResponse(BaseModel):
    results: List[ClassificationResponse]
    total_processing_time_ms: float
    total_cost_usd: float

def classify_single_item(designation: str, product_id: Optional[str] = None):
    """Process one product at a time"""
    start = time.time()  # track timing
    
    # setup initial state for the graph
    initial_state = {
        "description": designation,
        "step_history": []
    }
    
    result = app_langgraph.invoke(initial_state)
    
    proc_time = (time.time() - start) * 1000  # ms
    
    # Extract cost if available from the result
    total_cost = None
    if result.get("cost_info"):
        total_cost = result["cost_info"].get("total_cost_usd", 0.0)
    
    return {
        "final_label": result["final_label"],
        "confidence": result.get("confidence", result.get("t5_confidence", 1.0)),
        "database_confidence": result.get("database_confidence"),
        "database_prediction": result.get("database_prediction"),
        "t5_confidence": result.get("t5_confidence"),
        "t5_prediction": result.get("t5_prediction"),
        "path_taken": result["step_history"],
        "processing_time_ms": proc_time,
        "cost_usd": total_cost,
        "product_id": product_id
    }

@app.post("/classify", response_model=ClassificationResponse)
async def classify_product(request: ClassificationRequest):
    """Single product classification"""
    try:
        # run in separate thread to avoid blocking the main loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            thread_pool, 
            classify_single_item, 
            request.designation, 
            request.product_id
        )
        return ClassificationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify/batch", response_model=BatchClassificationResponse)
async def classify_products_batch(request: BatchClassificationRequest):
    """Batch classification - runs multiple products in parallel"""
    try:
        batch_start = time.time()
        
        # Create tasks for parallel execution
        loop = asyncio.get_event_loop()
        task_list = [
            loop.run_in_executor(
                thread_pool,
                classify_single_item,
                prod.designation,
                prod.product_id
            )
            for prod in request.products
        ]
        
        # Execute all tasks in parallel
        batch_results = await asyncio.gather(*task_list)
        
        batch_time = (time.time() - batch_start) * 1000
        batch_cost = sum(r.get("cost_usd", 0) or 0 for r in batch_results)
        
        return BatchClassificationResponse(
            results=[ClassificationResponse(**r) for r in batch_results],
            total_processing_time_ms=batch_time,
            total_cost_usd=batch_cost
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_stuff():
    """Load models when app starts up"""
    print("Loading models...")
    from services.t5_service import T5ModelService
    
    # Load T5 model 
    model_service = T5ModelService.get_instance()
    print("All set!")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Product classifier is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)  # Single worker for thread safety