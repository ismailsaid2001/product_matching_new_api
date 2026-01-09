from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from agent.graph import app_langgraph

app = FastAPI(title="Agentic Product Classifier")

# Thread pool for CPU-bound operations
executor = ThreadPoolExecutor(max_workers=4)

class ClassificationRequest(BaseModel):
    designation: str
    product_id: Optional[str] = None

class ClassificationResponse(BaseModel):
    final_label: str
    confidence: float
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

def process_single_product(designation: str, product_id: Optional[str] = None):
    """Synchronous product classification for thread pool execution"""
    start_time = time.time()
    
    initial_state = {
        "description": designation,
        "step_history": []
    }
    
    result = app_langgraph.invoke(initial_state)
    
    processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    # Extract cost if available from the result
    cost_usd = None
    if result.get("cost_info"):
        cost_usd = result["cost_info"].get("total_cost_usd", 0.0)
    
    return {
        "final_label": result["final_label"],
        "confidence": result.get("confidence", result.get("t5_confidence", 1.0)),
        "path_taken": result["step_history"],
        "processing_time_ms": processing_time,
        "cost_usd": cost_usd,
        "product_id": product_id
    }

@app.post("/classify", response_model=ClassificationResponse)
async def classify_product(request: ClassificationRequest):
    """Single product classification"""
    try:
        # Execute in thread pool to prevent blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            process_single_product, 
            request.designation, 
            request.product_id
        )
        return ClassificationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify/batch", response_model=BatchClassificationResponse)
async def classify_products_batch(request: BatchClassificationRequest):
    """Parallel batch classification"""
    try:
        start_time = time.time()
        
        # Create tasks for parallel execution
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                executor,
                process_single_product,
                product.designation,
                product.product_id
            )
            for product in request.products
        ]
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks)
        
        total_time = (time.time() - start_time) * 1000
        total_cost = sum(r.get("cost_usd", 0) or 0 for r in results)
        
        return BatchClassificationResponse(
            results=[ClassificationResponse(**r) for r in results],
            total_processing_time_ms=total_time,
            total_cost_usd=total_cost
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Product classifier is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)  # Single worker for thread safety