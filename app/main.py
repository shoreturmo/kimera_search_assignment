from fastapi import FastAPI, HTTPException
import numpy as np
import os
import time

# --- Configuration ---
DATA_DIR = "/app/data"
RAW_EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.bin")
EMBEDDING_DIM = 128
NUM_EMBEDDINGS = int(os.getenv("NUM_EMBEDDINGS", 10000))

app = FastAPI(
    title="Kimera High-Performance Search API",
    description="An API for finding approximate nearest neighbors in a vector dataset."
)

# --- Global State ---
raw_embeddings: np.ndarray = None
# This variable is available for you to store any data structures
# that you prepare on startup to make subsequent searches faster.
search_accelerator = None

@app.on_event("startup")
def startup_event():
    """
    On startup, load the raw embeddings from disk. This is a good place
    to perform any one-time setup that your search method may require.
    """
    global raw_embeddings, search_accelerator

    if not os.path.exists(RAW_EMBEDDINGS_FILE):
        raise RuntimeError(f"Data file not found at '{RAW_EMBEDDINGS_FILE}'. Ensure Docker CMD generates it.")

    print(f"Loading {NUM_EMBEDDINGS} embeddings from '{RAW_EMBEDDINGS_FILE}'...")


    raw_embeddings = np.fromfile(RAW_EMBEDDINGS_FILE, dtype=np.float32).reshape(
        NUM_EMBEDDINGS, EMBEDDING_DIM
    )
    
    # TODO: Perform any one-time setup needed for your search implementation.
    # The result of this can be stored in the global `search_accelerator` variable.
    print("Performing one-time setup for search...")
    # --- YOUR SETUP LOGIC GOES HERE ---
    
    
    # ------------------------------------
    print("Service is ready.")


@app.get("/search")
def search(item_id: int, num_neighbors: int):
    start_time = time.time()
    
    # Get the query vector for the given item_id.
    query_vector = np.expand_dims(raw_embeddings[item_id], axis=0)

    # TODO: Implement your search logic here to find the nearest neighbors.
    # Your implementation will be evaluated on its performance.
    
    # --- YOUR SEARCH LOGIC GOES HERE ---
    
    # Placeholder values for the search results.
    # Your logic should populate these with the actual indices and distances/scores.
    distances = np.array([[]])
    indices = np.array([[]])

    # -------------------------------------
    
    results = []
    for i in range(len(indices[0])):
        idx = int(indices[0][i])
        if idx != item_id:
            results.append({
                "item_id": idx,
                "score": float(distances[0][i])
            })
    
    search_time = time.time() - start_time

    return {
        "query_item_id": item_id, 
        "results": results[:num_neighbors], 
        "time_ms": search_time * 1000
    }
