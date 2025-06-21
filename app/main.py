from fastapi import FastAPI, HTTPException
import numpy as np
import os
import time
import subprocess
from io import StringIO

# --- Configuration ---
DATA_DIR = "/app/data"
RAW_EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.bin")
SEARCH_INDEX_FILE = os.path.join(DATA_DIR, "search.index")
SEARCH_TOOL = "/app/search_core/search_tool"
EMBEDDING_DIM = 128
NUM_EMBEDDINGS = int(os.getenv("NUM_EMBEDDINGS", 10000))

app = FastAPI(
    title="Kimera High-Performance Search API",
    description="An API for finding approximate nearest neighbors in a vector dataset.",
)

# --- Global State ---
raw_embeddings: np.ndarray = None


@app.on_event("startup")
def startup_event():
    """
    On startup, load the raw embeddings from disk for query lookup.
    The C++ tool handles the actual search.
    """
    global raw_embeddings

    if not os.path.exists(RAW_EMBEDDINGS_FILE):
        raise RuntimeError(
            f"Data file not found at '{RAW_EMBEDDINGS_FILE}'. Ensure Docker CMD generates it."
        )

    if not os.path.exists(SEARCH_INDEX_FILE):
        raise RuntimeError(
            f"Search index not found at '{SEARCH_INDEX_FILE}'. Ensure Docker CMD builds it."
        )

    print(f"Loading {NUM_EMBEDDINGS} embeddings from '{RAW_EMBEDDINGS_FILE}'...")
    raw_embeddings = np.fromfile(RAW_EMBEDDINGS_FILE, dtype=np.float32).reshape(
        NUM_EMBEDDINGS, EMBEDDING_DIM
    )
    print("Service is ready.")


@app.get("/search")
def search(item_id: int, num_neighbors: int):
    start_time = time.time()

    # Get the query vector
    query_vector = raw_embeddings[item_id]

    # Format query for C++ tool
    query_str = f"{num_neighbors + 1}," + ",".join(map(str, query_vector))

    try:
        # Execute C++ search tool with num_embeddings parameter
        process = subprocess.Popen(
            [SEARCH_TOOL, "search", SEARCH_INDEX_FILE, str(NUM_EMBEDDINGS)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Send query and get results
        stdout, stderr = process.communicate(input=query_str + "\n")

        if process.returncode != 0:
            raise RuntimeError(f"Search failed: {stderr}")

        # Parse results (format: index,score per line)
        results = []
        for line in StringIO(stdout):
            if line.strip():
                idx, score = map(float, line.strip().split(","))
                if int(idx) != item_id:
                    results.append({"item_id": int(idx), "score": score})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    search_time = time.time() - start_time

    return {
        "query_item_id": item_id,
        "results": results[:num_neighbors],
        "time_ms": search_time * 1000,
    }
