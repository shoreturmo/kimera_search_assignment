# =================================================================
#  Build Stage (Only for optional C++ bonus)
# =================================================================

FROM gcc:11 AS builder

# Install C++ build tools
RUN apt-get update && apt-get install -y \
    make \
    libopenblas-dev

WORKDIR /build

# Copy C++ source and Makefile
COPY ./search_core/ /build/

RUN make

# =================================================================
#  Final Stage (Python Application)
# =================================================================
FROM python:3.9-slim

# Install runtime dependencies for C++ solutions
RUN apt-get update && apt-get install -y \
    libopenblas0

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python application code
COPY ./app /app

# Uncomment the line below if you are completing the C++ bonus
COPY --from=builder /build/search_tool /app/search_core/

# This CMD orchestrates the entire setup process inside the container
CMD sh -c "\
    echo '--- Step 1: Creating data directory ---' && \
    mkdir -p /app/data && \
    \
    echo '--- Step 2: Generating raw embedding data ---' && \
    python generate_data.py \
        --num-embeddings ${NUM_EMBEDDINGS:-10000} \
        --output /app/data/embeddings.bin && \
    \
    echo '--- Step 3: Building search index ---' && \
    /app/search_core/search_tool build \
        /app/data/embeddings.bin \
        /app/data/search.index \
        ${NUM_EMBEDDINGS:-10000} && \
    \
    echo '--- Step 4: Starting API server ---' && \
    uvicorn main:app --host 0.0.0.0 --port 8000 \
    "
