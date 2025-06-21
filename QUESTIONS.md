### 1. Algorithm Choice & Justification

Describe the search algorithm and/or data structures you chose to implement in your service.

The implementation uses a normalized vector approach with cosine similarity, optimized through numpy's vectorized operations:
- Pre-processing: L2 normalization of all embeddings at startup
- Search: Vector dot product for similarity computation + efficient top-k selection using np.argpartition

- Why did you select this approach over a simple brute-force search?
  While this is still an O(n) approach like brute force, it's highly optimized because:
  1. Pre-computed L2 normalization reduces computation during search
  2. Numpy's vectorized operations are heavily optimized for matrix operations
  3. np.argpartition provides efficient top-k selection without full sorting
  4. The approach maintains 100% accuracy while being significantly faster than naive distance calculations

- As part of your justification, explain why cosine similarity (or the distance metric you used) is suitable for this type of high-dimensional data.
  Cosine similarity is ideal for high-dimensional embeddings because:
  1. It measures the angle between vectors, making it scale-invariant
  2. It works well with sparse high-dimensional data, reducing the impact of the "curse of dimensionality"
  3. For normalized vectors, it can be computed efficiently using just dot product
  4. It's particularly effective for semantic similarity in embedding spaces

Alternative distance metrics that could be used include:

1. Euclidean Distance (L2):
   - Measures straight-line distance between vectors
   - Good for physical or spatial data
   - More sensitive to magnitude differences
   - Can be computationally expensive in high dimensions

2. Manhattan Distance (L1):
   - Measures sum of absolute differences
   - More robust to outliers than Euclidean
   - Often used in urban/grid-like spaces
   - Can be faster to compute than L2

3. Jaccard Similarity:
   - Good for binary/sparse vectors
   - Measures overlap between sets
   - Less sensitive to magnitude
   - Useful for categorical data

4. Hamming Distance:
   - For binary vectors only
   - Counts differing positions
   - Very fast computation
   - Used in hash-based similarity search

5. Inner Product:
   - Similar to cosine but considers magnitude
   - Useful when vector magnitude is meaningful
   - Simpler computation than cosine
   - Can be biased towards longer vectors

For this specific case of similarity search with 128-dimensional embeddings, cosine similarity is the optimal choice for several reasons:

1. Nature of Embeddings:
   - Embeddings typically encode semantic meaning in the vector direction
   - Vector magnitude is less relevant than orientation
   - Embeddings can have different magnitudes without changing their semantic meaning

2. Computational Efficiency:
   - Upfront L2 normalization enables faster calculations during search
   - Vectorized dot product is highly efficient on modern hardware
   - Avoids costly operations like square root (Euclidean) or absolute values (Manhattan)

3. High-dimensionality Performance:
   - Less affected by the "curse of dimensionality"
   - Maintains semantic significance well in 128 dimensions
   - More robust than Euclidean distance in high-dimensional spaces

4. Interpretability:
   - Values are normalized between -1 and 1
   - Easy to interpret: 1 is maximum similarity, -1 is maximum dissimilarity
   - Allows for intuitive similarity thresholds

- What are the key parameters of your chosen method, and how do they affect the trade-off between search speed, memory usage, and accuracy?
  Key parameters include:
  1. Number of neighbors (k): Higher k means more sorting time but doesn't affect the main similarity computation
  2. Vector normalization: Uses additional memory (storing normalized vectors) to improve search speed
  3. Batch size: Our implementation processes one query at a time, trading some potential parallelism for lower memory usage

### 2. Scaling to Production

Your current service prepares its search structures on container startup.

- What are the limitations of this approach for a catalog of 50 million products with daily updates?
  Current limitations:
  1. Memory constraints: Loading 50M vectors (128-dim float32) would require ~25GB RAM just for raw data
  2. Startup time: Loading and normalizing large datasets would significantly increase container startup time
  3. Update handling: Any update requires full service restart, causing downtime
  4. No persistence: Normalized vectors are recalculated on every startup

- How would you design a production system to handle this scale? Describe the key components for index storage, a serving layer, and a process for handling updates.
  Production architecture proposal:
  1. Storage Layer:
     - Distributed vector database (e.g., Milvus, Qdrant) for persistent storage
     - Sharded architecture for handling large datasets
     - Delta updates support for efficient modifications

  2. Serving Layer:
     - Multiple read-only serving instances
     - Load balancer for request distribution
     - In-memory caching for frequently accessed vectors
     - Async update mechanism to prevent downtime

  3. Update Process:
     - Background indexing service for processing updates
     - Rolling updates to serving instances
     - Version control for indices
     - Validation pipeline for data quality

### 3. Production Monitoring

- How would you monitor the health and performance of this search API in production?
  Monitoring approach:
  1. API Metrics:
     - Request latency (p50, p95, p99)
     - Error rates and types
     - Request volume and patterns
     - Cache hit/miss ratios

  2. System Metrics:
     - Memory usage and patterns
     - CPU utilization
     - Network I/O
     - Container health metrics

  3. Infrastructure:
     - Load balancer metrics
     - Database connection pool stats
     - Disk usage and I/O patterns

- What key metrics (both system-level and business-level) would you track?
  Key metrics:
  1. Business Metrics:
     - Search relevancy scores
     - Click-through rates on search results
     - User engagement metrics
     - Business conversion rates

  2. System Performance:
     - Query latency distribution
     - Index update time
     - Resource utilization
     - Search accuracy vs ground truth

  3. SLA Metrics:
     - Service availability
     - Error budget consumption
     - Recovery time objectives (RTO)
     - Data freshness metrics
