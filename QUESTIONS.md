### 1. Algorithm Choice & Justification

Describe the search algorithm and/or data structures you chose to implement in your service.

- Why did you select this approach over a simple brute-force search?
- As part of your justification, explain why cosine similarity (or the distance metric you used) is suitable for this type of high-dimensional data.
- What are the key parameters of your chosen method, and how do they affect the trade-off between search speed, memory usage, and accuracy?

### 2. Scaling to Production

Your current service prepares its search structures on container startup.

- What are the limitations of this approach for a catalog of 50 million products with daily updates?
- How would you design a production system to handle this scale? Describe the key components for index storage, a serving layer, and a process for handling updates.

### 3. Production Monitoring

- How would you monitor the health and performance of this search API in production?
- What key metrics (both system-level and business-level) would you track?
