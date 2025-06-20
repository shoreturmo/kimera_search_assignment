# High-Performance Similarity Search API - Take-Home Assignment

Welcome! This repository contains the starter code for the AI/ML Engineer take-home assignment. Your goal is to build a performant, containerized search service that can efficiently find similar items in a large vector dataset.

The assignment is divided into a **Core Task (Python)** and an **Optional Bonus (C++)**. Your solution will be evaluated on its design, performance at scale, and code quality.

### Project Structure

- `README.md`: This file.
- `Dockerfile`: Builds the application container. You will need to modify it for the C++ bonus.
- `requirements.txt`: Python dependencies.
- `QUESTIONS.md`: A file for you to write your answers to the follow-up questions.
- `app/`: Contains the Python FastAPI service.
    - `main.py`: **Main file to modify for the core task.**
    - `generate_data.py`: A script to create the dummy data.
- `search_core/`: **(Optional Bonus)** Contains the C++ search tool.
    - `search.cpp`
    - `Makefile`

---

## Core Task: Python Implementation

Your main goal is to complete the Python service to be performant at scale. A simple brute-force search will not pass the performance requirements for larger datasets.

### Instructions

**1. Implement the Search Logic**
   - Open `app/main.py`.
   - Complete the `TODO` sections in the `startup_event` and `search` functions.
   - In `startup_event`, you should prepare any data structures needed for fast searching and store them in the `search_accelerator` variable.
   - In `search`, use your prepared `search_accelerator` to efficiently find the nearest neighbors.

**2. Build the Docker Image**
   From the root directory of the project, run:
   ```bash
   docker build -t kimera-search .
   ```

**3. Run the Container**
   You can control the dataset size with the `NUM_EMBEDDINGS` environment variable.

   *To run with 10,000 items (for quick development and testing):*
   ```bash
   docker run -d -p 8000:8000 --name kimera-app -e NUM_EMBEDDINGS=10000 kimera-search
   ```

   *To run with 100,000 items (for performance evaluation):*
   ```bash
   docker run -d -p 8000:8000 --name kimera-app -e NUM_EMBEDDINGS=100000 kimera-search
   ```
   You can check the startup logs with `docker logs kimera-app`.

**4. Test the API**
   Once the service is running, test it from another terminal:
   ```bash
   curl "http://127.0.0.1:8000/search?item_id=42&num_neighbors=5"
   ```

**5. Stop and Remove the Container**
   ```bash
   docker stop kimera-app && docker rm kimera-app
   ```

---

## Optional Bonus: C++ Implementation

To showcase your C++ skills, you can replace the Python search logic with the high-performance C++ tool.

### Instructions

**1. Complete the C++ Code**
   - Implement the `build` and `search` modes in `search_core/search.cpp`.
   - Modify `search_core/Makefile` to correctly link against any third-party libraries you use.

**2. Enable the Multi-Stage Docker Build**
   - Open the `Dockerfile`.
   - Uncomment the entire `builder` stage at the top.
   - Uncomment the `COPY --from=builder ...` line in the final `python` stage.

**3. Modify the Docker `CMD` for C++ Orchestration**
   - In the `Dockerfile`, modify the final `CMD` block to execute your C++ `build` command after generating data but before starting `uvicorn`. This will create the search index on disk.

**4. Update `main.py` to use the C++ Tool**
   - The `startup_event` in `main.py` will now be simpler, as its main job is just to load the raw embeddings for query lookup.
   - In the `search` function, replace the Python search logic with a call to Python's `subprocess` module, executing your C++ `search_tool` in `search` mode.

**5. Re-build and Run**
   - Re-build your Docker image using the same `docker build` command.
   - Run the container. The service should now be powered by your C++ core. Test it with the same `curl` command.
