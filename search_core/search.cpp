#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <memory>
#include <algorithm>
#include <cmath>
#include <queue>
#include <immintrin.h> // For SIMD
#include <thread>
#include <mutex>
#include <random>
#include <unordered_map>

const int EMBEDDING_DIM = 128;
const int BATCH_SIZE = 1000;
const int TREE_LEAF_SIZE = 32;

// A struct to hold search results
struct SearchResult {
    int index;
    float score;
};

// LSH parameters
const int LSH_NUM_TABLES = 10;
const int LSH_KEY_SIZE = 16;

// LSH hash tables
typedef std::vector<int> HashKey;
std::vector<std::unordered_map<std::string, std::vector<int>>> lsh_tables;
std::vector<std::vector<float>> lsh_random_vectors;
std::vector<std::vector<float>> global_vectors;

// Helper function for L2 normalization using SIMD
void normalize_vector(std::vector<float>& vec) {
    float norm = 0.0f;
    int i = 0;
    
    // Use AVX for parallel sum of squares
    __m256 sum = _mm256_setzero_ps();
    for (; i <= EMBEDDING_DIM - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(&vec[i]);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(v, v));
    }
    
    // Horizontal sum of AVX register
    float temp[8];
    _mm256_storeu_ps(temp, sum);
    for (int j = 0; j < 8; j++) {
        norm += temp[j];
    }
    
    // Handle remaining elements
    for (; i < EMBEDDING_DIM; i++) {
        norm += vec[i] * vec[i];
    }
    
    norm = std::sqrt(norm);
    if (norm > 0) {
        // Vectorized normalization
        __m256 norm_vec = _mm256_set1_ps(norm);
        for (i = 0; i <= EMBEDDING_DIM - 8; i += 8) {
            __m256 v = _mm256_loadu_ps(&vec[i]);
            v = _mm256_div_ps(v, norm_vec);
            _mm256_storeu_ps(&vec[i], v);
        }
        // Handle remaining elements
        for (; i < EMBEDDING_DIM; i++) {
            vec[i] /= norm;
        }
    }
}

// Optimized cosine similarity using SIMD
float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot_product = 0.0f;
    int i = 0;
    
    __m256 sum = _mm256_setzero_ps();
    for (; i <= EMBEDDING_DIM - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
    }
    
    float temp[8];
    _mm256_storeu_ps(temp, sum);
    for (int j = 0; j < 8; j++) {
        dot_product += temp[j];
    }
    
    for (; i < EMBEDDING_DIM; i++) {
        dot_product += a[i] * b[i];
    }
    
    return dot_product;
}

// Helper: generate random hyperplanes for LSH
void generate_lsh_planes(int num_tables, int key_size, int dim) {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    lsh_random_vectors.clear();
    for (int t = 0; t < num_tables * key_size; ++t) {
        std::vector<float> plane(dim);
        for (int d = 0; d < dim; ++d) {
            plane[d] = dist(gen);
        }
        lsh_random_vectors.push_back(plane);
    }
}

// Helper: compute LSH key for a vector
std::string compute_lsh_key(const std::vector<float>& vec, int table_idx) {
    std::string key;
    int offset = table_idx * LSH_KEY_SIZE;
    for (int i = 0; i < LSH_KEY_SIZE; ++i) {
        float dot = 0.0f;
        for (int d = 0; d < EMBEDDING_DIM; ++d) {
            dot += vec[d] * lsh_random_vectors[offset + i][d];
        }
        key += (dot > 0 ? '1' : '0');
    }
    return key;
}

// Build LSH index
template<typename T>
void build_lsh_index(const std::vector<T>& vectors) {
    lsh_tables.clear();
    lsh_tables.resize(LSH_NUM_TABLES);
    for (int t = 0; t < LSH_NUM_TABLES; ++t) {
        for (int i = 0; i < (int)vectors.size(); ++i) {
            std::string key = compute_lsh_key(vectors[i], t);
            lsh_tables[t][key].push_back(i);
        }
    }
}

// ANN search using LSH
std::vector<SearchResult> ann_search(const std::vector<float>& query, int k) {
    std::unordered_map<int, bool> candidate_set;
    for (int t = 0; t < LSH_NUM_TABLES; ++t) {
        std::string key = compute_lsh_key(query, t);
        auto it = lsh_tables[t].find(key);
        if (it != lsh_tables[t].end()) {
            for (int idx : it->second) {
                candidate_set[idx] = true;
            }
        }
    }
    // Fallback: if not enough candidates, use all
    std::vector<int> candidates;
    for (const auto& kv : candidate_set) candidates.push_back(kv.first);
    if ((int)candidates.size() < k) {
        for (int i = 0; i < (int)global_vectors.size(); ++i) {
            if (!candidate_set.count(i)) candidates.push_back(i);
            if ((int)candidates.size() >= k*2) break;
        }
    }
    // Score candidates
    std::priority_queue<std::pair<float, int>> top_k;
    for (int idx : candidates) {
        float sim = cosine_similarity(query, global_vectors[idx]);
        top_k.push({sim, idx});
        if ((int)top_k.size() > k) top_k.pop();
    }
    std::vector<SearchResult> results;
    while (!top_k.empty()) {
        auto [sim, idx] = top_k.top();
        top_k.pop();
        results.push_back({idx, sim});
    }
    std::reverse(results.begin(), results.end());
    return results;
}

void save_index(const std::string& path, const std::vector<std::vector<float>>& vectors) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open index file for writing");
    }
    
    for (const auto& vec : vectors) {
        out.write(reinterpret_cast<const char*>(vec.data()), EMBEDDING_DIM * sizeof(float));
    }
}

std::vector<std::vector<float>> load_embeddings(const std::string& file_path, int num_embeddings) {
    std::ifstream input_file(file_path, std::ios::binary);
    if (!input_file) {
        std::cerr << "Error opening file: " << file_path << std::endl;
        return {};
    }

    std::vector<std::vector<float>> embeddings(num_embeddings, std::vector<float>(EMBEDDING_DIM));
    
    for (int i = 0; i < num_embeddings; ++i) {
        input_file.read(reinterpret_cast<char*>(embeddings[i].data()), 
                       EMBEDDING_DIM * sizeof(float));
        if (!input_file) {
            std::cerr << "Error reading embedding " << i << std::endl;
            return {};
        }
    }
    
    return embeddings;
}

std::vector<float> parse_query_line(const std::string& line, int& k) {
    std::istringstream iss(line);
    std::string token;
    
    if (!std::getline(iss, token, ',')) return {};
    k = std::stoi(token);
    
    std::vector<float> query_vector(EMBEDDING_DIM);
    int i = 0;
    while (std::getline(iss, token, ',') && i < EMBEDDING_DIM) {
        query_vector[i++] = std::stof(token);
    }
    
    return (i == EMBEDDING_DIM) ? query_vector : std::vector<float>();
}

void print_usage(const char* program_name) {
    std::cerr << "Usage: \n"
              << "  " << program_name << " build <embeddings_file> <output_index_file> <num_embeddings>\n"
              << "  " << program_name << " search <index_file> <num_embeddings>\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "build") {
        if (argc != 5) {
            print_usage(argv[0]);
            return 1;
        }
        std::string embeddings_input_path = argv[2];
        std::string index_output_path = argv[3];
        int num_embeddings = std::stoi(argv[4]);

        std::cerr << "Mode: Build" << std::endl;
        std::cerr << "Loading embeddings from: " << embeddings_input_path << std::endl;

        global_vectors = load_embeddings(embeddings_input_path, num_embeddings);
        if (global_vectors.empty()) {
            std::cerr << "Fatal: Failed to load embeddings." << std::endl;
            return 1;
        }
        std::cerr << "Embeddings loaded successfully." << std::endl;

        std::cerr << "Building ANN (LSH) index..." << std::endl;
        for (auto& embedding : global_vectors) {
            normalize_vector(embedding);
        }
        generate_lsh_planes(LSH_NUM_TABLES, LSH_KEY_SIZE, EMBEDDING_DIM);
        build_lsh_index(global_vectors);
        std::cerr << "Index built in memory. Saving index to: " << index_output_path << std::endl;
        // Guardar solo los embeddings, como antes
        try {
            save_index(index_output_path, global_vectors);
        } catch (const std::exception& e) {
            std::cerr << "Error saving index: " << e.what() << std::endl;
            return 1;
        }
        std::cerr << "Index built and saved successfully." << std::endl;

    } else if (mode == "search") {
        if (argc != 4) {
            print_usage(argv[0]);
            return 1;
        }
        std::string index_file_path = argv[2];
        int num_embeddings = std::stoi(argv[3]);

        std::cerr << "Mode: Search" << std::endl;
        std::cerr << "Loading pre-built index from: " << index_file_path << std::endl;

        global_vectors = load_embeddings(index_file_path, num_embeddings);
        if (global_vectors.empty()) {
            std::cerr << "Fatal: Failed to load index." << std::endl;
            return 1;
        }
        for (auto& embedding : global_vectors) {
            normalize_vector(embedding);
        }
        generate_lsh_planes(LSH_NUM_TABLES, LSH_KEY_SIZE, EMBEDDING_DIM);
        build_lsh_index(global_vectors);
        std::cerr << "Index loaded. Ready to receive queries on stdin." << std::endl;

        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            int k = 0;
            std::vector<float> query_vector = parse_query_line(line, k);
            if (query_vector.empty()) continue;
            normalize_vector(query_vector);
            auto results = ann_search(query_vector, k);
            for (const auto& result : results) {
                std::cout << result.index << "," << result.score << "\n";
            }
            std::cout.flush();
        }
    } else {
        print_usage(argv[0]);
        return 1;
    }
    return 0;
}