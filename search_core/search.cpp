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

const int EMBEDDING_DIM = 128;
const int BATCH_SIZE = 1000;
const int TREE_LEAF_SIZE = 32;

// A struct to hold search results
struct SearchResult {
    int index;
    float score;
};

// K-d tree node structure
struct KDNode {
    std::vector<float> pivot;
    int split_dim;
    std::unique_ptr<KDNode> left;
    std::unique_ptr<KDNode> right;
    std::vector<int> point_indices;
    bool is_leaf;

    KDNode() : split_dim(0), is_leaf(false) {}
};

// Global k-d tree root
std::unique_ptr<KDNode> tree_root;
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

// Build k-d tree recursively
std::unique_ptr<KDNode> build_kdtree(const std::vector<int>& indices, int depth) {
    auto node = std::make_unique<KDNode>();
    
    if (indices.size() <= TREE_LEAF_SIZE) {
        node->is_leaf = true;
        node->point_indices = indices;
        return node;
    }
    
    // Choose splitting dimension (cycling through dimensions)
    node->split_dim = depth % EMBEDDING_DIM;
    
    // Find median value in the splitting dimension
    std::vector<float> dim_values;
    for (int idx : indices) {
        dim_values.push_back(global_vectors[idx][node->split_dim]);
    }
    size_t median_idx = dim_values.size() / 2;
    std::nth_element(dim_values.begin(), dim_values.begin() + median_idx, dim_values.end());
    float median_value = dim_values[median_idx];
    
    // Split points
    std::vector<int> left_indices, right_indices;
    for (int idx : indices) {
        if (global_vectors[idx][node->split_dim] < median_value) {
            left_indices.push_back(idx);
        } else {
            right_indices.push_back(idx);
        }
    }
    
    // Build subtrees
    if (!left_indices.empty()) {
        node->left = build_kdtree(left_indices, depth + 1);
    }
    if (!right_indices.empty()) {
        node->right = build_kdtree(right_indices, depth + 1);
    }
    
    return node;
}

// Search k-d tree for nearest neighbors
void search_kdtree(const KDNode* node, const std::vector<float>& query,
                  std::priority_queue<std::pair<float, int>>& top_k, int k,
                  float& worst_score) {
    if (!node) return;
    
    if (node->is_leaf) {
        for (int idx : node->point_indices) {
            float similarity = cosine_similarity(query, global_vectors[idx]);
            if (similarity > worst_score) {
                top_k.push({similarity, idx});
                if (top_k.size() > k) {
                    top_k.pop();
                    worst_score = top_k.top().first;
                }
            }
        }
        return;
    }
    
    float query_val = query[node->split_dim];
    float pivot_val = node->pivot[node->split_dim];
    
    if (query_val < pivot_val) {
        search_kdtree(node->left.get(), query, top_k, k, worst_score);
        if (std::abs(query_val - pivot_val) * std::abs(query_val - pivot_val) > worst_score) {
            search_kdtree(node->right.get(), query, top_k, k, worst_score);
        }
    } else {
        search_kdtree(node->right.get(), query, top_k, k, worst_score);
        if (std::abs(query_val - pivot_val) * std::abs(query_val - pivot_val) > worst_score) {
            search_kdtree(node->left.get(), query, top_k, k, worst_score);
        }
    }
}

std::vector<SearchResult> find_nearest_neighbors(const std::vector<float>& query, int k) {
    std::priority_queue<std::pair<float, int>> top_k;
    float worst_score = -1.0f;
    
    search_kdtree(tree_root.get(), query, top_k, k, worst_score);
    
    std::vector<SearchResult> results;
    while (!top_k.empty()) {
        auto [similarity, idx] = top_k.top();
        top_k.pop();
        SearchResult result;
        result.index = idx;
        result.score = similarity;
        results.push_back(result);
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

        std::cerr << "Building search index..." << std::endl;

        // Normalize vectors
        for (auto& embedding : global_vectors) {
            normalize_vector(embedding);
        }

        // Build k-d tree
        std::vector<int> all_indices(num_embeddings);
        std::iota(all_indices.begin(), all_indices.end(), 0);
        tree_root = build_kdtree(all_indices, 0);

        std::cerr << "Saving index to: " << index_output_path << std::endl;
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

        // Rebuild k-d tree
        std::vector<int> all_indices(num_embeddings);
        std::iota(all_indices.begin(), all_indices.end(), 0);
        tree_root = build_kdtree(all_indices, 0);

        std::cerr << "Index loaded. Ready to receive queries on stdin." << std::endl;

        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;

            int k = 0;
            std::vector<float> query_vector = parse_query_line(line, k);
            if (query_vector.empty()) continue;

            normalize_vector(query_vector);
            auto results = find_nearest_neighbors(query_vector, k);

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