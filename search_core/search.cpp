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

const int EMBEDDING_DIM = 128;

// A struct to hold search results.
struct SearchResult {
    int index;
    float score;
};

// Helper function for L2 normalization
void normalize_vector(std::vector<float>& vec) {
    float norm = 0.0f;
    for (float val : vec) {
        norm += val * val;
    }
    norm = std::sqrt(norm);
    if (norm > 0) {
        for (float& val : vec) {
            val /= norm;
        }
    }
}

// Compute cosine similarity between normalized vectors
float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot_product = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot_product += a[i] * b[i];
    }
    return dot_product;
}

// Save normalized vectors to disk
void save_index(const std::string& path, const std::vector<std::vector<float>>& vectors) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open index file for writing");
    }
    
    for (const auto& vec : vectors) {
        out.write(reinterpret_cast<const char*>(vec.data()), EMBEDDING_DIM * sizeof(float));
    }
}

// Load normalized vectors from disk
std::vector<std::vector<float>> load_index(const std::string& path, int num_embeddings) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open index file for reading");
    }
    
    std::vector<std::vector<float>> vectors(num_embeddings, std::vector<float>(EMBEDDING_DIM));
    
    for (auto& vec : vectors) {
        in.read(reinterpret_cast<char*>(vec.data()), EMBEDDING_DIM * sizeof(float));
        if (!in) {
            throw std::runtime_error("Failed to read vector from index");
        }
    }
    
    return vectors;
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
    
    // Parse k
    if (!std::getline(iss, token, ',')) return {};
    k = std::stoi(token);
    
    // Parse vector values
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

        std::vector<std::vector<float>> all_embeddings = load_embeddings(embeddings_input_path, num_embeddings);
        if (all_embeddings.empty()) {
            std::cerr << "Fatal: Failed to load embeddings." << std::endl;
            return 1;
        }
        std::cerr << "Embeddings loaded successfully." << std::endl;

        std::cerr << "Building search index..." << std::endl;
        for (auto& embedding : all_embeddings) {
            normalize_vector(embedding);
        }

        std::cerr << "Saving index to: " << index_output_path << std::endl;
        try {
            save_index(index_output_path, all_embeddings);
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

        std::vector<std::vector<float>> normalized_vectors;
        try {
            normalized_vectors = load_index(index_file_path, num_embeddings);
        } catch (const std::exception& e) {
            std::cerr << "Error loading index: " << e.what() << std::endl;
            return 1;
        }

        std::cerr << "Index loaded. Ready to receive queries on stdin." << std::endl;

        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;

            int k = 0;
            std::vector<float> query_vector = parse_query_line(line, k);
            if (query_vector.empty()) continue;

            normalize_vector(query_vector);

            std::priority_queue<std::pair<float, int>> top_k;
            
            for (size_t i = 0; i < normalized_vectors.size(); ++i) {
                float similarity = cosine_similarity(normalized_vectors[i], query_vector);
                top_k.push({similarity, static_cast<int>(i)});
            }

            std::vector<SearchResult> results;
            results.reserve(k);
            
            for (int i = 0; i < k && !top_k.empty(); ++i) {
                auto [similarity, idx] = top_k.top();
                top_k.pop();
                
                SearchResult result;
                result.index = idx;
                result.score = similarity;
                results.push_back(result);
            }

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