#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <memory> 

const int EMBEDDING_DIM = 128;

// A struct to hold search results.
struct SearchResult {
    int index;
    float score;
};

// --- Function Prototypes ---
void print_usage(const char* program_name);
std::vector<std::vector<float>> load_embeddings(const std::string& file_path, int num_embeddings);
std::vector<float> parse_query_line(const std::string& line, int& k);

/**
 * The main entry point for the search tool.
 * This tool operates in two modes: 'build' and 'search'.
 *
 * Mode 'build':
 *   - Reads raw embedding vectors from a file.
 *   - Constructs an efficient search index
 *   - Saves the built index to disk.
 *   - Usage: ./search_tool build <path_to_raw_embeddings> <path_for_output_index> <num_embeddings>
 *
 * Mode 'search':
 *   - Loads a pre-built index from disk.
 *   - Listens for queries on stdin and performs fast searches using the index.
 *   - Usage: ./search_tool search <path_to_built_index>
 */
int main(int argc, char* argv[]) {
    // Disable synchronization with C-style I/O for faster performance.
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

        std::cout << "Mode: Build" << std::endl;
        std::cout << "Loading embeddings from: " << embeddings_input_path << std::endl;

        // 1. Load the raw embeddings from the data file.
        std::vector<std::vector<float>> all_embeddings = load_embeddings(embeddings_input_path, num_embeddings);
        if (all_embeddings.empty()) {
            std::cerr << "Fatal: Failed to load embeddings." << std::endl;
            return 1;
        }
        std::cout << "Embeddings loaded successfully." << std::endl;

        // 2. TODO: Build your efficient search index here.
        std::cout << "Building search index..." << std::endl;

        // --- YOUR INDEX BUILDING LOGIC GOES HERE ---


        // -------------------------------------------

        // 3. TODO: Save the constructed index to the output file.
        std::cout << "Saving index to: " << index_output_path << std::endl;

        // --- YOUR INDEX SAVING LOGIC GOES HERE ---


        // -----------------------------------------

        std::cout << "Index built and saved successfully." << std::endl;

    } else if (mode == "search") {
        if (argc != 3) {
            print_usage(argv[0]);
            return 1;
        }
        std::string index_file_path = argv[2];

        std::cout << "Mode: Search" << std::endl;
        std::cout << "Loading pre-built index from: " << index_file_path << std::endl;
        
        // --- YOUR INDEX LOADING LOGIC GOES HERE ---


        // ------------------------------------------

        std::cout << "Index loaded. Ready to receive queries on stdin." << std::endl;

        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;

            int k = 0;
            std::vector<float> query_vector = parse_query_line(line, k);
            if (query_vector.empty()) continue;

            // 2. TODO: Use the loaded index to perform a fast search.
            //    The result will typically be two arrays: one for distances/scores
            //    and one for the indices of the neighbors.
            std::vector<SearchResult> results;

            // --- YOUR INDEX SEARCH LOGIC GOES HERE ---


            // -----------------------------------------

            // 3. Print the results to standard output.
            for (const auto& result : results) {
                std::cout << result.index << "," << result.score << "\n";
            }
        }
    } else {
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}

// --- Helper Function Implementations ---

void print_usage(const char* program_name) {
    std::cerr << "Usage: \n"
              << "  " << program_name << " build <embeddings_file> <output_index_file> <num_embeddings>\n"
              << "  " << program_name << " search <index_file>\n"
              << std::endl;
}

std::vector<std::vector<float>> load_embeddings(const std::string& file_path, int num_embeddings) {
    std::ifstream input_file(file_path, std::ios::binary);
    if (!input_file) {
        std::cerr << "Error opening file: " << file_path << std::endl;
        return {};
    }

    // TODO: Implement the logic to read the binary file into a 2D vector.
    // This part is crucial for the 'build' step.
    
    return {}; // Placeholder
}

std::vector<float> parse_query_line(const std::string& line, int& k) {
    // The line format is "k,v1,v2,v3,..."
    // TODO: Implement the parsing logic.
    // - Find the first comma to separate 'k'.
    // - Parse 'k' and the rest of the string into a vector of floats.
    
    return {}; // Placeholder
}