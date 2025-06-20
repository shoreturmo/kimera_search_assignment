import numpy as np
import argparse

def generate_and_save_data(num_embeddings, dim, output_file):
    """
    Generates a dummy dataset of normalized embeddings and saves them to a
    simple binary file (float32).
    """
    print(f"Generating {num_embeddings} embeddings of dimension {dim}...")
    rng = np.random.default_rng(seed=42)
    embeddings = rng.random((num_embeddings, dim), dtype=np.float32)

    with open(output_file, 'wb') as f:
        f.write(embeddings.tobytes())
    
    print(f"Saved {num_embeddings} raw embeddings to '{output_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dummy embedding data.")
    parser.add_argument("--num-embeddings", type=int, required=True, help="Number of embeddings to generate.")
    parser.add_argument("--dim", type=int, default=128, help="Dimension of embeddings.")
    parser.add_argument("--output", type=str, required=True, help="Output file path for raw embeddings.")
    args = parser.parse_args()

    generate_and_save_data(args.num_embeddings, args.dim, args.output)
