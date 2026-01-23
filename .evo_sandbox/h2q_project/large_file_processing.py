import gc

def process_large_file(filename):
    """Processes a large file line by line using a generator."""
    def line_generator(filename):
        with open(filename, 'r') as f:
            for line in f:
                yield line

    for line in line_generator(filename):
        # Process each line (example: count the number of words)
        words = line.split()
        num_words = len(words)
        print(f"Line: {line.strip()}, Word Count: {num_words}")
        del words, num_words #Explicitly delete to release memory after the loop

    # Explicitly trigger garbage collection
    gc.collect()

# Example Usage (Illustrative)
if __name__ == '__main__':
    # Create a dummy large file for demonstration
    filename = 'large_file.txt'
    with open(filename, 'w') as f:
        for i in range(1000): #Increased range to showcase memory usage
            f.write(f'This is line {i} with some random words. \n')

    process_large_file(filename)
