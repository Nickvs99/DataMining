"""
Generate a subset of the large training and test set.
"""

def main():

    large_files = ["test_set.csv", "training_set.csv"]
    N_LINES = 100000
    SUFFIX = f"_{N_LINES}"
    
    for filename in large_files:

        # Get first n lines
        with open(filename) as f:
            head = [next(f) for _ in range(N_LINES)]

        # Write first n lines
        new_filename =  "".join(filename.split(".")[:-1]) + SUFFIX + ".csv"
        lines = "".join(head)

        with open(new_filename, 'w') as f:
            f.write(lines)
                

if __name__ == "__main__":
    main()