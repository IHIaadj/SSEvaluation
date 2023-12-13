import random

def sample_architectures(search_space, num_samples):
    """
    Randomly sample architectures from the search space.

    Parameters:
    search_space (dict): A dictionary defining the search space.
    num_samples (int): Number of samples to generate.

    Returns:
    list: A list of sampled architectures.
    """
    sampled_architectures = []
    for _ in range(num_samples):
        architecture = {param: random.choice(values) for param, values in search_space.items()}
        sampled_architectures.append(architecture)

    return sampled_architectures

def get_layer_type(layer):
    """ Helper function to get the type of the layer as a string. """
    return str(layer.__class__).split(".")[-1].split("'")[0]

import copy

def find_pattern(sequence, max_size):
    def find_largest_repeating_pattern(sequence, max_size):
        n = len(sequence)
        max_pattern_length = min(n // 2, max_size)
        for length in range(max_pattern_length, 0, -1):
            for start in range(0, n - length):
                pattern = sequence[start:start + length]
                for next_start in range(start + length, n - length + 1):
                    if sequence[next_start:next_start + length] == pattern:
                        return pattern
        return None

    most_repeating_block = find_largest_repeating_pattern(sequence, max_size)
    if most_repeating_block:
        rest_blocks = sequence.copy()
        block_length = len(most_repeating_block)
        i = 0
        while i <= len(rest_blocks) - block_length:
            if rest_blocks[i:i + block_length] == most_repeating_block:
                del rest_blocks[i:i + block_length]
            else:
                i += 1
        return most_repeating_block, rest_blocks
    else:
        return None, sequence
