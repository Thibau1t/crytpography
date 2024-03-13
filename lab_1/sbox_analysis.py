######################## VILLEPREUX Thibault - lab 1 : march 2024 ########################

# Imports & Constants
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import time

input_file = "sbox.SBX"
output_file = "sbox_no_zeros.SBX"


# quesiton 1
def remove_trailing_zeros(input_file, output_file):
    with open(input_file, "rb") as infile, open(output_file, "wb") as outfile:
        content = infile.read()
        outfile.write(content[::2])

# question 2
def count_bits_at_position(byte_list, position):
    count_0 = 0
    count_1 = 0
    for byte in byte_list:
        binary_representation = bin(byte)[2:].zfill(8)
        if binary_representation[position] == '0':
            count_0 += 1
        else:
            count_1 += 1
    return count_0, count_1

def test_balanced(byte_list):
    print("\t\033[33mTesting balance of each function:\033[0m")
    for i in range(8):
        eq_0, eq_1 = count_bits_at_position(byte_list, i)
        if eq_0 == eq_1:
            print(f"\t\tFunction {i+1}: \033[92m(balanced)\033[0m ({eq_0} == {eq_1})")
        else:
            print(f"\t\tFunction {i+1}: \033[91m(not balanced)\033[0m ({eq_0} != {eq_1})")

# question 3
def generate_affine_functions(num_args=8):
    vectors = []

    # generate all possible combinations of 0 and 1
    for combination in product([0, 1], repeat=num_args+1):
        vectors.append(list(combination))

    # verify the number of generated functions
    if (len(vectors) != 2 ** (num_args+1)):
        print("\033[91mError in the generation of affine functions (exit) \033[0m")
        exit(1)
    else :
        print(f"\t\033[92m{len(vectors)}\033[0m affine functions generated ")

    return vectors

def generate_truth_table_sbox_functions(byte_list):
    functions = [[] for _ in range(8)]  # create a list of 8 empty lists

    for byte in byte_list:
        binary_representation = bin(byte)[2:].zfill(8)  # Convert the byte to binary and fill with zeros
        for i in range(8):
            functions[i].append(int(binary_representation[i]))  # Add the bit to the corresponding list

    return functions

def generate_truth_table_affine_functions(affine_vectors):
    truth_tables = []

    for vector in affine_vectors:
        truth_table = []
        for i in range(256):  # There are 256 possible inputs
            inputs = [(i >> j) & 1 for j in range(8)]
            output = vector[0]  # constante a0
            for input_bit, affine_bit in zip(inputs, vector[1:]):
                output ^= input_bit & affine_bit
            truth_table.append(output)
        truth_tables.append(truth_table)

    return truth_tables

def hamming_distance(vector1, vector2):
    return sum(el1 != el2 for el1, el2 in zip(vector1, vector2))

def calculate_nonlinearity(sbox_functions, affine_vectors):
    nonlinearity_vectors = [float('inf')] * 8
    i = 0

    for sbox_function in sbox_functions:
        min_distance = float('inf')
        i += 1

        for affine_vector in affine_vectors:
            distance = hamming_distance(sbox_function, affine_vector)
            min_distance = min(min_distance, distance)

        print(f"\tNonlinearity of the S-box {i}: \033[92m{min_distance}\033[0m")
        nonlinearity_vectors[i-1] = min_distance

    return nonlinearity_vectors

# question 4
def generate_bit_difference_pairs():
    bit_difference_pairs = {}

    for i in range(256):
        for j in range(256):
            bit_difference = bin(i ^ j)[2:].zfill(8)
            if bit_difference.count('1') == 1:
                key = bin(i)[2:].zfill(8)
                value = bin(j)[2:].zfill(8)
                if key not in bit_difference_pairs:
                    bit_difference_pairs[key] = [value]
                else:
                    bit_difference_pairs[key].append(value)

    return bit_difference_pairs

def calculate_sac_probability(pairs, byte_list):
    sac_probabilities = [0] * 8

    for function in range(8):
        num_changes = 0
        num_pairs = 2048

        for key, values in pairs.items():
            byte_index = int(key, 2)

            for value in values:
                if ((byte_list[byte_index] >> (7 - function)) & 1) != ((byte_list[int(value, 2)] >> (7 - function)) & 1):
                    num_changes += 1

        sac_probabilities[function] = num_changes / num_pairs
        print(f"\tSAC probability for function {function+1}: \033[92m{sac_probabilities[function]}\033[0m") 

    print(f"\tSAC probability of the entire S-box : \033[92m{sum(sac_probabilities)/8}\033[0m")

    return sac_probabilities

# question 5 : XOR profile 

def generate_input_pairs():
    pairs = []
    for i in range(2**8):
        binary_i = format(i, '08b')
        for j in range(0, 2**8):
            binary_j = format(j, '08b')
            pairs.append((binary_i, binary_j))

    return pairs

def generate_matrix_sbox(byte_list):
    matrix = [[0] * 16 for _ in range(16)]
    for i in range(16):
        for j in range(16):
            index = i * 16 + j
            matrix[i][j] = format(byte_list[index], '08b')
    return matrix

def get_value_from_matrix(row_binary, col_binary, matrix):
    row_decimal = int(row_binary, 2)
    col_decimal = int(col_binary, 2)
    return matrix[row_decimal][col_decimal]

def calculate_xor_profile(byte_list):
    pairs = generate_input_pairs()
    matrix = generate_matrix_sbox(byte_list)
    xor_profile = [[0] * 256 for _ in range(256)]

    for x1, x2 in pairs:
        row_binary_x1 = x1[:4]  # Extracting the row and column bits correctly
        col_binary_x1 = x1[4:]
        output_x1 = get_value_from_matrix(row_binary_x1, col_binary_x1, matrix)

        row_binary_x2 = x2[:4]
        col_binary_x2 = x2[4:]
        output_x2 = get_value_from_matrix(row_binary_x2, col_binary_x2, matrix)

        xor_input = int(x1, 2) ^ int(x2, 2)
        xor_output = int(output_x1, 2) ^ int(output_x2, 2)

        xor_profile[xor_input][xor_output] += 1  # Incrementing the correct index

    return xor_profile

def sum_2d_array(array):
    total_sum = 0
    for row in array:
        total_sum += sum(row)
    return total_sum


def plot_xor_profile(xor_profile):
    xor_profile[0][0] = 0
    xor_array = np.array(xor_profile)
    print(f"Max : {np.max(xor_array)}")
    plt.imshow(xor_array, cmap='hot', interpolation='nearest', origin='lower')
    plt.colorbar()
    plt.xlabel('Output XOR')
    plt.ylabel('Input XOR')
    plt.title('XOR Profile')
    plt.show()

# question 6 : Cycle length

def calculate_cycle_length(byte_list):
    cycle_length = [0] * 256
    for i in range(256):
        current = i
        count = 0
        while True:
            current = byte_list[current]
            count += 1
            if current == i:
                break
        cycle_length[i] = count
    return cycle_length


def main():
    # Remove trailing zeros : question 1
    print(f"\n\033[33mQuestion 1 : \033[0m", end="")
    remove_trailing_zeros(input_file, output_file)
    print(f"\033[92mDone\033[0m")

    # Test balance of each function : question 2
    print(f"\n\033[33mQuestion 2 : \033[0m")
    with open(output_file, "rb") as file:
        byte_list = list(file.read())
    test_balanced(byte_list)

    # Calculate nonlinearity : question 3
    print(f"\n\033[33mQuestion 3 : \033[0m")
    affine_funcitons = generate_affine_functions()
    sbox_truth_table_functions = generate_truth_table_sbox_functions(byte_list)
    affine_truth_table_functions = generate_truth_table_affine_functions(affine_funcitons)
    calculate_nonlinearity(sbox_truth_table_functions, affine_truth_table_functions)

    # Verification SAC is satisfied for each function : question 4
    print(f"\n\033[33mQuestion 4 : \033[0m")
    pairs = generate_bit_difference_pairs()
    calculate_sac_probability(pairs, byte_list)


    # Calculate XOR profile : question 5
    print(f"\n\033[33mQuestion 5 : \033[0m")
    xor_profile = calculate_xor_profile(byte_list)
    plot_xor_profile(xor_profile)
    print("\t\033[92mXOR profile calculated\033[0m")

    # Calculate Cycle length : question 6
    print(f"\n\033[33mQuestion 6 : \033[0m")
    cycle_length = calculate_cycle_length(byte_list)
    print(f"\tCycle length of the S-box : \033[92m{cycle_length}\033[0m")

    print ("\n\033[33mAll questions done :)\033[0m")


if __name__ == "__main__":
    main()
