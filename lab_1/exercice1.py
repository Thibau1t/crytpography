######################## VILLEPREUX Thibault - lab 1 : march 2024 ########################

# Imports & Constants
from itertools import product

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
            print(f"\t\tFunction {i}: \033[92m(balanced)\033[0m ({eq_0} == {eq_1})")
        else:
            print(f"\t\tFunction {i}: \033[91m(not balanced)\033[0m ({eq_0} != {eq_1})")

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
        print(f"\t\033[92m{len(vectors)} affine functions generated \033[0m")

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
    min_nonlinearity = float('inf')
    i = 0
    
    for sbox_function in sbox_functions:
        min_distance = float('inf')
        i += 1
        
        for affine_vector in affine_vectors:
            distance = hamming_distance(sbox_function, affine_vector)
            min_distance = min(min_distance, distance)
        
        print(f"\t\033[92mNonlinearity of the S-box {i}: {min_distance}\033[0m")
            
        min_nonlinearity = min(min_nonlinearity, min_distance)
    
    print(f"\t\033[92mNonlinearity of the S-box: {min_nonlinearity}\033[0m")
    
    return min_nonlinearity

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
        print(f"\tSAC probability for \033[92mfunction {function+1}: {sac_probabilities[function]}\033[0m") 

    

    return sac_probabilities

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

    print ("\n\033[33mAll questions done :)\033[0m")

if __name__ == "__main__":
    main()
