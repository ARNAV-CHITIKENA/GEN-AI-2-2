import torch


def display_title(title):
    print("\n" + "=" * 65)
    print(title.center(65))
    print("=" * 65)


def display_matrix(name, matrix):
    print(f"\n{name}")
    print(f"Shape: {tuple(matrix.shape)}")
    print(matrix)


def display_formula(description, formula):
    print(f"\n{description}")
    print(f"Formula: {formula}")


def display_multiplication(description, a_name, a, b_name, b, result_name, result):
    print(f"\n{description}")
    print(f"Formula: {result_name} = {a_name} x {b_name}")
    print(f"\n{a_name} Matrix:")
    print(a)
    print(f"\n{b_name} Matrix:")
    print(b)
    print(f"\n{result_name} Matrix:")
    print(result)
    print(f"Result Shape: {tuple(result.shape)}")