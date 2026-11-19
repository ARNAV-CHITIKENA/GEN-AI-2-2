import torch

from matrix_display import (
    display_title,
    display_matrix,
    display_formula,
    display_multiplication
)

from scaled_dot_product import scaled_dot_product_attention


def run_self_attention(X, generator):

    display_title("SINGLE HEAD SELF ATTENTION")

    d_model = X.shape[1]

    # Generate weight matrices
    W_Q = torch.randn(
        d_model, d_model, generator=generator
    ) * 0.5

    W_K = torch.randn(
        d_model, d_model, generator=generator
    ) * 0.5

    W_V = torch.randn(
        d_model, d_model, generator=generator
    ) * 0.5

    display_matrix("Input Embedding Matrix (X):", X)

    # Query
    display_formula("QUERY FORMULA", "Q = X x W_Q")

    Q = torch.matmul(X, W_Q)

    display_multiplication(
        "QUERY MATRIX CALCULATION",
        "X", X,
        "W_Q", W_Q,
        "Q", Q
    )

    # Key
    display_formula("KEY FORMULA", "K = X x W_K")

    K = torch.matmul(X, W_K)

    display_multiplication(
        "KEY MATRIX CALCULATION",
        "X", X,
        "W_K", W_K,
        "K", K
    )

    # Value
    display_formula("VALUE FORMULA", "V = X x W_V")

    V = torch.matmul(X, W_V)

    display_multiplication(
        "VALUE MATRIX CALCULATION",
        "X", X,
        "W_V", W_V,
        "V", V
    )

    # Scaled dot-product attention
    attention_output = scaled_dot_product_attention(
        Q, K, V, "SINGLE HEAD"
    )

    # Output projection
    W_O = torch.randn(
        d_model, d_model, generator=generator
    ) * 0.5

    display_formula(
        "OUTPUT PROJECTION",
        "Final Output = Attention Output x W_O"
    )

    final_output = torch.matmul(attention_output, W_O)

    display_multiplication(
        "FINAL OUTPUT PROJECTION",
        "Attention Output", attention_output,
        "W_O", W_O,
        "Final Output", final_output
    )

    return final_output