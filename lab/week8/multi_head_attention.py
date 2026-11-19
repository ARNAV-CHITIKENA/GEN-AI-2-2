import torch

from matrix_display import (
    display_title,
    display_matrix,
    display_formula,
    display_multiplication
)

from scaled_dot_product import scaled_dot_product_attention


def run_multi_head_attention(X, num_heads, generator):

    display_title(f"MULTI-HEAD ATTENTION - {num_heads} HEADS")

    d_model = X.shape[1]
    head_dim = d_model // num_heads

    display_matrix("Input Embedding Matrix (X):", X)

    print(f"\nNumber of Heads: {num_heads}")
    print(f"Model Dimension: {d_model}")
    print(f"Dimension Per Head: {head_dim}")

    display_formula(
        "HEAD DIMENSION FORMULA",
        "d_k = d_model / number_of_heads"
    )

    head_outputs = []

    # Process every attention head
    for head in range(num_heads):

        display_title(f"HEAD {head + 1} CALCULATIONS")

        # Independent projection weights for each head
        W_Q = torch.randn(
            d_model, head_dim, generator=generator
        ) * 0.5

        W_K = torch.randn(
            d_model, head_dim, generator=generator
        ) * 0.5

        W_V = torch.randn(
            d_model, head_dim, generator=generator
        ) * 0.5

        # Query
        Q = torch.matmul(X, W_Q)

        display_formula(
            f"HEAD {head + 1} QUERY",
            "Q = X x W_Q"
        )

        display_multiplication(
            "QUERY CALCULATION",
            "X", X,
            "W_Q", W_Q,
            "Q", Q
        )

        # Key
        K = torch.matmul(X, W_K)

        display_formula(
            f"HEAD {head + 1} KEY",
            "K = X x W_K"
        )

        display_multiplication(
            "KEY CALCULATION",
            "X", X,
            "W_K", W_K,
            "K", K
        )

        # Value
        V = torch.matmul(X, W_V)

        display_formula(
            f"HEAD {head + 1} VALUE",
            "V = X x W_V"
        )

        display_multiplication(
            "VALUE CALCULATION",
            "X", X,
            "W_V", W_V,
            "V", V
        )

        # Attention calculation for this head
        output = scaled_dot_product_attention(
            Q, K, V, f"HEAD {head + 1}"
        )

        head_outputs.append(output)

    # Concatenate all heads
    display_title("CONCATENATION OF ALL HEADS")

    display_formula(
        "CONCATENATION FORMULA",
        "Concat = Concat(Head1, Head2, ..., HeadN)"
    )

    concatenated = torch.cat(head_outputs, dim=-1)

    for i, head_output in enumerate(head_outputs):
        display_matrix(
            f"Head {i + 1} Output:",
            head_output
        )

    display_matrix(
        "Concatenated Attention Matrix:",
        concatenated
    )

    # Final output projection
    W_O = torch.randn(
        d_model, d_model, generator=generator
    ) * 0.5

    display_title("FINAL OUTPUT PROJECTION")

    display_formula(
        "MULTI-HEAD ATTENTION FORMULA",
        "MultiHead = Concat(Head1, ..., HeadN) x W_O"
    )

    final_output = torch.matmul(concatenated, W_O)

    display_multiplication(
        "FINAL MULTI-HEAD OUTPUT",
        "Concatenated", concatenated,
        "W_O", W_O,
        "Final Output", final_output
    )

    return final_output