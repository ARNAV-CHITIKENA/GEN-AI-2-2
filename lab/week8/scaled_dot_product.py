import torch
import math

from matrix_display import (
    display_title,
    display_matrix,
    display_formula,
    display_multiplication
)


def scaled_dot_product_attention(Q, K, V, head_name="Attention"):

    display_title(f"{head_name} - SCALED DOT PRODUCT ATTENTION")

    # Step 1: Transpose Key
    K_T = K.transpose(-2, -1)

    display_formula(
        "STEP 1: TRANSPOSE KEY MATRIX",
        "K^T = Transpose(K)"
    )

    display_matrix("Key Matrix (K):", K)
    display_matrix("Transposed Key Matrix (K^T):", K_T)

    # Step 2: Calculate Q * K transpose
    scores = torch.matmul(Q, K_T)

    display_multiplication(
        "STEP 2: ATTENTION SCORE CALCULATION",
        "Q", Q,
        "K^T", K_T,
        "Scores", scores
    )

    # Step 3: Scaling
    d_k = K.shape[-1]
    scaling_factor = math.sqrt(d_k)

    display_formula(
        "STEP 3: SCALING FORMULA",
        "Scaled Scores = (Q x K^T) / sqrt(d_k)"
    )

    print(f"\nKey Dimension (d_k): {d_k}")
    print(f"Square Root of d_k: {scaling_factor:.4f}")

    scaled_scores = scores / scaling_factor

    display_matrix("Scaled Attention Scores:", scaled_scores)

    # Step 4: Softmax
    display_formula(
        "STEP 4: SOFTMAX FORMULA",
        "Attention Weights = Softmax(Scaled Scores)"
    )

    attention_weights = torch.softmax(scaled_scores, dim=-1)

    display_matrix("Attention Weights (Softmax):", attention_weights)

    # Step 5: Multiply attention weights with V
    display_formula(
        "STEP 5: FINAL ATTENTION OUTPUT",
        "Output = Attention Weights x V"
    )

    output = torch.matmul(attention_weights, V)

    display_multiplication(
        "ATTENTION OUTPUT CALCULATION",
        "Attention Weights", attention_weights,
        "V", V,
        "Output", output
    )

    return output