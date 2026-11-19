import torch

from matrix_display import display_title, display_matrix
from self_attention import run_self_attention
from multi_head_attention import run_multi_head_attention


def main():

    torch.set_printoptions(
        precision=4,
        sci_mode=False,
        linewidth=120
    )

    display_title("WEEK 8: ATTENTION MECHANISM")

    print("\nIMPLEMENTATION OF SCALED DOT-PRODUCT")
    print("AND MULTI-HEAD ATTENTION USING PYTORCH")

    print("\nSelect Attention Type:")
    print("1. Self-Attention (1 Head)")
    print("2. Multi-Head Attention (2 Heads)")
    print("3. Multi-Head Attention (3 Heads)")

    try:
        choice = int(input("\nEnter your choice (1/2/3): "))

        if choice not in [1, 2, 3]:
            print("\nInvalid choice! Please select 1, 2 or 3.")
            return

        # Fixed seed for reproducible results
        generator = torch.Generator().manual_seed(42)

        # Input embedding matrix
        X = torch.tensor([
            [1., 0., 1., 0., 1., 0.],
            [0., 1., 0., 1., 0., 1.],
            [1., 1., 0., 0., 1., 1.]
        ])

        display_title("INPUT EMBEDDING")

        display_matrix("Input Matrix (X):", X)

        print("\nInput Matrix Dimensions:", tuple(X.shape))

        with torch.no_grad():

            if choice == 1:

                final_output = run_self_attention(
                    X, generator
                )

            elif choice == 2:

                final_output = run_multi_head_attention(
                    X, 2, generator
                )

            else:

                final_output = run_multi_head_attention(
                    X, 3, generator
                )

        display_title("EXPERIMENT COMPLETED")

        display_matrix(
            "FINAL ATTENTION OUTPUT:",
            final_output
        )

        print("\nWeek 8 Attention Implementation Completed Successfully!")

    except ValueError:
        print("\nPlease enter a valid numerical choice.")


if __name__ == "__main__":
    main()