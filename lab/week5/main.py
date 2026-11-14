
import os
from dotenv import load_dotenv

from token_economics import TokenEconomics

load_dotenv()

# Feature usage assumptions
USERS_PER_DAY = 1000
REQUESTS_PER_USER_PER_DAY = 5
INPUT_TOKENS_PER_REQUEST = 500
OUTPUT_TOKENS_PER_REQUEST = 300
PEAK_REQUESTS_PER_MINUTE = 10


def main():

    print("=" * 60)
    print("       TOKEN ECONOMICS - BUDGET CALCULATOR")
    print("=" * 60)

    # Only user input: select the provider
    provider = input(
        "Choose provider (gemini/ollama): "
    ).strip().lower()

    if provider == "gemini":

        input_price = float(
            os.getenv("GEMINI_INPUT_PRICE_PER_MILLION", 0)
        )

        output_price = float(
            os.getenv("GEMINI_OUTPUT_PRICE_PER_MILLION", 0)
        )

        rpm_limit = int(
            os.getenv("GEMINI_RPM_LIMIT", 15)
        )

        tpm_limit = int(
            os.getenv("GEMINI_TPM_LIMIT", 1000000)
        )

        model = "gemini"

    elif provider == "ollama":

        input_price = float(
            os.getenv("OLLAMA_INPUT_PRICE_PER_MILLION", 0)
        )

        output_price = float(
            os.getenv("OLLAMA_OUTPUT_PRICE_PER_MILLION", 0)
        )

        rpm_limit = int(
            os.getenv("OLLAMA_RPM_LIMIT", 60)
        )

        tpm_limit = int(
            os.getenv("OLLAMA_TPM_LIMIT", 1000000)
        )

        model = "Ollama"

    else:
        print("Invalid provider selected. Please choose gemini or ollama.")
        return

    # Predefined feature usage assumptions
    users = USERS_PER_DAY
    requests_per_user = REQUESTS_PER_USER_PER_DAY
    input_tokens = INPUT_TOKENS_PER_REQUEST
    output_tokens = OUTPUT_TOKENS_PER_REQUEST

    # Predefined peak usage
    peak_requests = PEAK_REQUESTS_PER_MINUTE

    peak_tokens = peak_requests * (
        input_tokens + output_tokens
    )

    # Initialize calculator
    calculator = TokenEconomics(
        input_price,
        output_price,
        rpm_limit,
        tpm_limit
    )

    # Calculate token usage
    token_data = calculator.calculate_tokens(
        users,
        requests_per_user,
        input_tokens,
        output_tokens
    )

    # Calculate costs
    cost_data = calculator.calculate_cost(
        token_data["input_tokens"],
        token_data["output_tokens"]
    )

    monthly_cost = calculator.calculate_monthly_cost(
        cost_data["total_cost"]
    )

    # Check rate limits
    rate_data = calculator.check_rate_limits(
        peak_requests,
        peak_tokens
    )

    # Display report
    print("\n" + "=" * 60)
    print("              TOKEN ECONOMICS REPORT")
    print("=" * 60)

    print("Selected Provider:", model)

    print("\nFEATURE USAGE")
    print("Users per day:", USERS_PER_DAY)
    print("Requests per user:", REQUESTS_PER_USER_PER_DAY)
    print("Input tokens per request:", INPUT_TOKENS_PER_REQUEST)
    print("Output tokens per request:", OUTPUT_TOKENS_PER_REQUEST)

    print("\nTOKEN USAGE")
    print("Daily Requests:", int(token_data["total_requests"]))
    print("Daily Input Tokens:", int(token_data["input_tokens"]))
    print("Daily Output Tokens:", int(token_data["output_tokens"]))
    print("Total Daily Tokens:", int(token_data["total_tokens"]))

    print("\nCOST ESTIMATION")
    print("Input Cost: $", round(cost_data["input_cost"], 6))
    print("Output Cost: $", round(cost_data["output_cost"], 6))
    print("Daily Cost: $", round(cost_data["total_cost"], 6))
    print("Monthly Cost: $", round(monthly_cost, 6))

    print("\nRATE LIMIT ANALYSIS")
    print("Peak RPM:", peak_requests)
    print("Allowed RPM:", rpm_limit)

    print(
        "RPM Status:",
        "PASS" if rate_data["rpm_status"] else "EXCEEDED"
    )

    print("Peak TPM:", int(peak_tokens))
    print("Allowed TPM:", tpm_limit)

    print(
        "TPM Status:",
        "PASS" if rate_data["tpm_status"] else "EXCEEDED"
    )

    print("\nFINAL RESULT")

    if rate_data["within_limits"]:
        print("Feature is within the configured rate limits.")
    else:
        print("Feature exceeds the configured rate limits.")

    print("=" * 60)


if __name__ == "__main__":
    main()
