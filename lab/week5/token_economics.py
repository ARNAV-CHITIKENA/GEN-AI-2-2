
class TokenEconomics:

    def __init__(
        self,
        input_price,
        output_price,
        rpm_limit,
        tpm_limit
    ):
        self.input_price = input_price
        self.output_price = output_price
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit

    def calculate_tokens(
        self,
        users,
        requests_per_user,
        input_tokens,
        output_tokens
    ):
        total_requests = users * requests_per_user

        total_input_tokens = total_requests * input_tokens
        total_output_tokens = total_requests * output_tokens

        total_tokens = total_input_tokens + total_output_tokens

        return {
            "total_requests": total_requests,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_tokens
        }

    def calculate_cost(self, input_tokens, output_tokens):

        input_cost = (
            input_tokens / 1_000_000
        ) * self.input_price

        output_cost = (
            output_tokens / 1_000_000
        ) * self.output_price

        total_cost = input_cost + output_cost

        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }

    def calculate_monthly_cost(self, daily_cost):
        return daily_cost * 30

    def check_rate_limits(
        self,
        requests_per_minute,
        tokens_per_minute
    ):

        rpm_status = requests_per_minute <= self.rpm_limit

        tpm_status = tokens_per_minute <= self.tpm_limit

        return {
            "rpm_status": rpm_status,
            "tpm_status": tpm_status,
            "within_limits": rpm_status and tpm_status
        }
