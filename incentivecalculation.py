from pydantic import BaseModel, Field

# Prompt describing the tool
CALCULATE_INCENTIVE_PROMPT = """
This tool calculates the incentive to pay to a reviewer based on their review score. 
The score ranges between 1 and 100, and the incentive is a value between 10^-6 and 10^-4.
"""

# Input argument schema
class CalculateIncentiveInput(BaseModel):
    """Input argument schema for calculate incentive action."""

    score: float = Field(
        ...,
        ge=1.0,
        le=100.0,
        description="The review score provided by the reviewer. Should be a float between 1.0 and 100.0.",
        example=85.5
    )

def calculate_incentive(score: float) -> float:
    """
    Calculate the incentive to pay a reviewer based on their score.

    Args:
        score (float): The review score, a float between 1.0 and 100.0.

    Returns:
        float: The calculated incentive, a value between 10^-6 and 10^-4.
    """
    # Scale the score to the range 10^-6 to 10^-4 using a linear transformation
    min_incentive = 1e-6
    max_incentive = 1e-4

    incentive = min_incentive + ((score - 1.0) / (100.0 - 1.0)) * (max_incentive - min_incentive)
    return incentive