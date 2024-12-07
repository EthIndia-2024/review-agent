from pydantic import BaseModel, Field
from cdp_langchain.tools import CdpTool
from cdp import Wallet
from textblob import TextBlob
# import nltk
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')

HELPFULNESS_PROMPT = """
This tool evaluates how helpful a review is to a company by analyzing its descriptiveness, sentiment, actionability, uniqueness, specificity, and length adequacy.
"""

class CheckReviewHelpfulnessInput(BaseModel):
    """Input schema for the review helpfulness tool."""

    review_text: str = Field(
        ...,
        description="The text of the review to evaluate.",
        example="The product is amazing but lacks a proper user manual, which made it difficult to set up."
    )
    # existing_reviews: list[str] = Field(
    #     ...,
    #     description="A list of existing reviews for comparison (to measure uniqueness).",
    #     example=["Great product!", "Easy to use and well-built."]
    # )

def calculate_review_helpfulness(review_text: str) -> str:
    """
    Calculate the helpfulness score of a review.

    Args:
        review_text (str): The text of the review.
        existing_reviews (list[str]): A list of existing reviews for comparison.

    Returns:
        str: A summary of the helpfulness score and contributing factors.
    """
    # Analyze descriptiveness
    blob = TextBlob(review_text)
    adjectives = sum(1 for _, tag in blob.tags if tag in {"JJ", "JJR", "JJS"})
    adverbs = sum(1 for _, tag in blob.tags if tag in {"RB", "RBR", "RBS"})
    word_count = len(blob.words)
    descriptiveness_score = ((adjectives + adverbs) / word_count) * 100 if word_count else 0

    # Actionability score (dummy logic for simplicity)
    actionable_keywords = [
    "fix", "improve", "enhance", "upgrade", "address", "modify", "correct", 
    "adjust", "update", "problem", "issue", "bug", "crash", "error", "defect", "lacks",
    "malfunction", "broken", "glitch", "recommend", "suggest", "consider", 
    "would prefer", "should add", "needs", "could be better", "option for", 
    "feature", "functionality", "option", "performance", "speed", "usability", 
    "compatibility", "design", "quality", "durability", "customer service", 
    "delivery", "support", "response", "shipping", "instructions", 
    "communication", "setup", "disappointed", "frustrated", "annoyed", 
    "confused", "unclear", "hard to use", "not satisfied"
]
    actionability_score = 100 if any(word in review_text.lower() for word in actionable_keywords) else 50

    # Specificity score
    specificity_score = 100 if len(blob.noun_phrases) > 2 else 50

    # Length adequacy score
    length = len(review_text.split())
    if 50 <= length <= 200:
        length_score = 100
    elif 20 <= length <= 50 or 200 <= length <= 500:
        length_score = 75
    else:
        length_score = 50

    # Final score
    final_score = (
        0.10 * descriptiveness_score +
        0.30 * actionability_score +
        0.30 * specificity_score +
        0.30 * length_score
    )

    return (
        f"Review Helpfulness Score: {final_score:.2f}\n"
        f"Contributing Scores:\n"
        f"- Descriptiveness: {descriptiveness_score:.2f}\n"
        f"- Actionability: {actionability_score:.2f}\n"
        f"- Specificity: {specificity_score:.2f}\n"
        f"- Length Adequacy: {length_score:.2f}"
    )

# print(calculate_review_helpfulness("The product is amazing but lacks a proper user manual, which made it difficult to set up."))
# print(calculate_review_helpfulness("The product works well overall, but the instructions for setup were confusing and lacked clarity. It would be great if you could include a step-by-step guide with diagrams for better understanding. Also, the app crashes occasionally when I try to upload large files. Fixing this issue would improve usability significantly."))
