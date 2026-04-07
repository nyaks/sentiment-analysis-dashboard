"""
Generate a synthetic dataset of 500+ reviews with ratings, dates, categories, and sources.
Produces a CSV file that can be directly loaded into the sentiment dashboard.
"""

import csv
import os
import random
from datetime import datetime, timedelta

# Seed for reproducibility
random.seed(42)

# Configuration
NUM_REVIEWS = 500
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sample_reviews.csv")

# Templates for generating realistic reviews by sentiment
POSITIVE_TEMPLATES = [
    "Absolutely love this {product}! The {feature} is incredible and it exceeded all my expectations.",
    "Best {product} I've ever used. The {feature} works flawlessly and the quality is top-notch.",
    "Really impressed with this {product}. Great {feature} and excellent customer service too.",
    "This {product} is fantastic! The {feature} is exactly what I needed. Highly recommend!",
    "Five stars! This {product} is outstanding. The {feature} makes it worth every penny.",
    "Amazing {product}! I've been using it for weeks and the {feature} is just perfect.",
    "Could not be happier with my {product}. The {feature} alone makes it a great buy.",
    "Excellent purchase! This {product} has the best {feature} in its category.",
    "I'm so glad I bought this {product}. The {feature} is superb and it looks great too.",
    "This {product} changed my life! The {feature} is revolutionary and so easy to use.",
    "Wonderful {product} - the {feature} is outstanding and the build quality is solid.",
    "Top quality {product}. The {feature} works perfectly and shipping was fast too.",
    "Great value for money. This {product} has an amazing {feature} that works beautifully.",
    "Superb {product} with an excellent {feature}. Would definitely buy again!",
    "The {feature} on this {product} is second to none. Very satisfied with my purchase.",
]

NEGATIVE_TEMPLATES = [
    "Terrible {product}. The {feature} stopped working after just a few days.",
    "Very disappointed with this {product}. The {feature} is nothing like advertised.",
    "Worst purchase ever. The {feature} on this {product} is completely unusable.",
    "Do not recommend this {product}. The {feature} keeps malfunctioning.",
    "This {product} is a complete waste of money. The {feature} broke within a week.",
    "Horrible experience with this {product}. The {feature} doesn't work at all.",
    "Cheap quality {product}. The {feature} feels flimsy and unreliable.",
    "I regret buying this {product}. The {feature} is poorly designed.",
    "Save your money. This {product} has a terrible {feature} that never works right.",
    "Extremely frustrated with this {product}. The {feature} failed immediately.",
    "Not worth the price. The {feature} on this {product} is subpar at best.",
    "Awful {product}. The {feature} is broken out of the box and customer service won't help.",
    "Very poor quality {product}. The {feature} stopped working after one use.",
    "Would give zero stars if I could. The {feature} on this {product} is a joke.",
    "Total disappointment. This {product} and its {feature} are both garbage.",
]

NEUTRAL_TEMPLATES = [
    "The {product} is okay. The {feature} works as expected, nothing special.",
    "Decent {product}. The {feature} does what it's supposed to do.",
    "Average {product}. The {feature} is neither great nor terrible.",
    "It's an acceptable {product}. The {feature} could be better but it functions.",
    "The {product} works fine. The {feature} meets basic requirements.",
    "Mixed feelings about this {product}. The {feature} has pros and cons.",
    "Not bad but not great. This {product}'s {feature} is just okay.",
    "The {product} is fairly standard. The {feature} performs adequately.",
    "I have an average opinion of this {product}. The {feature} is mediocre.",
    "This {product} gets the job done. The {feature} is functional but basic.",
]

PRODUCTS = [
    "wireless earbuds", "phone case", "laptop stand", "coffee maker",
    "running shoes", "backpack", "desk lamp", "tablet",
    "smart watch", "keyboard", "monitor", "headphones",
    "camera", "printer", "charger", "mouse",
    "book", "blender", "vacuum cleaner", "speakers"
]

FEATURES = [
    "battery life", "build quality", "sound quality", "screen display",
    "comfort", "durability", "speed", "connectivity",
    "camera quality", "design", "user interface", "value",
    "performance", "packaging", "ease of use", "weight"
]

CATEGORIES = ["Electronics", "Clothing & Shoes", "Home & Kitchen", "Books & Media", "Health & Beauty"]
SOURCES = ["Twitter", "Amazon", "Reddit", "Yelp", "Google Reviews"]

ADDITIONAL_DETAILS = [
    "", " Would buy again.", " Not recommended.", " Still testing it out.",
    " Update: still works!", " Had to return it.", " Got it on sale.",
    " Great for the price.", " Overpriced for what you get.", " Perfect gift idea.",
]


def generate_review(sentiment: str) -> str:
    """Generate a review text based on sentiment category."""
    if sentiment == "positive":
        template = random.choice(POSITIVE_TEMPLATES)
    elif sentiment == "negative":
        template = random.choice(NEGATIVE_TEMPLATES)
    else:
        template = random.choice(NEUTRAL_TEMPLATES)

    review = template.format(
        product=random.choice(PRODUCTS),
        feature=random.choice(FEATURES)
    )

    # Sometimes add additional detail
    if random.random() < 0.4:
        review += random.choice(ADDITIONAL_DETAILS)

    return review


def generate_dataset():
    """Generate a synthetic dataset and save to CSV."""
    rows = []
    start_date = datetime(2024, 1, 1)

    # Distribution: roughly 40% positive, 25% negative, 35% neutral
    sentiment_weights = [("positive", 0.40), ("negative", 0.25), ("neutral", 0.35)]

    for i in range(NUM_REVIEWS):
        # Pick sentiment based on weights
        sentiment = random.choices(
            [s for s, _ in sentiment_weights],
            weights=[w for _, w in sentiment_weights]
        )[0]

        # Map sentiment to a 1-5 rating
        if sentiment == "positive":
            rating = random.choice([4, 4, 5, 5, 5, 4])
        elif sentiment == "negative":
            rating = random.choice([1, 1, 2, 1, 2, 1])
        else:
            rating = random.choice([3, 3, 2, 3, 4, 3])

        # Generate a random date within the past year
        days_offset = random.randint(0, 365)
        review_date = start_date + timedelta(days=days_offset)

        category = random.choice(CATEGORIES)
        source = random.choice(SOURCES)

        review_text = generate_review(sentiment)

        rows.append({
            "id": i + 1,
            "date": review_date.strftime("%Y-%m-%d"),
            "text": review_text,
            "rating": rating,
            "category": category,
            "source": source,
        })

    # Shuffle to mix things up
    random.shuffle(rows)

    # Write CSV
    fieldnames = ["id", "date", "text", "rating", "category", "source"]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {NUM_REVIEWS} reviews -> {OUTPUT_FILE}")
    print(f"File size: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")

    # Print distribution
    sentiment_counts = {}
    for row in rows:
        sentiment = "positive" if row["rating"] >= 4 else ("negative" if row["rating"] <= 2 else "neutral")
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
    print("\nSynthetic label distribution (mapped from rating):")
    for s in ["positive", "neutral", "negative"]:
        print(f"  {s}: {sentiment_counts.get(s, 0)}")


if __name__ == "__main__":
    generate_dataset()
