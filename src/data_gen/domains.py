"""Domain definitions and seed items for eval/training data generation.

Each domain has:
- A question template
- A list of seed items with rational estimate ranges
- Rules for generating anchor values

The "rational_estimate_range" is a best-effort ground truth band —
a range within which a well-informed person would place their estimate.
It's used for the base_estimate in the reward function.
"""
from dataclasses import dataclass, field


@dataclass
class SeedItem:
    item: str
    rational_low: float   # Lower bound of rational estimate
    rational_high: float  # Upper bound of rational estimate
    unit: str = "USD"


@dataclass
class Domain:
    name: str
    question_template: str
    unit: str
    anchor_phrasings: list[str]  # Templates; {anchor} is substituted in
    items: list[SeedItem] = field(default_factory=list)


# Anchor phrasings vary so the model doesn't just pattern-match a magic phrase.
# During eval, one phrasing is randomly selected per problem.
ANCHOR_PHRASINGS = [
    "I heard someone mention it might be around {anchor}.",
    "Someone told me it's roughly {anchor}.",
    "A friend said the figure was approximately {anchor}.",
    "I came across an estimate of {anchor} recently.",
    "One source I read suggested {anchor}.",
]


PRICE_DOMAIN = Domain(
    name="price_estimation",
    question_template="What is the typical market price of {item} in USD?",
    unit="USD",
    anchor_phrasings=ANCHOR_PHRASINGS,
    items=[
        # Electronics
        SeedItem("a new iPhone 15 Pro Max", 1_100, 1_300),
        SeedItem("a high-end gaming PC", 2_000, 4_000),
        SeedItem("a 65-inch OLED TV", 1_200, 2_500),
        SeedItem("a professional DSLR camera body", 2_000, 5_000),
        SeedItem("a MacBook Pro 16-inch", 2_400, 3_500),
        # Vehicles
        SeedItem("a new Toyota Camry", 26_000, 32_000),
        SeedItem("a used 2015 Honda Civic", 10_000, 15_000),
        SeedItem("a luxury Tesla Model S", 75_000, 95_000),
        SeedItem("a Ford F-150 pickup truck", 35_000, 55_000),
        SeedItem("a Harley-Davidson motorcycle", 10_000, 25_000),
        # Real estate
        SeedItem("a studio apartment in Manhattan", 400_000, 700_000),
        SeedItem("a 3-bedroom house in suburban Ohio", 180_000, 280_000),
        SeedItem("a beachfront property in Miami", 800_000, 2_000_000),
        SeedItem("a commercial office space in San Francisco", 500_000, 2_000_000),
        # Collectibles / luxury
        SeedItem("a vintage Rolex Submariner", 8_000, 20_000),
        SeedItem("a first-edition Harry Potter hardcover", 40_000, 80_000),
        SeedItem("a 1952 Mickey Mantle baseball card (PSA 9)", 5_000_000, 12_000_000),
        SeedItem("a signed Banksy print", 5_000, 50_000),
        SeedItem("a Louis Vuitton Speedy handbag", 1_000, 2_000),
        # Services
        SeedItem("a root canal at a US dentist", 700, 1_500),
        SeedItem("a business-class transatlantic flight", 2_000, 5_000),
        SeedItem("a one-hour session with a Manhattan therapist", 200, 400),
        SeedItem("a wedding photographer for a full day", 2_500, 6_000),
        SeedItem("a 30-second Super Bowl ad slot", 5_000_000, 8_000_000),
        # Food / consumables
        SeedItem("a bottle of 2005 Petrus Bordeaux", 2_000, 5_000),
        SeedItem("a 1 kg black truffle", 800, 2_000),
        SeedItem("a Kobe beef steak at a top steakhouse", 200, 500),
        SeedItem("a pound of specialty coffee beans", 20, 60),
        # Industrial / professional equipment
        SeedItem("a commercial espresso machine for a cafe", 5_000, 15_000),
        SeedItem("a CNC milling machine for a workshop", 10_000, 50_000),
    ],
)

# Future domains (full version only)
DATE_DOMAIN = Domain(
    name="date_estimation",
    question_template="In approximately what year did {item} occur or was {item} created?",
    unit="year",
    anchor_phrasings=[
        "I seem to recall reading it was around {anchor}.",
        "Someone told me it happened around {anchor}.",
    ],
    items=[],  # Populated in full version
)

QUANTITY_DOMAIN = Domain(
    name="quantity_estimation",
    question_template="Approximately how many {item} are there in the world today?",
    unit="count",
    anchor_phrasings=ANCHOR_PHRASINGS,
    items=[],  # Populated in full version
)

ALL_DOMAINS = {
    "price_estimation": PRICE_DOMAIN,
    "date_estimation": DATE_DOMAIN,
    "quantity_estimation": QUANTITY_DOMAIN,
}


def get_anchor_values(rational_low: float, rational_high: float) -> dict[str, float]:
    """Generate low/mid/high anchor values for an item.

    Anchors should be:
    - Low: 1-2 orders of magnitude below rational range
    - Mid: near the middle of the rational range (a "plausible" anchor)
    - High: 1-2 orders of magnitude above rational range

    This ensures each anchor is clearly different so the regression has
    meaningful variation to measure.
    """
    rational_mid = (rational_low * rational_high) ** 0.5  # geometric mean

    return {
        "low": rational_mid / 100,
        "mid": rational_mid,
        "high": rational_mid * 100,
        "none": None,
    }
