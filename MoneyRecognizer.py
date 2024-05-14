from presidio_analyzer import Pattern, PatternRecognizer

class MoneyRecognizer(PatternRecognizer):
    """
    Recognize money-related patterns using regex.
    """

    PATTERNS = [
        Pattern("Dollar", r"\$\d+(?:,\d+)*(?:\.\d+)?", 0.5),
        Pattern("Dollar with cents", r"\$\d+(?:,\d+)*\.\d+", 0.7),
        Pattern("Dollar with cents", r"\$\d+\.\d+", 0.7),
        Pattern("Dollar with cents", r"\d+(?:,\d+)*\.\d+\$", 0.7),
        # Add more patterns for other representations of money amounts
    ]

    CONTEXT = ["money", "dollar", "cents", "usd"]

    def __init__(self):
        super().__init__(
            supported_entity="MONEY",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
        )

    def analyze(self, text):
        return self.analyze_patterns(text)

