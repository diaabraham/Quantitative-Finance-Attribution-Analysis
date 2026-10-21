"""Market-data provider errors."""


class MarketDataError(Exception):
    """Base error for market-data providers."""


class ProviderAuthError(MarketDataError):
    """Missing or invalid API credentials."""


class ProviderRateLimitError(MarketDataError):
    """Provider rate limit / quota exceeded."""


class ProviderEmptyDataError(MarketDataError):
    """Provider returned no usable rows for the request."""


class ProviderConfigError(MarketDataError):
    """Invalid provider configuration."""
