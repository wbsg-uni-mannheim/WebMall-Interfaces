from woocommerce import API
import os


def get_woocommerce_client(wc_url=None, wc_consumer_key=None, wc_consumer_secret=None, wc_version='wc/v3'):
    """
    Initialize and return a WooCommerce API client.

    Returns:
        WooCommerce API client instance
    """
    print(wc_url)
    print(wc_consumer_key)
    print(wc_consumer_secret)
    print(wc_version)
    # Get WooCommerce configuration from environment variables
    if wc_url is None:
        wc_url = os.getenv('WC_URL')
    if wc_consumer_key is None:
        wc_consumer_key = os.getenv('WC_CONSUMER_KEY')
    if wc_consumer_secret is None:
        wc_consumer_secret = os.getenv('WC_CONSUMER_SECRET')
    if wc_version is None:
        wc_version = os.getenv('WC_VERSION', 'wc/v3')

    # Validate required environment variables
    if not wc_url:
        raise ValueError("WC_URL environment variable is required")
    if not wc_consumer_key:
        raise ValueError("WC_CONSUMER_KEY environment variable is required")
    if not wc_consumer_secret:
        raise ValueError("WC_CONSUMER_SECRET environment variable is required")

    # Create and return the WooCommerce API client
    wcapi = API(
        url=wc_url,
        consumer_key=wc_consumer_key,
        consumer_secret=wc_consumer_secret,
        version=wc_version,
        timeout=30
    )

    return wcapi
