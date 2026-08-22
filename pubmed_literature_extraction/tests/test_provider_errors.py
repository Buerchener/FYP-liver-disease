import unittest

from cognitive_agent.provider_errors import (
    AUTH_ERROR,
    QUOTA_OR_TOKEN_EXHAUSTED,
    RATE_LIMITED,
    TRANSIENT_PROVIDER_ERROR,
    classify_provider_error,
    is_retryable_provider_error,
)


class ProviderErrorTests(unittest.TestCase):
    def test_auth_and_quota_are_not_retryable(self):
        self.assertEqual(classify_provider_error("Error code 401: Invalid token"), AUTH_ERROR)
        self.assertEqual(classify_provider_error("insufficient balance"), QUOTA_OR_TOKEN_EXHAUSTED)
        self.assertFalse(is_retryable_provider_error("401 Invalid token"))
        self.assertFalse(is_retryable_provider_error("quota exhausted"))

    def test_rate_limit_and_gateway_are_retryable(self):
        self.assertEqual(classify_provider_error("429 too many requests"), RATE_LIMITED)
        self.assertEqual(classify_provider_error("502 bad gateway"), TRANSIENT_PROVIDER_ERROR)
        self.assertEqual(classify_provider_error("Request timed out."), TRANSIENT_PROVIDER_ERROR)
        self.assertTrue(is_retryable_provider_error("429 too many requests"))
        self.assertTrue(is_retryable_provider_error("502 bad gateway"))
        self.assertTrue(is_retryable_provider_error("Request timed out."))


if __name__ == "__main__":
    unittest.main()
