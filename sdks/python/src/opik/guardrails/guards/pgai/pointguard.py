import os
from typing import List, Dict, Any, Optional

from ...guards import guard
from ...schemas import ValidationType


# Default PointGuard API base URL - can be overridden via POINTGUARD_BASE_URL env var
# for different environments (staging, production, etc.)
POINTGUARD_DEFAULT_BASE_URL = "https://api.sqa1.appsoc.com/aisec-rdc/api/v1"
POINTGUARD_DEFAULT_ORG = "demodevelop"


class PointGuard(guard.Guard):
    """
    Guard that validates text using the PointGuard (AppSoc) external API.

    Handles policy-based validation for both input and output text,
    supporting separate endpoints for input validation and output validation.
    """

    # PointGuard runs on an external API, not on the local Opik Guardrails backend
    remote = True

    def __init__(
        self,
        policy_name: str,
        api_key: Optional[str] = None,
        org: Optional[str] = None,
    ) -> None:
        """
        Initialize a PointGuard guard.

        Args:
            policy_name: Required - The name of the PointGuard policy to use for validation
            api_key: API key for PointGuard authentication. Falls back to POINTGUARD_API_KEY env var
            org: Organization name for AppSoc. Falls back to POINTGUARD_ORG env var

        Raises:
            ValueError: If policy_name, api_key, or org is not provided

        Environment Variables:
            POINTGUARD_API_KEY: API key for authentication (used if api_key not provided)
            POINTGUARD_ORG: Organization name (used if org not provided)
            POINTGUARD_BASE_URL: Optional override for the API base URL (for different environments)

        Example:
            ```python
            from opik.guardrails import Guardrail, PointGuard

            guard = Guardrail(guards=[
                PointGuard(policy_name="my-policy-001")
            ])

            # Validate input before LLM call
            guard.validate_input("User's question here")

            # Validate output after LLM response
            guard.validate_output(
                input_text="User's question here",
                output_text="LLM's response here"
            )
            ```
        """
        if not policy_name:
            raise ValueError("policy_name is required")

        self.policy_name = policy_name
        self.api_key = api_key or os.getenv("POINTGUARD_API_KEY")
        self.org = org or os.getenv("POINTGUARD_ORG", POINTGUARD_DEFAULT_ORG)
        
        # Base URL is internal - use default or allow override via env var
        self.base_url = os.getenv("POINTGUARD_BASE_URL", POINTGUARD_DEFAULT_BASE_URL)

        if not self.api_key:
            raise ValueError(
                "api_key is required. Provide it as an argument or set POINTGUARD_API_KEY environment variable"
            )
        
        if not self.org:
            raise ValueError(
                "org is required. Provide it as an argument or set POINTGUARD_ORG environment variable"
            )

    def get_validation_configs(self) -> List[Dict[str, Any]]:
        """
        Get the validation configuration for PointGuard.

        Note: PointGuard uses a different validation flow via its own API client,
        so this returns the policy configuration for reference purposes.

        Returns:
            List containing the PointGuard validation configuration
        """
        return [
            {
                "type": ValidationType.POINTGUARD,
                "config": {
                    "policy_name": self.policy_name,
                },
            }
        ]
