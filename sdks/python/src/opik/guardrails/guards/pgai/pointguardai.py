import os
from typing import List, Dict, Any, Optional

from ...guards import guard
from ...schemas import ValidationType


# Default PointGuardAi API base URL - can be overridden via PointGuardAi_BASE_URL env var
# for different environments (staging, production, etc.)
POINTGUARDAI_DEFAULT_BASE_URL = "https://api.sqa1.appsoc.com/aisec-rdc/api/v1"
POINTGUARDAI_DEFAULT_ORG = "demodevelop"


class PointGuardAi(guard.Guard):
    """
    Guard that validates text using the PointGuardAi (AppSoc) external API.

    Handles policy-based validation for both input and output text,
    supporting separate endpoints for input validation and output validation.
    """

    # PointGuardAi runs on an external API, not on the local Opik Guardrails backend
    remote = True

    def __init__(
        self,
        policy_name: str,
        api_key: Optional[str] = None,
        org: Optional[str] = None,
    ) -> None:
        """
        Initialize a PointGuardAi guard.

        Args:
            policy_name: Required - The name of the PointGuardAi policy to use for validation
            api_key: API key for PointGuardAi authentication. Falls back to PointGuardAi_API_KEY env var
            org: Organization name for AppSoc. Falls back to PointGuardAi_ORG env var

        Raises:
            ValueError: If policy_name, api_key, or org is not provided

        Environment Variables:
            POINTGUARDAI_API_KEY: API key for authentication (used if api_key not provided)
            POINTGUARDAI_ORG: Organization name (used if org not provided)
            POINTGUARDAI_BASE_URL: Optional override for the API base URL (for different environments)

        Example:
            ```python
            from opik.guardrails import Guardrail, PointGuardAi

            guard = Guardrail(guards=[
                PointGuardAi(policy_name="my-policy-001")
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
        self.api_key = api_key or os.getenv("POINTGUARDAI_API_KEY")
        self.org = org or os.getenv("POINTGUARDAI_ORG", POINTGUARDAI_DEFAULT_ORG)
        
        # Base URL is internal - use default or allow override via env var
        self.base_url = os.getenv("POINTGUARDAI_BASE_URL", POINTGUARDAI_DEFAULT_BASE_URL)

        if not self.api_key:
            raise ValueError(
                "api_key is required. Provide it as an argument or set POINTGUARDAI_API_KEY environment variable"
            )
        
        if not self.org:
            raise ValueError(
                "org is required. Provide it as an argument or set POINTGUARDAI_ORG environment variable"
            )

    def get_validation_configs(self) -> List[Dict[str, Any]]:
        """
        Get the validation configuration for PointGuardAi.

        Note: PointGuardAi uses a different validation flow via its own API client,
        so this returns the policy configuration for reference purposes.

        Returns:
            List containing the PointGuardAi validation configuration
        """
        return [
            {
                "type": ValidationType.PointGuardAi,
                "config": {
                    "policy_name": self.policy_name,
                },
            }
        ]
