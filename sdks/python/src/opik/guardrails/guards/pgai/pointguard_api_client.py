from typing import Optional, List, Dict, Any

import httpx
from ...schemas import PointGuardValidationDetails, PointGuardValidationResponse, ValidationType, ValidationResult


class PointGuardApiClient:
    """
    API client for the PointGuard (AppSoc) guardrails service.

    Handles communication with PointGuard's input and output validation endpoints
    using the AppSoc API format.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        org: str,
        timeout: Optional[int] = None,
    ) -> None:
        """
        Initialize the PointGuard API client.

        Args:
            base_url: Base URL for the PointGuard API
            api_key: API key for authentication (AppSoc API key)
            org: Organization name for AppSoc
            timeout: Request timeout in seconds (default: 30)
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._org = org
        self._timeout = timeout or 30
        self._httpx_client = httpx.Client(timeout=self._timeout)

    def __del__(self) -> None:
        """Clean up the HTTP client on deletion."""
        if hasattr(self, "_httpx_client"):
            self._httpx_client.close()

    def _get_headers(self) -> dict:
        """Get the common headers for API requests."""
        return {
            "Content-Type": "application/json",
            "X-appsoc-api-key": self._api_key,
        }

    def validate_input(
        self,
        text: str,
        policy_name: str,
correlation_key: Optional[str] = None,
    ) -> PointGuardValidationResponse:
        """
        Validate input text using the PointGuard V2 input endpoint.

        Args:
            text: The input text to validate
            policy_name: The name of the policy to use for validation
            correlation_key: Optional tag/identifier for this request

        Returns:
            PointGuardValidationResponse containing the validation result

        Raises:
            httpx.HTTPStatusError: If the API returns an error status code
        """
        payload = {
            "policyName": policy_name,
            "input": [{"role": "user", "content": text}],
        }
        
        if correlation_key:
            payload["correlationKey"] = correlation_key

        response = self._httpx_client.post(
            f"{self._base_url}/aisec-rdc-v2/api/v1/orgs/{self._org}/inspect/input",
            json=payload,
            headers=self._get_headers(),
        )
        response.raise_for_status()

        return self._parse_input_response(response.json(), policy_name, text)

    def validate_output(
        self,
        input_text: str,
        output_text: str,
        policy_name: str,
        correlation_key: Optional[str] = None,
    ) -> PointGuardValidationResponse:
        """
        Validate output text using the PointGuard V2 output endpoint.

        Args:
            input_text: The original input text (for context)
            output_text: The LLM output text to validate
            policy_name: The name of the policy to use for validation
            correlation_key: Optional tag/identifier for this request

        Returns:
            PointGuardValidationResponse containing the validation result

        Raises:
            httpx.HTTPStatusError: If the API returns an error status code
        """
        payload = {
            "policyName": policy_name,
            "input": [{"role": "user", "content": input_text}],
            "output": [{"role": "user", "content": output_text}],
        }
        
        if correlation_key:
            payload["correlationKey"] = correlation_key

        response = self._httpx_client.post(
            f"{self._base_url}/aisec-rdc-v2/api/v1/orgs/{self._org}/inspect/output",
            json=payload,
            headers=self._get_headers(),
        )
        response.raise_for_status()

        return self._parse_output_response(response.json(), policy_name, output_text)

    def _parse_input_response(
        self,
        response_data: dict,
        policy_name: str,
        original_text: str,
    ) -> PointGuardValidationResponse:
        """
        Parse the V2 input validation API response.

        V2 API response format:
        {
            "policyName": "...",
            "correlationKey": "...",
            "input": {
                "blocked": bool,
                "modified": bool,
                "content": [{
                    "role": "user",
                    "originalContent": "...",
                    "modifiedContent": "..." (or null),
                    "dlpViolations": [...],
                    "aiViolations": [...],
                }]
            }
        }
        """
        input_result = response_data.get("input", {})
        return self._parse_result_section(input_result, policy_name, original_text)

    def _parse_output_response(
        self,
        response_data: dict,
        policy_name: str,
        original_text: str,
    ) -> PointGuardValidationResponse:
        """
        Parse the V2 output validation API response.

        V2 API response format:
        {
            "policyName": "...",
            "correlationKey": "...",
            "input": { ... },
            "output": {
                "blocked": bool,
                "modified": bool,
                "content": [{
                    "role": "user",
                    "originalContent": "...",
                    "modifiedContent": "..." (or null),
                    "dlpViolations": [...],
                    "aiViolations": [...],
                }]
            }
        }
        """
        output_result = response_data.get("output", {})
        return self._parse_result_section(output_result, policy_name, original_text)

    def _parse_result_section(
        self,
        result_section: dict,
        policy_name: str,
        original_text: str,
    ) -> PointGuardValidationResponse:
        """
        Parse a result section (input or output) from the V2 API response.

        Args:
            result_section: The "input" or "output" section of the response
            policy_name: The policy name used for validation
            original_text: The original text that was validated

        Returns:
            PointGuardValidationResponse with parsed data
        """
        blocked = result_section.get("blocked", False)
        modified = result_section.get("modified", False)
        content_items = result_section.get("content", [])

        # Extract violations and modified content from first content item
        violations: List[Dict[str, Any]] = []
        modified_content: Optional[str] = None

        if content_items:
            item = content_items[0]
            # Combine DLP and AI violations
            dlp_violations = item.get("dlpViolations", [])
            ai_violations = item.get("aiViolations", [])
            
            # Normalize violations to a consistent format
            for v in dlp_violations + ai_violations:
                violations.append({
                    "name": v.get("name"),
                    "type": v.get("type"),
                    "action": v.get("action"),
                    "categories": v.get("categories", []),
                    "matchCount": v.get("matchCount", 0),
                })

            # Get modified content if available (V2 uses modifiedContent field)
            if modified:
                modified_content = item.get("modifiedContent")

        validation_details = PointGuardValidationDetails(
            blocked=blocked,
            modified=modified,
            violations=violations,
            modified_content=modified_content,
        )

        # Validation passes if not blocked
        validation_passed = not blocked

        validation_result = ValidationResult(
            validation_passed=validation_passed,
            type=ValidationType.POINTGUARD,
            validation_config={"policy_name": policy_name},
            validation_details=validation_details.model_dump(),
        )

        return PointGuardValidationResponse(
            validation_passed=validation_passed,
            validations=[validation_result],
            details=validation_details,
        )
