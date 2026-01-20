import pydantic
import enum
from typing import Dict, Any, List, Optional

from opik.rest_api.types.check_public_result import CheckPublicResult


class ValidationType(str, enum.Enum):
    PII = "PII"
    TOPIC = "TOPIC"
    POINTGUARD = "POINTGUARD"


class ValidationResult(pydantic.BaseModel):
    validation_passed: bool
    type: ValidationType
    validation_config: Dict[str, Any]
    validation_details: Dict[str, Any]


class ValidationResponse(pydantic.BaseModel):
    validation_passed: bool
    validations: List[ValidationResult]
    # This is client side injected
    guardrail_result: Optional[CheckPublicResult] = None


class PointGuardValidationDetails(pydantic.BaseModel):
    """Details of a PointGuard validation result."""

    blocked: bool
    modified: bool
    violations: List[Dict[str, Any]]
    modified_content: Optional[str] = None


class PointGuardValidationResponse(ValidationResponse):
    """Response from a PointGuard validation request."""

    details: PointGuardValidationDetails

    def get_validated_content(self, original_content: str) -> str:
        """
        Get the content to use after validation.
        
        Returns modified content if PII was masked/redacted, otherwise returns original content.
        
        Args:
            original_content: The original text that was validated
            
        Returns:
            str: The content to use (modified if PII was redacted, otherwise original)
            
        Example:
            ```python
            result = guard.validate_input("My SSN is 123-45-6789")
            safe_content = result.get_validated_content("My SSN is 123-45-6789")
            # safe_content might be "My SSN is [REDACTED]"
            ```
        """
        if self.details.modified and self.details.modified_content:
            return self.details.modified_content
        return original_content
    
    @property
    def status(self) -> str:
        """
        Get a simple status string for the validation.
        
        Returns:
            str: "blocked", "modified", or "passed"
        """
        if self.details.blocked:
            return "blocked"
        elif self.details.modified:
            return "modified"
        return "passed"
    
    def summary(self) -> str:
        """
        Get a human-readable summary of the validation result.
        
        Returns:
            str: Formatted summary including status and violations
            
        Example:
            ```python
            result = guard.validate_input(query)
            print(result.summary())
            # Output: "✓ PASSED - No violations detected"
            # or: "⚠ MODIFIED - 1 violation: PII redacted"
            # or: "✗ BLOCKED - 2 violations: harmful_content, toxicity"
            ```
        """
        status_icons = {"blocked": "✗", "modified": "⚠", "passed": "✓"}
        status_text = self.status.upper()
        icon = status_icons.get(self.status, "•")
        
        if self.details.blocked:
            violation_types = [v.get("type", "unknown") for v in self.details.violations]
            return f"{icon} {status_text} - {len(violation_types)} violation(s): {', '.join(violation_types)}"
        elif self.details.modified:
            violation_types = [v.get("type", "unknown") for v in self.details.violations]
            return f"{icon} {status_text} - {len(violation_types)} violation(s): {', '.join(violation_types)}"
        else:
            return f"{icon} {status_text} - No violations detected"
