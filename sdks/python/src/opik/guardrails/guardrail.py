from typing import (
    List,
    Optional,
    Union,
)

import httpx

import opik.exceptions as exceptions
import opik.config as config
from opik.api_objects import opik_client
from opik.message_processing.messages import (
    GuardrailBatchItemMessage,
    GuardrailBatchMessage,
)
from opik.opik_context import get_current_span_data, get_current_trace_data

from . import guards, rest_api_client, schemas, tracing
from .guards.pgai.pointguardai import PointGuardAi
from .guards.pgai.pointguardai_api_client import PointGuardAiApiClient

GUARDRAIL_DECORATOR = tracing.GuardrailsTrackDecorator()


class Guardrail:
    """
    Client for the Opik Guardrails API.

    This class provides a way to validate text against a set of guardrails.
    """

    def __init__(
        self,
        guards: List[guards.Guard],
        guardrail_timeout: Optional[int] = None,
    ) -> None:
        """
        Initialize a Guardrail client.

        Args:
            guards: List of Guard objects to validate text against

        Example:

        ```python
        from opik.guardrails import Guardrail, PII, Topic
        from opik import exceptions
        guard = Guardrail(
            guards=[
                Topic(restricted_topics=["finance"], threshold=0.8),
                PII(blocked_entities=["CREDIT_CARD", "PERSON"], threshold=0.4),
            ]
        )

        result = guard.validate("How can I start with evaluation in Opik platform?")
        # Guardrail passes

        try:
            result = guard.validate("Where should I invest my money?")
        except exceptions.GuardrailValidationFailed as e:
            print("Guardrail failed:", e)

        try:
            result = guard.validate("John Doe, here is my card number 4111111111111111 how can I use it in Opik platform?.")
        except exceptions.GuardrailValidationFailed as e:
            print("Guardrail failed:", e)
        ```

        """
        self.guards = guards
        self._client = opik_client.get_client_cached()

        self.config_ = config.get_from_user_inputs(
            guardrail_timeout=guardrail_timeout,
        )

        self._initialize_api_client(
            host_url=self._client.config.guardrails_backend_host,
        )

        # Initialize PointGuardAi API clients for any PointGuardAi guards
        self._pointguardai_clients: dict[str, PointGuardAiApiClient] = {}
        self._pointguardai_guards: List[PointGuardAi] = []

        for guard in guards:
            if isinstance(guard, PointGuardAi):
                self._pointguardai_guards.append(guard)
                # Create a client for this PointGuardAi if we don't have one for its base_url
                if guard.base_url not in self._pointguardai_clients:
                    self._pointguardai_clients[guard.base_url] = (
                        PointGuardAiApiClient(
                            base_url=guard.base_url,
                            api_key=guard.api_key,
                            org=guard.org,
                            timeout=self.config_.guardrail_timeout,
                        )
                    )

    def _initialize_api_client(self, host_url: str) -> None:
        self._api_client = rest_api_client.GuardrailsApiClient(
            httpx_client=httpx.Client(timeout=self.config_.guardrail_timeout),
            host_url=host_url,
        )

    def validate(self, text: str) -> schemas.ValidationResponse:
        """
        Validate text against all configured guardrails.

        Args:
            text: Text to validate

        Returns:
            ValidationResponse: API response containing validation results

        Raises:
            opik.exceptions.GuardrailValidationFailed: If validation fails
            httpx.HTTPStatusError: If the API returns an error status code
        """
        result: schemas.ValidationResponse = self._validate(generation=text)  # type: ignore

        return self._parse_result(result)

    @GUARDRAIL_DECORATOR.track
    def _validate(self, generation: str) -> schemas.ValidationResponse:
        validations = []

        for guard in self.guards:
            validations.extend(guard.get_validation_configs())

        result = self._api_client.validate(generation, validations)

        if not result.validation_passed:
            result.guardrail_result = "failed"
        else:
            result.guardrail_result = "passed"

        batch = []

        # Makes mypy happy that a current span and trace exists
        current_span = get_current_span_data()
        current_trace = get_current_trace_data()
        assert current_span is not None
        assert current_trace is not None

        for validation in result.validations:
            guardrail_batch_item_message = GuardrailBatchItemMessage(
                project_name=self._client._project_name,
                entity_id=current_trace.id,
                secondary_id=current_span.id,
                name=validation.type,
                result="passed" if validation.validation_passed else "failed",
                config=validation.validation_config,
                details=validation.validation_details,
            )
            batch.append(guardrail_batch_item_message)

        message = GuardrailBatchMessage(batch=batch)
        self._client._streamer.put(message)

        return result

    def _parse_result(
        self, result: schemas.ValidationResponse
    ) -> schemas.ValidationResponse:
        if not result.validation_passed:
            failed_validations = []
            for validation in result.validations:
                if not validation.validation_passed:
                    failed_validations.append(validation)

            raise exceptions.GuardrailValidationFailed(
                "Guardrail validation failed",
                validation_results=result,
                failed_validations=failed_validations,
            )

        return result

    def validate_input(
        self, text: str, correlation_key: Optional[str] = None
    ) -> Union[schemas.ValidationResponse, schemas.PointGuardAiValidationResponse]:
        """
        Validate input text against all configured guardrails.

        This method validates text before it is sent to an LLM. For PointGuardAi guards,
        it calls the PointGuardAi input validation endpoint. For other guards, it uses
        the standard validation flow.

        Args:
            text: The input text to validate
            correlation_key: Optional tag/identifier for this request (PointGuardAi only)

        Returns:
            ValidationResponse or PointGuardAiValidationResponse containing validation results

        Raises:
            opik.exceptions.GuardrailValidationFailed: If validation fails
            httpx.HTTPStatusError: If the API returns an error status code

        Example:
            ```python
            from opik.guardrails import Guardrail, PointGuardAi

            guard = Guardrail(guards=[
                PointGuardAi(policy_name="my-policy")
            ])

            # Validate user input before sending to LLM
            result = guard.validate_input("What is the capital of France?")
            ```
        """
        # If we have PointGuardAi guards, use their input validation
        if self._pointguardai_guards:
            result: schemas.PointGuardAiValidationResponse = self._validate_input(
                generation=text, correlation_key=correlation_key
            )  # type: ignore
            return self._parse_result(result)

        # Fall back to standard validation for non-PointGuardAi guards
        return self.validate(text)

    @GUARDRAIL_DECORATOR.track
    def _validate_input(self, generation: str, correlation_key: Optional[str] = None) -> schemas.PointGuardAiValidationResponse:
        """Internal method for PointGuardAi input validation with automatic span tracking."""
        all_validations: List[schemas.ValidationResult] = []
        all_passed = True
        last_details = None

        for pg_guard in self._pointguardai_guards:
            client = self._pointguardai_clients[pg_guard.base_url]
            result = client.validate_input(generation, pg_guard.policy_name, correlation_key)

            all_validations.extend(result.validations)
            if not result.validation_passed:
                all_passed = False
            last_details = result.details

        combined_result = schemas.PointGuardAiValidationResponse(
            validation_passed=all_passed,
            validations=all_validations,
            details=last_details,
        )
        
        # Set guardrail result for tracking
        if not all_passed:
            combined_result.guardrail_result = "failed"
        else:
            combined_result.guardrail_result = "passed"

        return combined_result

    def validate_output(
        self, input_text: str, output_text: str, correlation_key: Optional[str] = None
    ) -> Union[schemas.ValidationResponse, schemas.PointGuardAiValidationResponse]:
        """
        Validate output text against all configured guardrails.

        This method validates text after it is received from an LLM. For PointGuardAi guards,
        it calls the PointGuardAi output validation endpoint with both the original input
        and the LLM's output. For other guards, it uses the standard validation flow
        on the output text.

        Args:
            input_text: The original input text
            output_text: The LLM output text to validate
            correlation_key: Optional tag/identifier for this request (PointGuardAi only)

        Returns:
            ValidationResponse or PointGuardAiValidationResponse containing validation results

        Raises:
            opik.exceptions.GuardrailValidationFailed: If validation fails
            httpx.HTTPStatusError: If the API returns an error status code

        Example:
            ```python
            from opik.guardrails import Guardrail, PointGuardAi

            guard = Guardrail(guards=[
                PointGuardAi(policy_name="my-policy")
            ])

            # Validate LLM output
            result = guard.validate_output(
                input_text="Tell me about security",
                output_text="Here are some security best practices...",
            )
            ```
        """
        # If we have PointGuardAi guards, use their output validation
        if self._pointguardai_guards:
            result: schemas.PointGuardAiValidationResponse = self._validate_output(
                generation=output_text, input_text=input_text, correlation_key=correlation_key
            )  # type: ignore
            return self._parse_result(result)

        # Fall back to standard validation for non-PointGuardAi guards
        return self.validate(output_text)
    
    def validate_and_get_input(self, text: str, correlation_key: Optional[str] = None) -> str:
        """
        Validate input text and return the validated content in one step.
        
        This is a convenience method that combines validate_input() and get_validated_content().
        Use this when you just need the safe content to pass to your LLM.
        
        Args:
            text: The input text to validate
            correlation_key: Optional tag/identifier for this request (PointGuardAi only)
            
        Returns:
            str: The validated content (modified if PII was redacted, otherwise original)
            
        Raises:
            opik.exceptions.GuardrailValidationFailed: If validation fails (content blocked)
            
        Example:
            ```python
            # Simple one-liner
            safe_input = guard.validate_and_get_input("My SSN is 123-45-6789")
            response = llm.chat(safe_input)
            ```
        """
        result = self.validate_input(text, correlation_key)
        return result.get_validated_content(text)
    
    def validate_and_get_output(self, input_text: str, output_text: str, correlation_key: Optional[str] = None) -> str:
        """
        Validate output text and return the validated content in one step.
        
        This is a convenience method that combines validate_output() and get_validated_content().
        Use this when you just need the safe content to return to your user.
        
        Args:
            input_text: The original input text
            output_text: The LLM output text to validate
            correlation_key: Optional tag/identifier for this request (PointGuardAi only)
            
        Returns:
            str: The validated content (modified if PII was redacted, otherwise original)
            
        Raises:
            opik.exceptions.GuardrailValidationFailed: If validation fails (content blocked)
            
        Example:
            ```python
            # Simple one-liner
            safe_output = guard.validate_and_get_output(user_query, llm_response)
            return safe_output
            ```
        """
        result = self.validate_output(input_text, output_text, correlation_key)
        return result.get_validated_content(output_text)

    @GUARDRAIL_DECORATOR.track
    def _validate_output(self, generation: str, input_text: str, correlation_key: Optional[str] = None) -> schemas.PointGuardAiValidationResponse:
        """Internal method for PointGuardAi output validation with automatic span tracking."""
        all_validations: List[schemas.ValidationResult] = []
        all_passed = True
        last_details = None

        for pg_guard in self._pointguardai_guards:
            client = self._pointguardai_clients[pg_guard.base_url]
            result = client.validate_output(input_text, generation, pg_guard.policy_name, correlation_key)

            all_validations.extend(result.validations)
            if not result.validation_passed:
                all_passed = False
            last_details = result.details

        combined_result = schemas.PointGuardAiValidationResponse(
            validation_passed=all_passed,
            validations=all_validations,
            details=last_details,
        )
        
        # Set guardrail result for tracking
        if not all_passed:
            combined_result.guardrail_result = "failed"
        else:
            combined_result.guardrail_result = "passed"

        return combined_result
