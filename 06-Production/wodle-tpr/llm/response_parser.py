import json
import re
import logging


class ResponseParser:
    """
    Parses LLM responses into structured format.
    Handles JSON extraction and validation.
    """

    DEFAULT_RESPONSE = {
        'classification': 'Unknown',
        'threat_type': 'None',
        'user_operations': [],
        'explanation': 'Unable to parse LLM response',
        'recommended_actions': [],
        'confidence': 'Low',
        'parse_error': True
    }

    def __init__(self):
        self.logger = logging.getLogger('wodle-tpr.llm.parser')

    def parse(self, llm_response: str) -> dict:
        """
        Parse LLM response text into structured dictionary.

        Args:
            llm_response: Raw text response from LLM

        Returns:
            Parsed and validated response dictionary
        """
        if not llm_response:
            self.logger.warning("Empty LLM response received")
            return self.DEFAULT_RESPONSE.copy()

        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    self.logger.warning("No JSON structure found in LLM response")
                    return self._create_fallback_response(llm_response)

            parsed = json.loads(json_str)
            return self._validate_and_normalize(parsed)

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            return self._create_fallback_response(llm_response)
        except Exception as e:
            self.logger.error(f"Unexpected parsing error: {e}")
            return self._create_fallback_response(llm_response)

    def _validate_and_normalize(self, parsed: dict) -> dict:
        """Validate and normalize parsed response."""
        valid_classifications = ['Normal Activity', 'Suspicious', 'Likely Attack', 'False Positive']
        valid_threat_types = ['None', 'Brute Force', 'Data Exfiltration', 
                             'Privilege Escalation', 'DoS', 'Reconnaissance', 'Other']
        valid_confidence = ['Low', 'Medium', 'High']

        result = {
            'classification': parsed.get('classification', 'Unknown'),
            'threat_type': parsed.get('threat_type', 'None'),
            'user_operations': parsed.get('user_operations', []),
            'explanation': parsed.get('explanation', ''),
            'recommended_actions': parsed.get('recommended_actions', []),
            'confidence': parsed.get('confidence', 'Low'),
            'parse_error': False
        }

        if result['classification'] not in valid_classifications:
            result['classification'] = 'Unknown'

        if result['threat_type'] not in valid_threat_types:
            result['threat_type'] = 'Other'

        if result['confidence'] not in valid_confidence:
            result['confidence'] = 'Low'

        if not isinstance(result['user_operations'], list):
            result['user_operations'] = [str(result['user_operations'])]

        if not isinstance(result['recommended_actions'], list):
            result['recommended_actions'] = [str(result['recommended_actions'])]

        return result

    def _create_fallback_response(self, raw_response: str) -> dict:
        """Create fallback response when JSON parsing fails."""
        response = self.DEFAULT_RESPONSE.copy()

        # Store full response but truncate for logging
        max_length = 2000
        if len(raw_response) > max_length:
            truncated = raw_response[:max_length] + f"... (truncated {len(raw_response) - max_length} chars)"
            response['explanation'] = f"Raw LLM response (parsing failed): {truncated}"
            self.logger.debug(f"Full unparseable response: {raw_response}")
        else:
            response['explanation'] = f"Raw LLM response (parsing failed): {raw_response}"

        response['raw_response'] = raw_response
        return response

    def format_for_log(self, parsed_response: dict) -> str:
        """
        Format parsed response for logging.

        Args:
            parsed_response: Validated response dictionary

        Returns:
            Formatted string for log output
        """
        lines = [
            f"Classification: {parsed_response.get('classification', 'Unknown')}",
            f"Threat Type: {parsed_response.get('threat_type', 'None')}",
            f"Confidence: {parsed_response.get('confidence', 'Low')}",
            "",
            "User Operations:"
        ]

        for op in parsed_response.get('user_operations', []):
            lines.append(f"  - {op}")

        lines.append("")
        lines.append(f"Explanation: {parsed_response.get('explanation', '')}")
        lines.append("")
        lines.append("Recommended Actions:")

        for action in parsed_response.get('recommended_actions', []):
            lines.append(f"  - {action}")

        return "\n".join(lines)
