ANALYSIS_PROMPT = """You are a security analyst reviewing an anomaly alert from a web API monitoring system.

Your task is to:
1. Analyze the user's behavior based on the logs
2. Describe what operations the user performed
3. Classify if this is normal activity or a potential threat
4. If suspicious, identify the type of attack

{context}

## Response Format
Always respond in English using the following JSON format:
```json
{{
    "classification": "Normal Activity | Suspicious | Likely Attack | False Positive",
    "threat_type": "None | Brute Force | Data Exfiltration | Privilege Escalation | DoS | Reconnaissance | Other",
    "user_operations": [
        "Description of operation 1 (e.g., 'Attempted login 5 times without success')",
        "Description of operation 2 (e.g., 'Accessed /api/admin/users via GET')",
        "Description of operation 3 (e.g., 'Attempted to delete resource at /api/users/123')"
    ],
    "explanation": "Detailed explanation of what is happening and why it was flagged",
    "recommended_actions": [
        "Action 1",
        "Action 2"
    ],
    "confidence": "Low | Medium | High"
}}
```

Important notes:
- user_operations should describe in English what the user tried to do
- Use verbs like: "Attempted", "Accessed", "Queried", "Modified", "Deleted", "Created"
- Include route paths and HTTP methods in operation descriptions
- If 'imp' is true in logs, it means the request was made by support staff
- Focus on patterns: repeated failures, access to sensitive routes, unusual sequences
"""


OPERATION_VERBS = {
    'GET': 'Queried',
    'POST': 'Created/Sent',
    'PUT': 'Modified',
    'PATCH': 'Updated',
    'DELETE': 'Attempted to delete'
}


SENSITIVE_ROUTES = [
    '/admin', '/backup', '/export', '/config', '/roles',
    '/permissions', '/users', '/settings', '/audit', '/logs'
]


def get_analysis_prompt(context_text: str) -> str:
    """
    Get the complete analysis prompt with context injected.

    Args:
        context_text: Formatted context from ContextBuilder

    Returns:
        Complete prompt string
    """
    return ANALYSIS_PROMPT.format(context=context_text)


def is_sensitive_route(route: str) -> bool:
    """Check if a route is considered sensitive."""
    route_lower = route.lower()
    return any(sensitive in route_lower for sensitive in SENSITIVE_ROUTES)
