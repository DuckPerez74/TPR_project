import re
from typing import Optional


class EntityValidator:
    """Validates entity IDs according to project rules."""

    # Alphanumeric, underscore, hyphen, and dot allowed
    ENTITY_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]+$')
    MAX_ENTITY_ID_LENGTH = 256

    @classmethod
    def is_valid(cls, entity_id: str) -> bool:
        if not isinstance(entity_id, str):
            return False
        if len(entity_id) == 0 or len(entity_id) > cls.MAX_ENTITY_ID_LENGTH:
            return False
        if not cls.ENTITY_ID_PATTERN.match(entity_id):
            return False
        return True

    @classmethod
    def sanitize_for_filename(cls, entity_id: str) -> str:
        # Replace non-alphanumeric with underscore
        return "".join([c if c.isalnum() else "_" for c in entity_id])


class UserValidator:
    """Validates user IDs for L2 metrics."""

    # Similar to entity validation but may have different rules
    USER_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.@]+$')
    MAX_USER_ID_LENGTH = 512

    @classmethod
    def is_valid(cls, user_id: str) -> bool:
        if not isinstance(user_id, str):
            return False
        if len(user_id) == 0 or len(user_id) > cls.MAX_USER_ID_LENGTH:
            return False
        if not cls.USER_ID_PATTERN.match(user_id):
            return False
        return True

    @classmethod
    def sanitize_for_filename(cls, user_id: str) -> str:
        return "".join([c if c.isalnum() else "_" for c in user_id])


class MetricsValidator:

    @staticmethod
    def is_valid_window(window: int) -> bool:
        return window in [10, 30, 60]

    @staticmethod
    def is_valid_layer(layer: str) -> bool:
        return layer in ['L1', 'L2']

    @staticmethod
    def is_valid_dimension(dimension: str) -> bool:
        from constants import L2_SUPPORTED_DIMENSIONS
        return dimension in L2_SUPPORTED_DIMENSIONS


# Convenience functions for backward compatibility
def is_valid_entity_id(entity_id: str) -> bool:
    return EntityValidator.is_valid(entity_id)


def sanitize_entity_id(entity_id: str) -> str:
    return EntityValidator.sanitize_for_filename(entity_id)


def is_valid_user_id(user_id: str) -> bool:
    return UserValidator.is_valid(user_id)


def sanitize_user_id(user_id: str) -> str:
    return UserValidator.sanitize_for_filename(user_id)
