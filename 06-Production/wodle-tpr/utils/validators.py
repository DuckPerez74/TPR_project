"""
Centralized validation utilities for TPR Anomaly Detection.

This module provides reusable validators for common data validation
scenarios across the project, ensuring consistency and DRY principles.
"""
import re
from typing import Optional


class EntityValidator:
    """Validates entity IDs according to project rules."""

    # Alphanumeric, underscore, hyphen, and dot allowed
    ENTITY_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]+$')
    MAX_ENTITY_ID_LENGTH = 256

    @classmethod
    def is_valid(cls, entity_id: str) -> bool:
        """
        Check if entity_id is valid.

        Args:
            entity_id: Entity identifier to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(entity_id, str):
            return False
        if len(entity_id) == 0 or len(entity_id) > cls.MAX_ENTITY_ID_LENGTH:
            return False
        if not cls.ENTITY_ID_PATTERN.match(entity_id):
            return False
        return True

    @classmethod
    def sanitize_for_filename(cls, entity_id: str) -> str:
        """
        Sanitize entity_id for safe use in filenames.

        Args:
            entity_id: Entity identifier

        Returns:
            Sanitized string safe for filenames
        """
        # Replace non-alphanumeric with underscore
        return "".join([c if c.isalnum() else "_" for c in entity_id])


class UserValidator:
    """Validates user IDs for L2 metrics."""

    # Similar to entity validation but may have different rules
    USER_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.@]+$')
    MAX_USER_ID_LENGTH = 512

    @classmethod
    def is_valid(cls, user_id: str) -> bool:
        """
        Check if user_id is valid.

        Args:
            user_id: User identifier to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(user_id, str):
            return False
        if len(user_id) == 0 or len(user_id) > cls.MAX_USER_ID_LENGTH:
            return False
        if not cls.USER_ID_PATTERN.match(user_id):
            return False
        return True

    @classmethod
    def sanitize_for_filename(cls, user_id: str) -> str:
        """
        Sanitize user_id for safe use in filenames.

        Args:
            user_id: User identifier

        Returns:
            Sanitized string safe for filenames
        """
        return "".join([c if c.isalnum() else "_" for c in user_id])


class MetricsValidator:
    """Validates metrics data."""

    @staticmethod
    def is_valid_window(window: int) -> bool:
        """
        Check if observation window is valid.

        Args:
            window: Window size in minutes

        Returns:
            True if valid, False otherwise
        """
        return window in [10, 30, 60]

    @staticmethod
    def is_valid_layer(layer: str) -> bool:
        """
        Check if detection layer is valid.

        Args:
            layer: Layer identifier

        Returns:
            True if valid, False otherwise
        """
        return layer in ['L1', 'L2']

    @staticmethod
    def is_valid_dimension(dimension: str) -> bool:
        """
        Check if L2 dimension is valid.

        Args:
            dimension: Dimension identifier

        Returns:
            True if valid, False otherwise
        """
        from constants import L2_SUPPORTED_DIMENSIONS
        return dimension in L2_SUPPORTED_DIMENSIONS


# Convenience functions for backward compatibility
def is_valid_entity_id(entity_id: str) -> bool:
    """Convenience function for entity validation."""
    return EntityValidator.is_valid(entity_id)


def sanitize_entity_id(entity_id: str) -> str:
    """Convenience function for entity sanitization."""
    return EntityValidator.sanitize_for_filename(entity_id)


def is_valid_user_id(user_id: str) -> bool:
    """Convenience function for user validation."""
    return UserValidator.is_valid(user_id)


def sanitize_user_id(user_id: str) -> str:
    """Convenience function for user sanitization."""
    return UserValidator.sanitize_for_filename(user_id)
