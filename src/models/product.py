"""Product model for database representation."""

from dataclasses import dataclass


@dataclass
class Product:
    """Product data model representing a product record."""

    id: int
    name: str
    description: str
    technical_specs: str
    manufacturer: str
