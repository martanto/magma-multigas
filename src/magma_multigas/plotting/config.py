"""Plot configuration for magma-multigas v2.0."""

from dataclasses import dataclass, field
from typing import Dict, Optional

from ..config.variables import plot_properties


@dataclass
class PlotConfig:
    """Configuration for plotting with intelligent defaults.

    Attributes:
        width: Figure width in inches
        height: Figure height in inches
        dpi: Dots per inch for figure resolution
        style: Seaborn style (whitegrid, darkgrid, white, dark, ticks)
        context: Seaborn context (paper, notebook, talk, poster)
        font_scale: Scale factor for fonts
        color_palette: Color palette name or list of colors
        figsize: Tuple of (width, height) - computed from width and height
    """

    width: int = 12
    height: int = 4
    dpi: int = 300
    style: str = "whitegrid"
    context: str = "notebook"
    font_scale: float = 1.0
    color_palette: str = "deep"
    title_fontsize: int = 14
    label_fontsize: int = 12
    tick_fontsize: int = 10
    legend_fontsize: int = 10

    # Column properties from variables.py
    column_properties: Dict[str, Dict[str, str]] = field(
        default_factory=lambda: plot_properties.copy()
    )

    @property
    def figsize(self) -> tuple[int, int]:
        """Get figure size as tuple.

        Returns:
            Tuple of (width, height)
        """
        return (self.width, self.height)

    def get_column_property(
        self, column: str, property_name: str, default: Optional[str] = None
    ) -> Optional[str]:
        """Get property for a column.

        Args:
            column: Column name
            property_name: Property name (label, color, marker)
            default: Default value if not found

        Returns:
            Property value or default
        """
        if column in self.column_properties:
            return self.column_properties[column].get(property_name, default)
        return default

    def get_column_label(self, column: str) -> str:
        """Get display label for column.

        Args:
            column: Column name

        Returns:
            Display label (or column name if not configured)
        """
        return self.get_column_property(column, "label", column)

    def get_column_color(self, column: str, default: str = "#1f77b4") -> str:
        """Get color for column.

        Args:
            column: Column name
            default: Default color if not configured

        Returns:
            Color hex code
        """
        return self.get_column_property(column, "color", default)

    def get_column_marker(self, column: str, default: str = "o") -> str:
        """Get marker style for column.

        Args:
            column: Column name
            default: Default marker if not configured

        Returns:
            Marker style
        """
        return self.get_column_property(column, "marker", default)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PlotConfig":
        """Create PlotConfig from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            PlotConfig instance
        """
        return cls(**config_dict)

    def copy(self, **kwargs) -> "PlotConfig":
        """Create copy with optional overrides.

        Args:
            **kwargs: Properties to override

        Returns:
            New PlotConfig instance
        """
        config_dict = {
            "width": self.width,
            "height": self.height,
            "dpi": self.dpi,
            "style": self.style,
            "context": self.context,
            "font_scale": self.font_scale,
            "color_palette": self.color_palette,
            "title_fontsize": self.title_fontsize,
            "label_fontsize": self.label_fontsize,
            "tick_fontsize": self.tick_fontsize,
            "legend_fontsize": self.legend_fontsize,
            "column_properties": self.column_properties.copy(),
        }
        config_dict.update(kwargs)
        return PlotConfig(**config_dict)
