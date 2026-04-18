from typing import Optional

class TraceableConstant(float):
    """
    A numerical float that carries pedagogical context (citation, description)
    for the textbook. Functions exactly as a standard Python float in all
    mathematical operations, ensuring backwards compatibility with existing 
    physics formulas, while exposing metadata for educational rendering.
    """
    def __new__(cls, value, name: str, description: str, citation: str, url: Optional[str] = None):
        obj = super().__new__(cls, value)
        obj.name = name
        obj.description = description
        obj.citation = citation
        obj.url = url
        return obj

    def render_markdown(self) -> str:
        """Renders the assumption as a markdown block for textbook/lab integration."""
        lines = [
            f"**Assumption: {self.name}** = `{float(self)}`",
            "",
            f"_{self.description}_",
            "",
        ]
        if self.url:
            lines.append(f"> **Source:** [{self.citation}]({self.url})")
        else:
            lines.append(f"> **Source:** {self.citation}")
        return "\n".join(lines)
