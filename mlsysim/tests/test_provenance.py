import pytest
import pint
from mlsysim.core.provenance import TraceableConstant
from mlsysim.core.constants import ureg

def test_system_assumption_behaves_like_float():
    """Ensure the pedagogical wrapper doesn't break basic math operations."""
    assump = TraceableConstant(
        0.85, 
        name="Test MFU", 
        description="A test assumption.", 
        citation="Test Citation"
    )
    
    assert isinstance(assump, float)
    assert assump == 0.85
    assert assump * 2 == 1.7
    assert 1.0 - assump == pytest.approx(0.15)
    assert assump / 2 == 0.425
    assert assump > 0.80

def test_system_assumption_preserves_metadata():
    """Ensure the metadata is accessible for textbook rendering."""
    assump = TraceableConstant(
        0.50, 
        name="Test MFU", 
        description="A test assumption.", 
        citation="Test Citation",
        url="https://example.com"
    )
    
    assert assump.name == "Test MFU"
    assert assump.description == "A test assumption."
    assert assump.citation == "Test Citation"
    
    md = assump.render_markdown()
    assert "Test MFU" in md
    assert "https://example.com" in md
    assert "0.5" in md

def test_system_assumption_with_pint():
    """Ensure pint unit operations work correctly on the wrapper."""
    assump = TraceableConstant(
        15.0, 
        name="Test Overhead", 
        description="Overhead in ms.", 
        citation="Test Citation"
    )
    
    # Multiplying by a pint unit
    quant = assump * ureg.ms
    assert isinstance(quant, pint.Quantity)
    assert quant.magnitude == 15.0
    assert quant.units == ureg.ms
    
    # Division
    rate = 1.0 / quant
    assert rate.magnitude == pytest.approx(1/15.0)
