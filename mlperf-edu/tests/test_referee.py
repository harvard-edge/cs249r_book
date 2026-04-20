import torch
from mlperf_edu.core import Referee

def test_referee_catches_early_stop():
    """Test that a student cannot stop the referee clock without reaching target accuracy."""
    referee = Referee("student_cheater", "vision-baseline", 0.90)
    referee.start_clock()
    
    # Student passes fake bad predictions
    preds = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
    targets = torch.tensor([1, 0]) # 0% accuracy
    
    acc = referee.evaluate_epoch(preds, targets)
    
    assert acc == 0.0
    assert referee.is_done() == False
    assert referee._end_time is None # Clock is still running!

def test_referee_validates_success():
    """Test that the referee successfully generates a receipt when target is hit."""
    referee = Referee("student_honest", "vision-baseline", 0.90)
    referee.start_clock()
    
    # Student passes 100% accurate predictions
    preds = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
    targets = torch.tensor([0, 1]) 
    
    acc = referee.evaluate_epoch(preds, targets)
    
    assert acc == 1.0
    assert referee.is_done() == True
    assert referee._result is not None
    assert referee._result.passed == True
