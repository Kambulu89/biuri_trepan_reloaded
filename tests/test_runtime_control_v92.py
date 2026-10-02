import time, pytest
from core.runtime_control import Deadline, check_interruption

def test_deadline_interrupts_within_tolerance():
    d=Deadline(.02); time.sleep(.03)
    with pytest.raises(TimeoutError): check_interruption(deadline=d)

def test_cancel_interrupts_immediately():
    with pytest.raises(InterruptedError,match='cancelado'): check_interruption(cancel_fn=lambda:True)
