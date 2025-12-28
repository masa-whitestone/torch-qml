"""Two-qubit gates"""

from ..core.qvector import QubitRef, QVector


def cx(control: QubitRef, target: QubitRef):
    """CNOT (Controlled-X) gate"""
    qvector = control.qvector
    assert target.qvector is qvector, "Control and target must be from same qvector"
    
    op = {
        "gate": "x",
        "targets": [target.index],
        "params": None,
        "controls": [control.index],
        "adjoint": False,
    }
    qvector._add_operation(op)


def cy(control: QubitRef, target: QubitRef):
    """Controlled-Y gate"""
    qvector = control.qvector
    assert target.qvector is qvector
    
    op = {
        "gate": "y",
        "targets": [target.index],
        "params": None,
        "controls": [control.index],
        "adjoint": False,
    }
    qvector._add_operation(op)


def cz(control: QubitRef, target: QubitRef):
    """Controlled-Z gate"""
    qvector = control.qvector
    assert target.qvector is qvector
    
    op = {
        "gate": "z",
        "targets": [target.index],
        "params": None,
        "controls": [control.index],
        "adjoint": False,
    }
    qvector._add_operation(op)


def swap(qubit1: QubitRef, qubit2: QubitRef):
    """SWAP gate"""
    qvector = qubit1.qvector
    assert qubit2.qvector is qvector
    
    op = {
        "gate": "swap",
        "targets": [qubit1.index, qubit2.index],
        "params": None,
        "controls": None,
        "adjoint": False,
    }
    qvector._add_operation(op)
