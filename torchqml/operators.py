"""
Pauli演算子とハミルトニアン

期待値計算のための観測量を定義
"""

from typing import List, Tuple, Union
from numbers import Number


class PauliOperator:
    """
    Pauli演算子（ハミルトニアン）
    
    内部表現: List[Tuple[coefficient, List[Tuple[pauli_type, qubit]]]]
    例: 0.5 * Z(0) @ X(1) = [(0.5, [('Z', 0), ('X', 1)])]
    """
    
    def __init__(self, terms: List[Tuple[float, List[Tuple[str, int]]]] = None):
        self.terms = terms if terms is not None else []
    
    @classmethod
    def single(cls, pauli_type: str, qubit: int) -> 'PauliOperator':
        """単一Pauli演算子"""
        return cls([(1.0, [(pauli_type, qubit)])])
    
    @classmethod
    def identity(cls) -> 'PauliOperator':
        """Identity演算子"""
        return cls([(1.0, [])])
    
    def to_observable(self) -> List[Tuple[float, List[Tuple[str, int]]]]:
        """backend.pyで使用する形式に変換"""
        return self.terms
    
    # ==================== 演算子オーバーロード ====================
    
    def __add__(self, other: Union['PauliOperator', Number]) -> 'PauliOperator':
        """加算"""
        if isinstance(other, (int, float)):
            return PauliOperator(self.terms + [(float(other), [])])
        elif isinstance(other, PauliOperator):
            return PauliOperator(self.terms + other.terms)
        raise TypeError(f"Cannot add PauliOperator with {type(other)}")
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other: Union['PauliOperator', Number]) -> 'PauliOperator':
        """減算"""
        if isinstance(other, (int, float)):
            return self + (-other)
        elif isinstance(other, PauliOperator):
            return self + (-1.0 * other)
        raise TypeError(f"Cannot subtract {type(other)} from PauliOperator")
    
    def __rsub__(self, other):
        return (-1.0 * self) + other
    
    def __mul__(self, scalar: Number) -> 'PauliOperator':
        """スカラー乗算"""
        return PauliOperator([(coeff * scalar, ops) for coeff, ops in self.terms])
    
    def __rmul__(self, scalar: Number) -> 'PauliOperator':
        return self.__mul__(scalar)
    
    def __neg__(self) -> 'PauliOperator':
        """符号反転"""
        return -1.0 * self
    
    def __matmul__(self, other: 'PauliOperator') -> 'PauliOperator':
        """テンソル積 (Z(0) @ X(1))"""
        if not isinstance(other, PauliOperator):
            raise TypeError(f"Cannot compute tensor product with {type(other)}")
        
        new_terms = []
        for c1, ops1 in self.terms:
            for c2, ops2 in other.terms:
                new_terms.append((c1 * c2, ops1 + ops2))
        return PauliOperator(new_terms)
    
    def __repr__(self) -> str:
        if not self.terms:
            return "0"
        
        parts = []
        for coeff, ops in self.terms:
            if not ops:
                parts.append(f"{coeff}")
            else:
                ops_str = " @ ".join(f"{p}({q})" for p, q in ops)
                if coeff == 1.0:
                    parts.append(ops_str)
                elif coeff == -1.0:
                    parts.append(f"-{ops_str}")
                else:
                    parts.append(f"{coeff}*{ops_str}")
        
        return " + ".join(parts)


# ファクトリー関数
def X(qubit: int) -> PauliOperator:
    """Pauli-X"""
    return PauliOperator.single('X', qubit)

def Y(qubit: int) -> PauliOperator:
    """Pauli-Y"""
    return PauliOperator.single('Y', qubit)

def Z(qubit: int) -> PauliOperator:
    """Pauli-Z"""
    return PauliOperator.single('Z', qubit)

def I(qubit: int = 0) -> PauliOperator:
    """Identity"""
    return PauliOperator.identity()
