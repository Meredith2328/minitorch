from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # n元函数f的求导: df/dx_i = (f(x1, ..., x_i + e, ..., x_n-1) - f(x1, ..., x_i, ..., x_n-1)) / e
    # 中心差分所以是 (f(x + h) - f(x - h)) / 2h
    valsAdd: Tuple = ()
    valsSub: Tuple = ()
    for i in range(len(vals)):
        if i == arg:
            valsAdd += (vals[i] + epsilon,)
            valsSub += (vals[i] - epsilon,)
        else:
            valsAdd += (vals[i],)
            valsSub += (vals[i],)
    assert len(valsAdd) == len(vals) and len(valsSub) == len(vals)
    # 这里解包*才是传入n个参数, 否则是传入一个元组
    return (f(*valsAdd) - f(*valsSub)) / (2 * epsilon)



variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visited = set()
    order = []

    def dfs(node: Variable):
        # 按说这里用Tensor才能用.history, 但这样能过用例
        if node.unique_id in visited or node.history is None:
            return
        visited.add(node.unique_id)
        # breakpoint() # DEBUG
        for parent in node.parents:
            dfs(parent)
        order.append(node) # 这里生成正拓扑排序
        
    dfs(variable)
    return order[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    sorted_vars_order: Iterable[Variable] = topological_sort(variable)
    deriv_dict = {variable.unique_id: deriv} # Var -> derivative, 默认传入了1.0, 注意unhashable所以要用unique_id
    for var in sorted_vars_order:
        d_out = deriv_dict.get(var.unique_id, None)
        if (var.is_leaf()):
            var.accumulate_derivative(d_out) # 可以从单节点图来得到上面这段逻辑
        else:
            for parent, d_parent in var.chain_rule(d_out):
                if (parent.unique_id not in deriv_dict):
                    deriv_dict[parent.unique_id] = d_parent
                else:
                    deriv_dict[parent.unique_id] += d_parent
            




@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
