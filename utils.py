import copy
import torch
import sympy
from sympy.parsing.sympy_parser import parse_expr


def is_solution_same(i1, i2, form_manager):
    """
    Check if the solutions represented by two mathematical expressions are the same.

    Args:
    - i1 (list): List of indices representing the first mathematical expression.
    - i2 (list): List of indices representing the second mathematical expression.
    - form_manager: A manager for handling mathematical expressions.

    Returns:
    - bool: True if solutions are the same, False otherwise.
    """
    # Convert indices to string expressions
    c1 = " ".join([form_manager.get_idx_symbol(x) for x in i1])
    c2 = " ".join([form_manager.get_idx_symbol(x) for x in i2])

    # Check for equality and presence of '=' in both expressions
    if ('=' not in c1) or ('=' not in c2):
        return False
    elif (form_manager.unk_token in c1) or (form_manager.unk_token in c2):
        return False
    else:
        try:
            # Parse and solve equations
            s1 = c1.split('=')
            s2 = c2.split('=')
            eq1 = []
            eq2 = []
            x = sympy.Symbol('x')
            eq1.append(parse_expr(s1[0]))
            eq1.append(parse_expr(s1[1]))
            eq2.append(parse_expr(s2[0]))
            eq2.append(parse_expr(s2[1]))
            res1 = sympy.solve(sympy.Eq(eq1[0], eq1[1]), x)
            res2 = sympy.solve(sympy.Eq(eq2[0], eq2[1]), x)

            # Check if solutions are the same
            if not res1 or not res2:
                return False
            if res1[0] == res2[0]:
                # print("Excution_true: ", c1, '\t', c2)
                pass
            return res1[0] == res2[0]

        except BaseException:
            # print("Excution_error: ", c1, '\t', c2)
            pass
            return False


def is_all_same(c1, c2, form_manager):
    """
    Check if two lists of expressions are entirely the same.

    Args:
    - c1 (list): List of indices representing the first expression.
    - c2 (list): List of indices representing the second expression.
    - form_manager: A manager for handling mathematical expressions.

    Returns:
    - bool: True if expressions are entirely the same, False otherwise.
    """
    all_same = False
    if len(c1) == len(c2):
        all_same = True
        for j in range(len(c1)):
            if c1[j] != c2[j]:
                all_same = False
                break
    if all_same == False:
        # If not entirely the same, check if solutions are the same
        if is_solution_same(c1, c2, form_manager):
            return True
        return False
    else:
        return True


def compute_accuracy(candidate_list, reference_list, form_manager):
    """
    Compute accuracy between lists of candidate and reference expressions.

    Args:
    - candidate_list (list): List of lists of indices representing candidate expressions.
    - reference_list (list): List of lists of indices representing reference expressions.
    - form_manager: A manager for handling mathematical expressions.

    Returns:
    - float: Accuracy between candidate and reference expressions.
    """
    if len(candidate_list) != len(reference_list):
        print("candidate list has length {}, reference list has length {}\n".format(
            len(candidate_list), len(reference_list)))
    len_min = min(len(candidate_list), len(reference_list))
    c = 0
    for i in range(len_min):
        # Check if expressions are the same
        if is_all_same(candidate_list[i], reference_list[i], form_manager):
            c = c+1
        else:
            pass
    return c/float(len_min)


def compute_tree_accuracy(candidate_list_, reference_list_, form_manager):
    """
    Compute tree accuracy between lists of candidate and reference expressions.

    Args:
    - candidate_list_ (list): List of lists of indices representing candidate expressions.
    - reference_list_ (list): List of lists of indices representing reference expressions.
    - form_manager: A manager for handling mathematical expressions.

    Returns:
    - float: Tree accuracy between candidate and reference expressions.
    """
    candidate_list = []
    for i in range(len(candidate_list_)):
        candidate_list.append(candidate_list_[i])
    reference_list = []
    for i in range(len(reference_list_)):
        reference_list.append(reference_list_[i])
    return compute_accuracy(candidate_list, reference_list, form_manager)


def prepare_ext_vocab(batch_graph, src_vocab, device):
    """
    Prepare an extended vocabulary for a batch of graphs.

    Args:
    - batch_graph: Batch of graphs.
    - src_vocab: Source vocabulary.
    - device: Torch device.

    Returns:
    - oov_dict: Extended vocabulary.
    """
    oov_dict = copy.deepcopy(src_vocab)
    token_matrix = []
    for n in batch_graph.node_attributes:
        node_token = n["token"]
        if (n.get("type") is None or n.get("type") == 0) and oov_dict.get_symbol_idx(
            node_token
        ) == oov_dict.get_symbol_idx(oov_dict.unk_token):
            oov_dict.add_symbol(node_token)
        token_matrix.append(oov_dict.get_symbol_idx(node_token))
    batch_graph.node_features["token_id_oov"] = torch.tensor(token_matrix, dtype=torch.long).to(
        device
    )
    return oov_dict
