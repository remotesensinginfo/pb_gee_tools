import re
from typing import List

def convert_xgboost_model_to_r_tree_format(xgb_model) -> List[str]:
    """
    A function which takes an XGBoost model (xgboost.Booster()) and converts to the
    r tree format which ee.Classifier.decisionTreeEnsemble takes as input.

    :param xgb_model: a xgboost.Booster() object
    :return: a list of tree structures
    """
    def _convert_tree_format(input_string):
        nodes = {}
        root_id = None

        # 1. Parse the input string into a dictionary
        for line in input_string.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            # Match a split node: e.g., "0:[f4<-0.494357675] yes=1,no=2,missing=2"
            m_split = re.match(r"(\d+):\[(.*?)\] yes=(\d+),no=(\d+)", line)
            if m_split:
                node_id, cond, left_id, right_id = m_split.groups()

                # Extract feature, operator, and value
                m_cond = re.match(r"([A-Za-z0-9_]+)([<>=]+)(.*)", cond)
                if m_cond:
                    feature, op, val = m_cond.groups()
                    nodes[node_id] = {
                        "type": "split",
                        "feature": feature,
                        "op": op,
                        "val": val,
                        "left": left_id,
                        "right": right_id,
                    }
                    if root_id is None:
                        root_id = node_id
            else:
                # Match a leaf node: e.g., "15:leaf=0.199297011"
                m_leaf = re.match(r"(\d+):leaf=(.*)", line)
                if m_leaf:
                    node_id, val = m_leaf.groups()
                    nodes[node_id] = {"type": "leaf", "val": val}
                    if root_id is None:
                        root_id = node_id

        # 2. Setup the output header (Raw strings, no extra quotes or alignment spaces)
        output_lines = [
            "n= 2",
            "node), split, n, loss, yval, (yprob)",
            "* denotes terminal node",
        ]

        # 3. Recursive traversal to build the rpart format
        def traverse(node_id, out_id, depth, is_left, parent_cond_str):
            if node_id not in nodes:
                return

            node = nodes[node_id]
            indent = " " * depth

            # Default values for rpart format, based on if it's a leaf or split
            cond_str = parent_cond_str
            n = "1"
            loss = "0.0000"
            yprob = (
                "(0.0 1.0)" if is_left else "(1.0 0.0)"
            )  # default for non-root branches
            leaf_str = ""
            yval = ""

            if node["type"] == "leaf":
                yval = node["val"]
                leaf_str = " *"
            else:  # Split node
                yval = "1" if is_left else "0"  # Default for non-root split branches

            # Special handling for the root node (out_id == 1)
            if out_id == 1:
                cond_str = "root"
                n = "2"  # Hardcoded in the original snippet, matches rpart root N
                loss = "1.0000"  # Hardcoded in the original snippet, matches rpart root loss
                if node["type"] == "leaf":
                    # If root is a leaf, its yval and leaf_str are already set above.
                    # Its yprob should be what the example showed for a root leaf.
                    yprob = "(1.0 0.0)"
                else:  # Root is a split node
                    # Its yval is a placeholder for rpart, often 0 or average.
                    # The original code hardcoded yval as "0" for split root.
                    yval = "0"
                    yprob = (
                        "(0.5 0.5)"  # Hardcoded in the original snippet for split root
                    )

            # Build the formatted string line
            line_content = (
                f"{indent}{out_id}) {cond_str} {n} {loss} {yval} {yprob}{leaf_str}"
            )
            output_lines.append(line_content)

            # Traverse children if it's a split node
            if node["type"] == "split":
                feat, val = node["feature"], node["val"]
                # Left branch gets < , Right branch gets >=
                left_cond = f"{feat}<{val}"
                right_cond = f"{feat}>={val}"

                # Left child = out_id * 2 | Right child = (out_id * 2) + 1
                traverse(node["left"], out_id * 2, depth + 1, True, left_cond)
                traverse(node["right"], (out_id * 2) + 1, depth + 1, False, right_cond)

        # Start recursion at the root
        if root_id is not None:
            traverse(root_id, 1, 0, False, "")

        return "\n".join(output_lines) + "\n"

    tree_dump = xgb_model.get_dump()
    trees_lst = list()
    for td in tree_dump:
        trees_lst.append((_convert_tree_format(td)))
    return trees_lst
