"""Readable grouping names. Raw records keep internal IDs (p01, p0110, ...)."""

THREE_OP_ALIASES = {"p00": "full", "p01": "first_pair", "p10": "last_pair", "p11": "separate"}
SHORT = {"add": "add", "sub": "sub", "relu": "relu", "matmul": "mm", "gelu": "gelu",
         "layer_norm": "ln", "mean": "mean", "softmax": "softmax"}


def grouping_name(partition, kinds=None):
    """separate/first_pair/last_pair/full for 3-op chains; otherwise e.g. 'add+ln_mm_add+gelu'."""
    if len(partition.cuts) == 2:
        return THREE_OP_ALIASES[partition.id]
    if all(bit == 0 for bit in partition.cuts):
        return "full"
    if all(bit == 1 for bit in partition.cuts):
        return "separate"
    groups, current = [], [kinds[0]]
    for bit, kind in zip(partition.cuts, kinds[1:]):
        if bit:
            groups.append(current)
            current = [kind]
        else:
            current.append(kind)
    groups.append(current)
    return "_".join("+".join(SHORT.get(k, k) for k in group) for group in groups)
