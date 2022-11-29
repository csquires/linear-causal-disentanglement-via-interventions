


def argmax_dict(d: dict, one_choice=False):
    max_value = max(d.values())
    matching = {k for k, v in d.items() if v == max_value}
    if one_choice:
        k = list(matching)[0]
        return {k}
    return matching