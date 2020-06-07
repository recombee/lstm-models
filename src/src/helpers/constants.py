BUY = 3
CLICK = 1
CART = 2
NONE = 0


def get_weight(it):
    """
    Return weight of interaction.
    """
    it = int(it)
    if it == CLICK:
        return 0.25
    if it == BUY:
        return 0.75
    return 0


def get_interaction_type(name):
    """
    Return normalized type of interaction.
    """
    if name == 'click':
        return CLICK
    if name == 'buy':
        return BUY
    if name == 'cart':
        return BUY
    return NONE
