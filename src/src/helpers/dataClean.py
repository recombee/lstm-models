def reduction(users, items, logger, pruning=3):
    """
    Reduction the data matrix to minimum interaction in both direction.
    """
    logger.info("Stating reduction: {} users, {} items".format(len(users), len(items)))
    stable = False
    steps = 0
    while not stable:
        stable = True
        users_to_delete = []
        for (user_id, entries) in users.items():
            for key in [item_id for item_id in entries if item_id not in items]:
                del entries[key]
            if len(entries) < pruning:
                users_to_delete.append(user_id)
                stable = False
        for user_id in users_to_delete:
            del users[user_id]
        items_to_delete = []
        for (item_id, entries) in items.items():
            for key in [user_id for user_id in entries if user_id not in users]:
                del entries[key]
            if len(entries) < pruning:
                items_to_delete.append(item_id)
                stable = False
        for item_id in items_to_delete:
            del items[item_id]
        steps += 1
        if stable:
            logger.info("Dataset stable after {} reduction steps ({} users, {} items)".format(steps, len(users),
                                                                                              len(items)))
        else:
            logger.info("Reduction step #{}: {} users, {} items".format(steps, len(users), len(items)))
    return users, items


def remove_robots(users, items, logger, threshold=0.25):
    """
    Function for remove robots from dataset. Threshold is in percent how much items have to user saw to be robot.
    """

    cnt = 0
    to_delete_users = []
    for user in users:
        if len(users[user]) > threshold * len(items):
            logger.info("Remove robot {} with {} interactions.".format(user, users[user]))
            for item in users[user]:
                del items[item][user]
            to_delete_users.append(user)
            cnt += 1
    for user in to_delete_users:
        del users[user]
    logger.info("Remove {} robot users.".format(cnt))
    return users, items
