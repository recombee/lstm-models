from ..helpers.helpFunctions import PercentCounter, Timer


def evaluate_model(csv_writer, model, data_storage, evaluation_config, batch, logger):
    """
    Evaluate model on test set of users and save the results.
    """

    good = {}
    cnt = {}
    items = {}
    repeat = {}

    per_print = PercentCounter(data_storage.get_n_test_users(), 0.05, logger)
    Timer.restart("rec")

    if batch:
        test_users, hide_items = [], []
        for test_user, hide_item in data_storage.get_test_tuples():
            test_users.append(test_user)
            hide_items.append(hide_item)

        recommendation = model.get_n_recommendation_batch(test_users, evaluation_config['top-n'], None)

        for i, user_recomm in enumerate(recommendation):

            good, cnt, items, repeat = save_results_to_dicts(user_recomm=user_recomm, good=good, cnt=cnt,
                                                             repeat=repeat, items=items, test_user=test_users[i],
                                                             hide_item=hide_items[i]['item_id'], per_print=per_print)

    else:
        for test_user, hide_item in data_storage.get_test_tuples():
            recommendation = model.get_n_recommendation(test_user, evaluation_config['top-n'], None)

            good, cnt, items, repeat = save_results_to_dicts(user_recomm=recommendation, good=good, cnt=cnt,
                                                             repeat=repeat, items=items, test_user=test_user,
                                                             hide_item=hide_item['item_id'], per_print=per_print)

    write_results_to_csv(csv_writer, good, cnt, items, data_storage.n_train_columns(), repeat, logger)


def save_results_to_dicts(user_recomm, good, cnt, repeat, items, test_user, hide_item, per_print):
    """
    Save results from one recommendation. Doing average measurement.
    """

    for parametrization in user_recomm:
        if parametrization not in cnt:
            good[parametrization] = 0
            cnt[parametrization] = 0
            repeat[parametrization] = 0
            items[parametrization] = set()
        rec = set(user_recomm[parametrization])
        if hide_item in rec:
            good[parametrization] += 1
        repeat[parametrization] += len(set([x['item_id'] for x in test_user[:-1]]).intersection(set(user_recomm[parametrization])))
        cnt[parametrization] += 1
        items[parametrization].update(rec)
    per_print.increment("In {} sec evaluate % of test users.".format(Timer.get_time_second("rec")))
    return good, cnt, items, repeat


def write_results_to_csv(csv_writer, good, cnt, item, num_items, repeat, logger):
    """
    Write results of evaluation to file. Compute some metric from data.
    """

    for parametrization in good:
        par = split_parametrization(parametrization)
        par["recall"] = good[parametrization]/cnt[parametrization]
        par["coverage"] = len(item[parametrization])/num_items
        par["recommitems"] = len(item[parametrization])
        par["repeatitems"] = repeat[parametrization]
        par["cnt"] = cnt[parametrization]
        logger.info("Parametrization: {}".format(parametrization))
        logger.info("Recall: {}".format(par["recall"]))
        logger.info("Coverage: {}".format(par["coverage"]))
        csv_writer.append_line(**par)


def split_parametrization(string):
    return {x.split("=")[0]: x.split("=")[1] for x in string.split(",")}
