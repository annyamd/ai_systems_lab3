import matplotlib.pyplot as plt
from math import log
import csv
import pprint
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def print_line():
    print("--------------------------------")


def print_help():
    print_line()
    print("Command list:\n"
          "tree -- make solution tree;\n"
          "test -- run on test data;\n"
          "val - valuate;"
          # "-----------------\n"
          )
    print_line()


def print_err():
    print("Incorrect command. To see the list of commands, type \"h\".")


def plot_graph():
    # nx.draw(g, with_labels=True)
    plt.show()


def draw_auc_roc(res):
    res = sorted(res, key=lambda d: d['pred'], reverse=True)
    x = [0]
    y = [0]
    xi = 0
    yi = 0

    print("\n\n----------------\n")
    print(res)

    for item in res:
        if item['real_ans'] == 'passed':
            yi += 1
        else:
            xi += 1
        x.append(xi)
        y.append(yi)

    x = [item / xi for item in x]
    y = [item / yi for item in y]

    print(x)
    print(y)

    # plt.figure()
    lw = 2
    plt.plot(
        x,
        y,
        color="darkorange",
        lw=lw
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.show()


def draw_auc_pr(res, y_test):
    res = sorted(res, key=lambda d: d['pred'], reverse=True)

    precision_scores = []
    recall_scores = []

    print("\n\n----------------\n")
    print(res)

    probability_thresholds = np.linspace(0, 1, num=100)

    for p in probability_thresholds:

        y_test_preds = []

        for r in res:
            if r['pred'] > p:
                y_test_preds.append('passed')
            else:
                y_test_preds.append('not passed')

        precision, recall = calc_precision_recall(y_test, y_test_preds)

        precision_scores.append(precision)
        recall_scores.append(recall)

    # plt.figure()
    lw = 2
    plt.plot(
        recall_scores,
        precision_scores,
        color="darkorange",
        lw=lw
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.show()


def calc_precision_recall(y_true, y_pred):
    # Instantiate counters
    TP = 0
    FP = 0
    FN = 0

    # Determine whether each prediction is TP, FP, TN, or FN
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] == 'passed':
            TP += 1
        if y_pred[i] == 'passed' and y_true[i] != y_pred[i]:
            FP += 1
        if y_pred[i] == 'not passed' and y_true[i] != y_pred[i]:
            FN += 1

    # Calculate true positive rate and false positive rate
    # Use try-except statements to avoid problem of dividing by 0
    try:
        precision = TP / (TP + FP)
    except:
        precision = 1

    try:
        recall = TP / (TP + FN)
    except:
        recall = 1

    return precision, recall


def apply_tree_on_list(tree, list_x, list_y):
    result_y = []
    i = 0
    for item in list_x:
        print('Elem: ' + str(item))
        res = apply_tree_on_elem(tree, item)
        res['real_ans'] = list_y[i]
        result_y.append(res)
        i += 1
    return result_y


def apply_tree_on_elem(tree, elem):
    cur_tree = tree
    while 'res' not in cur_tree:
        attr = list(cur_tree.keys())[0]
        values = cur_tree[attr]
        val = elem[attr]
        cur_tree = values[val]
    print(cur_tree)
    return cur_tree


def get_positive():
    return 'passed'


def get_values(results):
    tp, fp, tn, fn = 0, 0, 0, 0

    for res in results:
        if res['res'] == res['real_ans'] and res['real_ans'] == get_positive():
            tp += 1
        elif res['real_ans'] != res['res'] and res['real_ans'] == get_positive():
            fp += 1
        elif res['real_ans'] != res['res'] and res['real_ans'] != get_positive():
            fn += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)


def precision(tp, fp):
    return tp / (tp + fp)


def recall(tp, fn):
    return tp / (tp + fn)


def gain_ratio(dataset, attr_id):
    info = calc_info_val(dataset)
    info_x, split = calc_info_and_split_x_val(dataset, attr_id)
    if split == 0:
        return -1
    return (info - info_x) / split


def calc_info_val(data_set):
    info = 0.0
    n = len(data_set)

    freq = {'passed': 0, 'not passed': 0}
    for item in data_set:
        if item[5] == 'passed':
            freq['passed'] += 1
        else:
            freq['not passed'] += 1

    for cls in freq:
        # print("val: " + cls + " = " + str(freq[cls]/n))
        if freq[cls] == 0:
            continue
        info -= freq[cls] / n * log(freq[cls] / n, 2)

    # print('p: ' + str(freq['passed']))
    # print('np: ' + str(freq['not passed']))
    return info


def calc_info_and_split_x_val(dataset, attr_id):
    splited_ds = split_dataset(dataset, attr_id)
    info = 0.0
    n = len(dataset)

    for item in splited_ds:
        info += len(splited_ds[item]) / n * calc_info_val(splited_ds[item])

    return info, split_info_x_val(splited_ds, n)


def split_info_x_val(splited_ds, n):
    info = 0.0
    # print("spplliiittteedd " + str(splited_ds))
    for item in splited_ds:
        info -= len(splited_ds[item]) / n * log(len(splited_ds[item]) / n, 2)

    return info


def split_dataset(dataset, attr_id):
    res = {}
    for item in dataset:
        if item[attr_id] in res:
            res[item[attr_id]].append(item)
        else:
            res[item[attr_id]] = [item]

    return res


def get_min_gain_ratio_id(dataset, attr_id_list):
    max_attr = -1
    max_ratio = -1

    for attr_id in attr_id_list:
        cur_ratio = gain_ratio(dataset, attr_id)

        if cur_ratio == -1:
            return attr_id

        print(str(attr_id) + ": " + str(cur_ratio))

        if cur_ratio > max_ratio or max_ratio < 0:
            max_attr = attr_id
            max_ratio = cur_ratio

    return max_attr


def get_attr_count():
    return len(labels)-2


def get_max_class(dataset):
    vals = get_unique_vals(dataset, get_attr_count())

    max_class = None
    max_count = 0
    for item in vals:
        if max_count < len(vals[item]):
            max_count = len(vals[item])
            max_class = item
    return max_class


def create_sol_tree(tree, dataset, attributes):

    attr_list = list(attributes)

    res = [i[len(i)-1] for i in dataset]
    if res.count(res[0]) == len(res):
        print("-----------")
        print('res ----> ' + res[0] + " attr: " + str(attr_list))
        print("freq " + str(freq(get_unique_vals(dataset, 5))) + " / " + str(len(dataset)))
        print("-----------")
        return {'res': res[0], 'pred': 1}

    if len(attr_list) < 1:
        print("-----------")
        print("around " + str(get_max_class(dataset)) + 'attr: ' + str(attr_list))
        print("freq " + str(freq(get_unique_vals(dataset, 5))) + " / " + str(len(dataset)))
        print("-----------")
        return {'res': get_max_class(dataset), 'pred': freq(get_unique_vals(dataset, 5))['passed']/len(dataset)}

    next_attr = get_min_gain_ratio_id(dataset, attr_list)
    vals = get_unique_vals(dataset, next_attr)

    print('next: ' + str(next_attr) + ' attr list: ' + str(attr_list))
    attr_list.remove(next_attr)

    tree_c = dict(tree)
    tree_c[next_attr] = {}
    for val in vals:
        print("----val " + str(val) + " attr " + str(next_attr))
        tree_c[next_attr][val] = create_sol_tree(tree, vals[val], attr_list)

    # print('tree now ----> ')
    # pp.pprint(tree_c)
    return tree_c


def freq(dataset):
    d = {}
    for item in dataset:
        d[item] = len(dataset[item])
    return d


def get_unique_vals(dataset, attr_id):
    vals = {}
    for item in dataset:
        if item[attr_id] in vals:
            vals[item[attr_id]].append(item)
        else:
            vals[item[attr_id]] = [item]
    return vals


def get_dataset():
    with open('test.csv', mode='r') as infile:
        reader = csv.reader(infile)
        # print(next(reader))
        return [title for title in next(reader)], \
               [[rows[0], int(rows[1]), int(rows[2]), int(rows[3]), int(rows[4]), int(rows[5]), int(rows[6])]
                for rows in reader]


def prompt():
    tree = {}
    results = []
    while 1:
        try:
            inp = input(">")
        except EOFError:
            print("")
            break

        if inp == "tree":
            tree = create_sol_tree({}, data_x_y, list(range(get_attr_count())))
            pp.pprint(tree)
        elif inp == "test":
            results = apply_tree_on_list(tree, list_test_x, list_test_y)
            print('Results: ' + str(results))
        elif inp == "val":
            tp, fp, tn, fn = get_values(results)
            print('tp: ' + str(tp) + ", fp: " + str(fp) + ", tn: " + str(tn) + ", fn: " + str(fn))
            print('accuracy: ' + str(accuracy(tp, tn, fp, fn)))
            print('precision: ' + str(precision(tp, fp)))
            print('recall: ' + str(recall(tp, fn)))
        elif inp == "draw":
            tree = create_sol_tree({}, data_x_y, list(range(get_attr_count())))
            results = apply_tree_on_list(tree, list_test_x, list_test_y)
            draw_auc_roc(results)
        elif inp == "draw2":
            tree = create_sol_tree({}, data_x_y, list(range(get_attr_count())))
            results = apply_tree_on_list(tree, list_test_x, list_test_y)
            draw_auc_pr(results, list_test_y)
        elif inp == "h":
            print_help()
        elif inp == "q":
            break
        else:
            print_err()


def main():
    print_help()
    prompt()


pp = pprint.PrettyPrinter(depth=16)

labels, data = get_dataset()
print(labels)
students = [row[0] for row in data]
data_x_y = [[int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]), 'passed' if row[6] > 0 else 'not passed']
            for row in data]
data_x = [[item[0], item[1], item[2], item[3], item[4]] for item in data_x_y]
data_y = [item[5] for item in data_x_y]

lst = train_test_split(data_x, data_y, test_size=0.2, random_state=4)
train_x, test_x, train_y, test_y = pd.DataFrame(lst[0]), pd.DataFrame(lst[1]), pd.Series(lst[2]), pd.Series(lst[3])
list_test_y = test_y.values.tolist()
list_test_x = test_x.values.tolist()

print(list_test_x[0])

# print(data_x)
# print(data_y)
#
# print(train_x)
# print(train_y)
#
# print(test_x)
# print(test_y)

main()
