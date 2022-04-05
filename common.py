from sklearn import metrics

encoding = 'utf-8'

def parse_dataset(filepath):
    x_set = []
    y_set = []

    entry_count = 0
    skipped_lines = 0

    print(f'reading file {filepath}...')

    with open(filepath, encoding=encoding) as f:
        for line in f:
            split_line = line.strip().split(' ', 1)
            if len(split_line) != 2:
                skipped_lines += 1
                continue

            y_set.append(int(split_line[0].strip()))
            x_set.append(split_line[1].strip())
            entry_count += 1

    print(f'parsed {entry_count} lines, skipped {skipped_lines} lines')

    return x_set, y_set, entry_count

def calculate_metrics(y, pred, report, categories, confusion_matrix):
    print('calculating metrics...')
    accuracy = metrics.accuracy_score(y, pred)
    f1 = metrics.f1_score(y, pred, average='macro')
    print("accuracy:   %0.3f" % accuracy)
    print("f1 score:   %0.3f" % f1)
    if report:
        print("classification report:")
        print(metrics.classification_report(y, pred, target_names=categories))
    if confusion_matrix:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y, pred))