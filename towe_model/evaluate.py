
from numpy import core


pad_i = 0
tag2id = {'B': 1, 'I': 2, 'O': 3 ,'X': 4, 'SEP':5, 'CLS': 6}

def score_BIO(predicted, golden, ignore_index=0,sentence_id=None):
    assert len(predicted) == len(golden)
    assert len(predicted) == len(sentence_id)

    sum_all = 0
    sum_correct = 0
    golden_01_count = 0
    predict_01_count = 0
    correct_01_count = 0
    count_null = 0

    # total, null, under-extracted, over-extracted, others
    ids_ = []
    pred_ = []
    label_ = []

    for i in range(len(golden)):
        # for each sentence
        length = len(golden[i]) # seq len

        golden_01 = 0
        correct_01 = 0
        predict_01 = 0
        predict_items = []
        golden_items = []
        golden_seq = []
        predict_seq = []

        for j in range(length): # for each word
            if golden[i][j] == ignore_index:
                break

            if golden[i][j] == tag2id['B']:
                if len(golden_seq) > 0:  # 00
                    golden_items.append(golden_seq)
                    golden_seq = []
                golden_seq.append(j)
            elif golden[i][j] == tag2id['I']:
                if len(golden_seq) > 0:
                    golden_seq.append(j)
            elif golden[i][j] == tag2id['O']:
                if len(golden_seq) > 0:
                    golden_items.append(golden_seq)
                    golden_seq = []
            
            if predicted[i][j] == tag2id['B']:
                if len(predict_seq) > 0:  # 00
                    predict_items.append(predict_seq)
                    predict_seq = []
                predict_seq.append(j)
            elif predicted[i][j] == tag2id['I']:
                if len(predict_seq) > 0:
                    predict_seq.append(j)
            elif predicted[i][j] == tag2id['O']:
                if len(predict_seq) > 0:
                    predict_items.append(predict_seq)
                    predict_seq = []
        
        if len(golden_seq) > 0:
            golden_items.append(golden_seq)
        if len(predict_seq) > 0:
            predict_items.append(predict_seq)
        golden_01 = len(golden_items)
        predict_01 = len(predict_items)
        correct_01 = sum([item in golden_items for item in predict_items])

        if predict_01 == 0:
            count_null += 1
            
        if correct_01 != predict_01:
            ids_.append(sentence_id[i])
            pred_.append(predicted[i])
            label_.append(golden[i])

        golden_01_count += golden_01
        predict_01_count += predict_01
        correct_01_count += correct_01
    precision = correct_01_count / predict_01_count if predict_01_count > 0 else 0
    recall = correct_01_count / golden_01_count if golden_01_count > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    score_dict = {'precision': precision, 'recall': recall, 'f1': f1}
    print("NULL:", count_null)
    return score_dict, (ids_, pred_, label_)
