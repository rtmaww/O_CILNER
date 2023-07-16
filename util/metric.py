import torch
class Metrics():
    def __init__(self, ignore_index=-100):
        '''
        word_encoder: Sentence encoder
        
        You need to set self.cost as your own loss function.
        '''
        self.ignore_index = ignore_index

    def __get_class_span_dict__(self, label, is_string=False):
        '''
        return a dictionary of each class label/tag corresponding to the entity positions in the sentence
        {label:[(start_pos, end_pos), ...]}
        '''
        class_span = {}
        current_label = None
        i = 0
        if not is_string:
            # having labels in [0, num_of_class] 
            while i < len(label):
                if label[i] > 0:
                    start = i
                    current_label = label[i]
                    i += 1
                    while i < len(label) and label[i] == current_label:
                        i += 1
                    if current_label in class_span:
                        class_span[current_label].append((start, i))
                    else:
                        class_span[current_label] = [(start, i)]
                else:
                    assert label[i] == 0
                    i += 1
        else:
            # having tags in string format ['O', 'O', 'person-xxx', ..]
            while i < len(label):
                if label[i] != 'O':
                    start = i
                    current_label = label[i]
                    i += 1
                    while i < len(label) and label[i] == current_label:
                        i += 1
                    if current_label in class_span:
                        class_span[current_label].append((start, i))
                    else:
                        class_span[current_label] = [(start, i)]
                else:
                    i += 1
        return class_span

    def __get_intersect_by_entity__(self, pred_class_span, label_class_span):
        '''
        return the count of correct entity
        '''
        cnt = 0
        for label in label_class_span:
            cnt += len(list(set(label_class_span[label]).intersection(set(pred_class_span.get(label,[])))))
        return cnt

    def __get_cnt__(self, label_class_span):
        '''
        return the count of entities
        '''
        cnt = 0
        for label in label_class_span:
            cnt += len(label_class_span[label])
        return cnt

    def __get_correct_span__(self, pred_span, label_span):
        '''
        return count of correct entity spans
        '''
        pred_span_list = []
        label_span_list = []
        for pred in pred_span:
            pred_span_list += pred_span[pred]
        for label in label_span:
            label_span_list += label_span[label]
        return len(list(set(pred_span_list).intersection(set(label_span_list))))

    def __get_wrong_within_span__(self, pred_span, label_span):
        '''
        return count of entities with correct span, correct coarse type but wrong finegrained type
        '''
        cnt = 0
        for label in label_span:
            coarse = label.split('-')[0]
            within_pred_span = []
            for pred in pred_span:
                if pred != label and pred.split('-')[0] == coarse:
                    within_pred_span += pred_span[pred]
            cnt += len(list(set(label_span[label]).intersection(set(within_pred_span))))
        return cnt

    def __get_wrong_outer_span__(self, pred_span, label_span):
        '''
        return count of entities with correct span but wrong coarse type
        '''
        cnt = 0
        for label in label_span:
            coarse = label.split('-')[0]
            outer_pred_span = []
            for pred in pred_span:
                if pred != label and pred.split('-')[0] != coarse:
                    outer_pred_span += pred_span[pred]
            cnt += len(list(set(label_span[label]).intersection(set(outer_pred_span))))
        return cnt

    def __get_type_error__(self, pred, label, query):
        '''
        return finegrained type error cnt, coarse type error cnt and total correct span count
        '''
        pred_tag, label_tag = self.__transform_label_to_tag__(pred, query)
        pred_span = self.__get_class_span_dict__(pred_tag, is_string=True)
        label_span = self.__get_class_span_dict__(label_tag, is_string=True)
        total_correct_span = self.__get_correct_span__(pred_span, label_span) + 1e-6
        wrong_within_span = self.__get_wrong_within_span__(pred_span, label_span)
        wrong_outer_span = self.__get_wrong_outer_span__(pred_span, label_span)
        return wrong_within_span, wrong_outer_span, total_correct_span
                
    def metrics_by_entity_(self, pred, label):
        '''
        return entity level count of total prediction, true labels, and correct prediction
        '''
        pred_class_span = self.__get_class_span_dict__(pred, is_string=True)
        label_class_span = self.__get_class_span_dict__(label, is_string=True)
        pred_cnt = self.__get_cnt__(pred_class_span)
        label_cnt = self.__get_cnt__(label_class_span)
        correct_cnt = self.__get_intersect_by_entity__(pred_class_span, label_class_span)
        return pred_cnt, label_cnt, correct_cnt

    def metrics_by_entity(self, pred, label):
        pred_cnt = 0
        label_cnt = 0
        correct_cnt = 0
        for i in range(len(pred)):
            p_cnt, l_cnt, c_cnt = self.metrics_by_entity_(pred[i], label[i])
            pred_cnt += p_cnt
            label_cnt += l_cnt
            correct_cnt += c_cnt
        precision = correct_cnt / (pred_cnt + 1e-8)
        recall = correct_cnt / (label_cnt + 1e-8)
        #print("***correct_cnt{}***".format(correct_cnt))
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1
    def metrics_by_token(self, pred, label):
        pred_cnt = 0
        label_cnt = 0
        correct_cnt = 0
        for i in range(len(pred)):
            for j in range(len(pred[i])):
                if label[i][j] != "O":
                    label_cnt += 1
                if pred[i][j] != "O":
                    pred_cnt += 1
                if label[i][j] == pred[i][j] and label[i][j] != "O":
                    correct_cnt += 1
        precision = correct_cnt / (pred_cnt + 1e-8)
        recall = correct_cnt / (label_cnt + 1e-8)
        # print("***correct_cnt{}***".format(correct_cnt))
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

def flatten_lists(lists):
    """
    将列表的列表拼成一个列表
    :param lists:
    :return:
    """
    flatten_list = []
    for l in lists:
        if type(l) == list:
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list

def compute_metrics(metric):

    results = metric.compute()
    print(results)
    # Unpack nested dictionaries
    macro_results = {}
    num_entity = 0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    for key, value in results.items():
        if isinstance(value, dict):

            precision += value["precision"]
            recall += value["recall"]
            f1 += value["f1"]
            num_entity += 1
    macro_results["precision"] = precision/num_entity
    macro_results["recall"] = recall/num_entity
    macro_results["f1"] = f1/num_entity
    micro_results = {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"]
        }
    return macro_results, micro_results, results

def cal_forget(f1_matrix):
    forget = torch.zeros([f1_matrix.size(0)])
    for step_i in range(1, f1_matrix.size(0)):
        forget_step_i = 0
        for task_j in range(0, step_i):
            forget_step_i += torch.max(f1_matrix[:step_i + 1, task_j]) - f1_matrix[step_i, task_j]
        # step_i pairs
        forget_step_i = forget_step_i / step_i
        forget[step_i] = forget_step_i
    return forget

