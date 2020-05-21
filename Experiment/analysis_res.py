from BasicalClass import *


def select_thresh(score, truth, req):
    truth = truth.reshape([-1])
    score_cand = np.sort(score)
    for val in score_cand:
        pred = np.int32(score > val).reshape([-1])
        acc = (np.sum((pred == 1) * (truth == 1))) / (np.sum(pred) + 0.00001)
        coverage = (np.sum((pred == 1) * (truth == 1))) / (np.sum(truth))
        if acc > req:
            return val, coverage, acc
    return None, None, None

def simaliar_matrix(filter_index : list):
    metric_num = len(filter_index)
    sim_mat = np.zeros([metric_num, metric_num])
    for i in range(metric_num):
        for j in range(metric_num):
            union_set = len(set(filter_index[i]) & set(filter_index[j])).__float__()
            sim_mat[i, j] = \
                union_set / (min(len(set(filter_index[i])), len(set(filter_index[j]))) + 0.00001)
    return sim_mat


def run(train_score_list, test_score_list, train_truth, test_truth):
    req_list = [0.95, 0.99]
    for req in req_list:
        filter_list = []
        for i, score in enumerate(train_score_list):
            thresh, _, _ = select_thresh(score, train_truth, req)
            if thresh is not None:
                filter_index = list(np.where(test_score_list[i] > thresh)[0])
            else:
                filter_index = list(np.where(test_score_list[i] > np.inf)[0])
            filter_list.append(filter_index)
        sim_mat = simaliar_matrix(filter_list)
        print('=============',req,'===================')
        for sim_vec in sim_mat:
            print(sim_vec)

def main():
    truth = torch.load('./Result/Fashion/truth.res')
    dropout_score = torch.load('./Result/Fashion/dropout.res')
    scale_score = torch.load('./Result/Fashion/scale.res')
    viallina_score = torch.load('./Result/Fashion/viallina.res')

    train_score_list = [dropout_score[0], scale_score[0], viallina_score[0]]
    test_score_list = [dropout_score[1], scale_score[1], viallina_score[1]]
    train_truth = truth[1]
    test_truth = truth[2]
    run(train_score_list, test_score_list, train_truth, test_truth)



if __name__ == '__main__':
    main()
