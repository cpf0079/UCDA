import torch
import torch.nn as nn
from sympy import *
import math
import sympy
import numpy as np
from model import Model
from basedataset import MyDataset
from functions import weight_avg, find_para, v_prob, u_prob, single_distance
import xlwt
from config import cfg


device = cfg.DEVICE
batch_size = cfg.BATCH_SIZE_TEST
eta = cfg.ETA
epsilon = cfg.EPSILON


def cluster_subdomain(subjective_list, lambda1):
    subjective_list = sorted(subjective_list, key=lambda img: img[1])
    copy_list = subjective_list.copy()
    subjective_rank = [item[0] for item in subjective_list]

    confident_split = subjective_rank[ : int(len(subjective_rank) * lambda1)]
    uncertain_split = subjective_rank[int(len(subjective_rank) * lambda1): ]

    with open('confident_split.txt', 'w+') as f:
        for item in confident_split:
            f.write('%s\n' % item)

    with open('uncertain_split.txt', 'w+') as f:
        for item in uncertain_split:
            f.write('%s\n' % item)

    return copy_list


def subjectivity(prob):
    # print(prob)
    # prob_item = prob.item()
    avg = weight_avg(prob)
    # print(avg)
    med = u_prob(prob)

    prob = prob.squeeze(0)
    para = find_para(avg)
    dud = v_prob(para)

    dud = dud.to(device)
    med = med.to(device)

    return single_distance(prob, dud) + epsilon * single_distance(prob, med)


def main():
    model = Model().to(device)
    model.load_state_dict(torch.load(cfg.BEST_MODEL_DIR))
    model.eval()

    target_data = MyDataset(txt_dir=cfg.TARGET_TXT_DIR,
                            label_dir=cfg.TARGET_LABEL_DIR,
                            root=cfg.TARGET_FRAME_DIR,
                            num_samples=cfg.TARGET_DATA_SAMPLE)

    target_loader = torch.utils.data.DataLoader(dataset=target_data,
                                                batch_size=batch_size,
                                                shuffle=True)

    target_loader_iter = iter(target_loader)

    list = []
    workbook = xlwt.Workbook()
    worksheet = workbook.add_sheet('Sheet1')

    for index in range(len(target_loader)):
        print(index+1)
        data_target = target_loader_iter.__next__()
        t_img, _, name = data_target

        with torch.no_grad():
            t_img = t_img.to(device)
            # t_img = t_img
            class_output, _ = model(input_data=t_img, alpha=1.0)

            out = class_output.squeeze(0)
            out = out.tolist()

            for i in range(len(out)):
                worksheet.write(index, i, out[i])

            I = subjectivity(class_output)
            print(I)

            list.append((name[0], I.item()))

    workbook.save('pseudo_target.xls')
    cluster_subdomain(list, eta)


if __name__ == '__main__':
    main()
    # inputs = torch.rand(1, 5)
    # func = nn.Softmax(dim=1)
    # outputs = func(inputs)
    # # output = outputs.squeeze(0)
    # dis = subjectivity(outputs)
    # print(dis)

    # list = []
    # for i in range(500):
    #     a = torch.rand(1)
    #     list.append(((i+1), a.item()))
    # print(list)
    # cluster_subdomain(list, 0.6)


