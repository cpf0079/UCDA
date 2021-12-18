import random
import torch
import os
import sys
import torch.optim as optim
import torch.utils.data
import numpy as np
from basedataset import MyDataset
from model import Model
from functions import distance, optimizer_scheduler
from config import cfg


manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)

batch_size = cfg.BATCH_SIZE_TRAIN
lr = cfg.LR_FIRST_UDA
n_epoch = cfg.EPOCH_FIRST_UDA
theta = cfg.THETA

device = cfg.DEVICE

source_data = MyDataset(txt_dir=cfg.SOURCE_TXT_DIR,
                        label_dir=cfg.SOURCE_LABEL_DIR,
                        root=cfg.SOURCE_FRAME_DIR,
                        num_samples=cfg.SOURCE_DATA_SAMPLE)

source_loader = torch.utils.data.DataLoader(dataset=source_data,
                                            batch_size=batch_size,
                                            num_workers=cfg.NUM_WORKERS,
                                            shuffle=cfg.TRAIN_SHUFFLE)

target_data = MyDataset(txt_dir=cfg.TARGET_TXT_DIR,
                        label_dir=cfg.STARGET_LABEL_DIR,
                        root=cfg.TARGET_FRAME_DIR,
                        num_samples=cfg.TARGET_DATA_SAMPLE)

target_loader = torch.utils.data.DataLoader(dataset=target_data,
                                            batch_size=batch_size,
                                            num_workers=cfg.NUM_WORKERS,
                                            shuffle=cfg.SOURCE_DATA_SAMPLE)

my_net = Model().to(device)

optimizer = optim.Adam(my_net.parameters(), lr=lr)

loss_domain = torch.nn.NLLLoss()
min_loss = cfg.MIN_LOSS

for p in my_net.parameters():
    p.requires_grad = True

for epoch in range(n_epoch):
    batch_losses = []

    len_dataloader = min(len(source_loader), len(target_loader))
    data_source_iter = iter(source_loader)
    data_target_iter = iter(target_loader)

    for i in range(len_dataloader):
        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        data_source = data_source_iter.__next__()
        s_img, s_label, _ = data_source
        # print(s_img.size())
        # print(s_label.size())

        # my_net.zero_grad()
        batch_size = len(s_label)
        # print(batch_size)

        optimizer = optimizer_scheduler(optimizer, p)
        optimizer.zero_grad()

        domain_label = torch.zeros(batch_size).long()

        s_img = s_img.to(device)
        s_label = s_label.to(device)
        domain_label = domain_label.to(device)

        class_output, domain_output = my_net(input_data=s_img, alpha=alpha)
        # print(class_output.size())
        # print(s_label.size())
        err_s_label = distance(class_output, s_label)
        err_s_domain = loss_domain(domain_output, domain_label)

        # training model using target data
        data_target = data_target_iter.__next__()
        t_img, _, _ = data_target

        batch_size = len(t_img)

        domain_label = torch.ones(batch_size).long()

        t_img = t_img.to(device)
        domain_label = domain_label.to(device)

        _, domain_output = my_net(input_data=t_img, alpha=alpha)
        err_t_domain = loss_domain(domain_output, domain_label)

        domain_loss = err_t_domain + err_s_domain

        loss = err_s_label + theta * domain_loss
        batch_losses.append(loss.item())

        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Class_loss: {:.4f}, Domain_loss: {:.4f}'.format(
                                                                        epoch + 1, n_epoch, i + 1, len_dataloader,
                                                                        loss.item(), err_s_label.item(),
                                                                        domain_loss.item()))

    avg_loss = sum(batch_losses) / len_dataloader
    print('Epoch {}, Averaged loss: {:.4f} %'.format(epoch+1, avg_loss))

    is_best = avg_loss < min_loss
    min_loss = min(avg_loss, min_loss)
    if is_best:
        torch.save(my_net.state_dict(), 'best_model.pth')
        print('Save the best weights!')

