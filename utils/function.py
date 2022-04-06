from collections import deque
import random
import numpy as np
import tensorflow as tf
import torch
from torch import tensor

from torch.utils.data import DataLoader
from turbojpeg import TurboJPEG
from LipLRN.models.DRN import DRN


def dataset2dataloader(dataset, batch_size, num_workers, shuffle=True):
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=shuffle,
                        drop_last=False,
                        pin_memory=True)
    return loader


jpeg = TurboJPEG()

memory = deque(maxlen=512)
replay_size = 64
epoches = 2000
pre_train_num = 256
gamma = 0.  # every state is i.i.d
alpha = 0.5
forward = 512
epislon_total = 2018


def createDQN(args):
    model = DRN(args)
    # video_input = Input(shape=(40, 88, 88, 1), dtype='float32', name='video_inputs')
    # conv1 = Conv3D(64, (5, 7, 7), strides=(1, 2, 2), padding=(2, 3, 3), use_bias=False)(video_input)

    return model


def copy_critic_to_actor(critic_model, actor_model):
    # critic_weights = critic_model.get_weights()
    # actor_wegiths = actor_model.get_weights()
    # for i in range(len(critic_weights)):
    #     actor_wegiths[i] = critic_weights[i]
    # actor_model.set_weights(actor_wegiths)
    actor_model.load_state_dict(critic_model.state_dict())


def get_q_values(model_, state):
    inputs_ = state.reshape(1, *state.shape)
    qvalues = model_.predict(inputs_)
    return qvalues[0]


def predict(model, states, num_actions):
    inputs_ = [states, np.ones(shape=(len(states), num_actions))]
    qvalues = model.predict(inputs_)
    return np.argmax(qvalues, axis=1)


def epsilon_calc(step, ep_min=0.01, ep_max=1, ep_decay=0.0001, esp_total=1000):
    return max(ep_min, ep_max - (ep_max - ep_min) * step / esp_total)


def epsilon_greedy(env, actor_q_model, state, step, ep_min=0.01, ep_decay=0.0001, ep_total=1000):
    epsilon = epsilon_calc(step, ep_min, 1, ep_decay, ep_total)
    if np.random.rand() < epsilon:
        return env.sample_actions(), 0
    qvalues = get_q_values(actor_q_model, state)
    qvalues = qvalues.numpy()
    return np.argmax(qvalues), np.max(qvalues)


def remember(state, action, action_q, reward, next_state):
    memory.append([state, action, action_q, reward, next_state])


def sample_ram(sample_num):
    list = random.sample(memory, sample_num)
    for i in range(len(list)):
        for j in range(len(list[i])):
            a = list[i][j]
            if torch.is_tensor(list[i][j]):
                list[i][j] = list[i][j].detach().numpy()

    return np.array(list)


def fit(epoch, model, trainloader, testloder, q):
    correct = 0
    total = 0
    running_loss = 0
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    # 注意这里与前几次图片分类的不一样
    # 返回的是一个批次成对数据
    for b in trainloader:
        x, y = b.text, b.label
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()

    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = correct / total

    test_correct = 0
    test_total = 0
    test_running_loss = 0

    model.eval()
    with torch.no_grad():
        # 这里也是同样变化
        for b in testloder:
            x, y = b.text, b.label
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()

    epoch_test_loss = test_running_loss / len(testloder.dataset)
    epoch_test_acc = test_correct / test_total

    print('epoch: ', epoch,
          'train_loss: ', round(epoch_loss, 3),
          'train_accuracy: ', round(epoch_acc, 3),
          'test_loss: ', round(epoch_test_loss, 3),
          'test_accuracy: ', round(epoch_test_acc, 3)
          )

    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc
