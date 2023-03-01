import os
import sys
import yaml
import torch
import argparse
import trainer
from utils import scaler
import models
from dataset import TPDataset
from torch.utils.data import DataLoader


def load_config(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def train_main():
    if (start_epoch == 0) and (not os.path.exists(params_path)):
        os.makedirs(params_path)
        print('create params directory %s' % (params_path))
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path))
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        print('train from params directory %s' % (params_path))
    else:
        raise SystemExit('Wrong type of model!')

    print('param list:')
    print('CUDA\t', DEVICE)
    print('in_channels\t', in_channels)
    print('nb_block\t', nb_block)
    print('nb_chev_filter\t', nb_chev_filter)
    print('nb_time_filter\t', nb_time_filter)
    print('time_strides\t', time_strides)
    print('batch_size\t', batch_size)
    print('graph_signal_matrix_filename\t', graph_signal_matrix_filename)
    print('start_epoch\t', start_epoch)
    print('epochs\t', epochs)
    masked_flag=0
    criterion = nn.L1Loss().to(DEVICE)
    criterion_masked = masked_mae
    if loss_function=='masked_mse':
        criterion_masked = masked_mse         #nn.MSELoss().to(DEVICE)
        masked_flag=1
    elif loss_function=='masked_mae':
        criterion_masked = masked_mae
        masked_flag = 1
    elif loss_function == 'mae':
        criterion = nn.L1Loss().to(DEVICE)
        masked_flag = 0
    elif loss_function == 'rmse':
        criterion = nn.MSELoss().to(DEVICE)
        masked_flag= 0
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    sw = SummaryWriter(logdir=params_path, flush_secs=5)
    print(net)

    print('Net\'s state_dict:')
    total_param = 0
    for param_tensor in net.state_dict():
        print(param_tensor, '\t', net.state_dict()[param_tensor].size())
        total_param += np.prod(net.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)

    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    start_time = time()

    if start_epoch > 0:

        params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)

        net.load_state_dict(torch.load(params_filename))

        print('start epoch:', start_epoch)

        print('load weight from: ', params_filename)

    # train model
    for epoch in range(start_epoch, epochs):

        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        if masked_flag:
            val_loss = compute_val_loss_mstgcn(net, val_loader, criterion_masked, masked_flag,missing_value,sw, epoch)
        else:
            val_loss = compute_val_loss_mstgcn(net, val_loader, criterion, masked_flag, missing_value, sw, epoch)


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename)

        net.train()  # ensure dropout layers are in train mode

        for batch_index, batch_data in enumerate(train_loader):

            encoder_inputs, labels = batch_data

            optimizer.zero_grad()

            outputs = net(encoder_inputs)

            if masked_flag:
                loss = criterion_masked(outputs, labels,missing_value)
            else :
                loss = criterion(outputs, labels)


            loss.backward()

            optimizer.step()

            training_loss = loss.item()

            global_step += 1

            sw.add_scalar('training_loss', training_loss, global_step)

            if global_step % 1000 == 0:

                print('global step: %s, training loss: %.2f, time: %.2fs' % (global_step, training_loss, time() - start_time))

    print('best epoch:', best_epoch)

    # apply the best model on the test set
    predict_main(best_epoch, test_loader, test_target_tensor,metric_method ,_mean, _std, 'test')

def predict_main(global_step, data_loader, data_target_tensor,metric_method, _mean, _std, type):
    '''

    :param global_step: int
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param mean: (1, 1, 3, 1)
    :param std: (1, 1, 3, 1)
    :param type: string
    :return:
    '''

    params_filename = os.path.join(params_path, 'epoch_%s.params' % global_step)
    print('load weight from:', params_filename)

    net.load_state_dict(torch.load(params_filename))

    predict_and_save_results_mstgcn(net, data_loader, data_target_tensor, global_step, metric_method,_mean, _std, params_path, type)

def main(args):
    model_config = load_config(args.model_config_path)
    train_config = load_config(args.train_config_path)
    torch.manual_seed(train_config['seed'])
    torch.cuda.manual_seed(train_config['seed'])
    # ----------------------- Load data ------------------------
    Scaler = getattr(sys.modules['utils.scaler'], train_config['scaler'])
    data_scaler = Scaler(axis=(0, 1, 2))

    data_config = model_config['dataset']
    USE_CUDA = torch.cuda.is_available()
    device = torch.device(data_config['device'] if USE_CUDA else 'cpu')
    print("CUDA:", USE_CUDA, device )#设置是使用GPU还是CPU
    data_names = ('train.npz', 'valid.npz', 'test.npz')

    data_loaders = []
    for data_name in data_names:
        dataset = TPDataset(os.path.join(data_config['data_dir'], data_name))
        if data_name == 'train.npz':
            data_scaler.fit(dataset.data['x'])
        dataset.fit(data_scaler)
        data_loader = DataLoader(dataset, batch_size=data_config['batch_size'])
        data_loaders.append(data_loader)

    # --------------------- Trainer setting --------------------
    net_name = args.model_name
    net_config = model_config[net_name]
    net_config.update(data_config)

    Model = models.LSTM.LSTM
    if Model is None:
        raise ValueError('Model {} is not right!'.format(net_name))
    net_pred = Model(**net_config).to(device)

    Optimizer = getattr(sys.modules['torch.optim'], train_config['optimizer'])
    optimizer_pred = Optimizer(
        net_pred.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay']
    )
    sc = train_config.get('lr_scheduler', None)
    if sc is None:
        scheduler_pred, scheduler_graph = None, None
    else:
        Scheduler = getattr(sys.modules['torch.optim.lr_scheduler'], sc.pop('name'))
        scheduler_pred = Scheduler(optimizer_pred, **sc)


    # --------------------------- Train -------------------------

    train_main(data_loaders[0], data_loaders[1])

    predict_main(13, data_loaders[-1], test_target_tensor,metric_method, _mean, _std, 'test')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config_path', type=str, default='./config/train_pems0709.yaml',
                        help='Config path of models')
    parser.add_argument('--train_config_path', type=str, default='./config/train_config.yaml',
                        help='Config path of Trainer')
    parser.add_argument('--model_name', type=str, default='LSTM', help='Model name to train')
    parser.add_argument('--num_epoch', type=int, default=5, help='Training times per epoch')
    parser.add_argument('--num_iter', type=int, default=20, help='Maximum value for iteration')
    parser.add_argument('--model_save_path', type=str, default='./model_states/AdapGLA_pems0709.pkl',
                        help='Model save path')
    parser.add_argument('--max_graph_num', type=int, default=3, help='Volume of adjacency matrix set')
    args = parser.parse_args()

    main(args)
