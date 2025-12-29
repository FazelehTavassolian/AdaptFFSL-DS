# Train Clients
import warnings
import pickle

warnings.simplefilter('ignore', UserWarning)

# Common
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import Union
from pathlib import Path
import copy
import shutil
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
from torchsummary import summary
from sklearn import metrics
import torch.nn.functional as F

# Torch
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

# Utils
from utils import read_py_config

# Loss
from loss import RegisterLoss, mk_loss

# Optimizer
from optim import RegisterOptim, mk_optim

# Env
from environment import get_dataset_environment, get_model_environment

# Updater
from local import LocalUpdater
from local import LocalUpdate as LU

from sklearn.metrics.pairwise import cosine_similarity

from DQL import DQL

def myflatten(model):
    n = sum(p.numel() for p in model.parameters())
    params = torch.zeros(n)
    i = 0
    for p in model.parameters():
        params_slice = params[i:i + p.numel()]
        params_slice.copy_(p.flatten())
        p.data = params_slice.view(p.shape)
        i += p.numel()
    return params.tolist()


def flatten_state(state_list):
    '''
    Flatten a list of states.

    Parameters:
    - state_list: List of states where each state is represented as a list of sublists.

    Returns:
    - Flattened list of states as a torch.Tensor.
    '''
    result_list = []
    max_length = max(len(sublist) for sublist in state_list)  # Find the maximum length of sublists

    for i in range(max_length):
        for sublist in state_list:
            if i < len(sublist):
                element = sublist[i]
                if isinstance(element, list):
                    result_list.extend(element)
                else:
                    result_list.append(element)
    
    return torch.Tensor(result_list)
    #return (result_list)

def main(names, cores, frequency, bandwidth,min_freq, max_freq, min_bw, max_bw, args: Namespace) -> None:
    """
    Starting point
    """
    lamb=0.6
    hidden_layer = 32

    # Paths
    root: Path = args.save_root
    root.mkdir(parents=True, exist_ok=True)

    assets = root.joinpath('assets')
    assets.mkdir(parents=True, exist_ok=True)

    plots = root.joinpath('plots')
    plots.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if args.verbose:
        print(device)
    dataset_fs = open(str(root.joinpath('dataset-separation.txt')), 'a')

    # Save Parameter
    with open(str(root.joinpath('parameters.txt')), 'a') as f:
        f.write('Experiment Instruction\n')
        for key, value in vars(args).items():
            f.write(f'{key}\t{value}\n')

    shutil.copy(args.ds_config, root.joinpath(args.ds_config.name))
    cnf = read_py_config(args.ds_config)

    LocalUpdate = LocalUpdater[args.agg]

    # Dataset Split
    clients_ds, val_ds, _ = get_dataset_environment(args, cnf, dataset_fs)

    valid_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    # Server Model
    server_model = get_model_environment(args, cnf)
    server_model.to(device)
    print('global', next(server_model.parameters()).is_cuda)

    
    with open(assets.joinpath('architecture.txt'), 'w') as f:
        total_params = sum(p.numel() for p in server_model.parameters())
        total_trainable_params = sum(p.numel() for p in server_model.parameters() if p.requires_grad)
        print(f'Total Parameters: {total_params}', file=f)
        print(f'Total Trainable Parameters: {total_trainable_params}', file=f)
        print(f'Total None Trainable Parameters: {total_params - total_trainable_params}', file=f)
        print(server_model, file=f)

    criterion = mk_loss(args)

    rnd_vl_acc = []
    rnd_vl_loss = []
    rnd_cl = []
    rnd_resume = 0
    if cnf['model'].get('checkpoint', None) is not None:
        state_dict = torch.load(cnf['model']['checkpoint'])
        server_model.load_state_dict(state_dict['model'])
        rnd_vl_acc = state_dict['rnd_vl_acc']
        rnd_vl_loss = state_dict['rnd_vl_loss']
        rnd_cl = state_dict['rnd_cl']
        rnd_resume = min(state_dict['rnd_resume'], len(rnd_cl))

    client_updater = [LocalUpdate(names[idx], args, server_model, device, cores[idx], frequency[idx], bandwidth[idx], dataset=cl_ds) for
                      idx, cl_ds in enumerate(clients_ds)]
    
    # A: the first iter for all clients
    # A: delay and acc calculate for select the best clients
    # Send Models
    
    global_weight_init = myflatten(copy.deepcopy(server_model))
    
    weights_cosine_similarity = []
    y = np.array(global_weight_init)
    y = y.reshape(1, -1)

    acc = []
    delay = []
    Q_values = []
    alpha = 0.7; beta = 0.3    
    state_list = []
    reputation_clients_t = np.full(len(client_updater), 0.01)
    reputation_list = []
    reputation_list.append(copy.deepcopy(reputation_clients_t))
    
    Accuracy_global_pervoius = 0
    min_del = []
    max_del = []

    # cosine_sim = []
    for cl_idx, cl in enumerate(client_updater):
        cl.set_parameter(server_model)

        # Train Client ith - Local update
        local_models = []
        local_weights = []
        total_samples = 0
        str_rnd = "0"
        local_cl_res = []
        # Load Server model on Client ith
        cl_u: LU = client_updater[cl_idx]
        #print(cl_u.model)
        res = cl_u.train(1, str_rnd=str_rnd, cnf=cnf, args=args)
        local_weights.append(cl_u.train_samples)
        local_models.append(cl_u.model)
        total_samples += cl_u.train_samples

        local_cl_res.append({
            'name': cl_u.name,
            **res,
            **cl_u.train_time_cost
        })
        acc.append(res['accuracy'])

        number_parmeters = sum(p.numel() for p in cl_u.model.parameters() if p.requires_grad)
        #print('number_parmeters', number_parmeters)
        latency = (cl_u.train_samples * 64*40*20) / (cl_u.cores * 1000000 * cl_u.freq) + (number_parmeters * 64) / (10000000 * cl_u.bw)
        min_latency = (cl_u.train_samples * 64*40*20) / (cl_u.cores * 1000000 * min_freq[cl_idx]) + (number_parmeters * 64) / (10000000 * min_bw[cl_idx])
        max_latency = (cl_u.train_samples * 64*40*20) / (cl_u.cores * 1000000 * max_freq[cl_idx]) + (number_parmeters * 64) / (10000000 * max_bw[cl_idx])
        
        delay.append(latency)
        min_del.append(min_latency)
        max_del.append(max_latency)
        #print('acc', res['accuracy'][0], 'lat', latency)
        Q_values.append(alpha * res['accuracy'][0] - beta * latency)

        x = np.array(myflatten(cl_u.model))
        x = x.reshape(1, -1)
        
        #print('sim: ', cosine_similarity(x, y))
        cosine_sim = cosine_similarity(x, y)[0][0]
        weights_cosine_similarity.append(cosine_sim)

        client_state = []
        client_state.append(cosine_sim)
        client_state.append(cl_u.train_samples)
        #clienglobal_weight_initt_state.append(cl_u.cores)
        client_state.append(cl_u.freq)
        client_state.append(cl_u.bw)
        
        state_list.append(client_state)

        rnd_cl.append(local_cl_res)
    
    # State is a concatenation of the different reduced weights
    ####print(state_list)
    state = flatten_state(state_list)
    
    # init dql
    
    dql = DQL(len(state), len(client_updater), batch_size=50)

    list_loss_DQL = []
    rewards = []
    
    time_rounds = []
    time_rounds_sum = []
    total_time=0
    # Train For each communication round 
    for rnd_idx in range(rnd_resume, args.rounds):
        # Select clients based on q-values + explotation
        
        ###print('\n \n Device info train: \n', state_list[cl_idx],'\n ')
        if args.client_select == 'selected':
            # Verify if we need to update the target network
            if (rnd_idx + 1) % dql.update_rate == 0:
                dql.update_target_network()
            
            if (rnd_idx == 0):
                #random selection
                # Random
                m = max(int(args.client_frac * args.n_clients), 1)
                idx_users = np.random.choice(range(args.n_clients), m, replace=False)
            else:
                # epslion greedy
                idx_users = dql.multiaction_selection(state, args.client_frac, rnd_idx, mode = "Mode1")
                idx_users.sort()

            

            # sort q-values
            idx_q_values = sorted(range(len(Q_values)), key=lambda k: Q_values[k])
            idx_users = idx_q_values[-8:] # select the best clients
            explotation = np.setdiff1d(range(args.n_clients), idx_users)
            add_rand = np.random.choice(explotation, 2)
            idx_users.extend(add_rand)

            qq = []
            for i in idx_users:
                qq.append(Q_values[i])

            sfit = sorted(qq)
            th1 = sfit[6]
            th2 = sfit[3]

            #print(idx_users, qq)

        else:
            # Random
            m = max(int(args.client_frac * args.n_clients), 1)
            idx_users = np.random.choice(range(args.n_clients), m, replace=False)

        
        # Train Client ith - Local update
        local_models = []
        local_weights = []
        total_samples = 0
        str_rnd = f"[{rnd_idx + 1}/{args.rounds}]"
        local_cl_res = []
        acc_all_models = []
        time_roundt = []
        weight_local_clients = []

        for cl_idx in idx_users:
            # Load Server model on Client ith
            cl_u: LU = client_updater[cl_idx]
            ep = 5
            if (Q_values[cl_idx]>=th1):
                ep = 7
            elif(th1>Q_values[cl_idx]>th2):
                ep = 5
            else:
                ep = 3

            ####print('\n \n \n', ep, '\n')
            res = cl_u.train(ep, str_rnd=str_rnd, cnf=cnf, args=args)
            acc_all_models.append(res['accuracy'][0])

            local_weights.append(cl_u.train_samples)
            local_models.append(cl_u.model)

            # Append the local model weight
            weight_local_client = myflatten(cl_u.model)

            weight_local_clients.append(weight_local_client)
            # A: MAJ des weights PCA
            x = np.array(weight_local_client)
            
            x = x.reshape(1, -1)
            y = copy.deepcopy(myflatten(server_model))
            y = np.array(y)
            y = y.reshape(1, -1)
            
           ##### print('cl_idx', cl_idx,'\n Device info train: \n', state_list[cl_idx],'\n ',cosine_similarity(x,y)[0][0])
            weights_cosine_similarity[cl_idx] = cosine_similarity(x,y)[0][0]
            #print(cosine_sim[client_index])
            state_list[cl_idx][0] =  weights_cosine_similarity[cl_idx] #list((pca.transform(np.array(self.flatten(copy.deepcopy(client_w))).reshape(1, -1)))[0])
                

            total_samples += cl_u.train_samples
            local_cl_res.append({
                'name': cl_u.name,
                **res,
                **cl_u.train_time_cost
            })

            latency = (cl_u.train_samples * 64*40*20) / (cl_u.cores * 1000000 * cl_u.freq) + (number_parmeters * 64) / (10000000 * cl_u.bw)
            delay[cl_idx]=latency
            Q_values[cl_idx] = alpha * res['accuracy'][0] - beta * delay[cl_idx]
            time_roundt.append(latency)
            total_time = latency+total_time

        input_layer = 5 * 101
        NoDQNParameters = (input_layer * hidden_layer + 1) + (hidden_layer + 1)
        CPUSpeed = 4900 #MIPS
        DQN_latency = NoDQNParameters / CPUSpeed
        time_rounds.append(max(time_roundt) + DQN_latency)
        time_rounds_sum.append(sum(time_roundt))
        rnd_cl.append(local_cl_res)

        # Normalize the weights
        for i, w in enumerate(local_weights):
            local_weights[i] = w / total_samples

        # Aggregation
        assert len(local_models) > 0
        server_model = copy.deepcopy(local_models[0])
        for param in server_model.parameters():
            param.data.zero_()
        for w, client_model in zip(local_weights, local_models):
            for srv_param, cl_param in zip(server_model.parameters(), client_model.parameters()):
                srv_param.data += cl_param.data.clone() * w


        # Evaluate Server Model
        with torch.no_grad():
            val_loss = 0.0
            val_acc = 0.0
            val_iter = tqdm(valid_ld)
            str_rd = f"[{rnd_idx + 1}/{args.rounds}]"
            step = 0
            for i, data in enumerate(val_iter):
                str_st = f"[{i + 1}/{len(valid_ld)}]"
                data = tuple(fr.to(device) for fr in data)
                y = data[-1].view(-1)
                # if cnf['model']['name'] == 'relationNet':
                y = F.one_hot(y, 5)
                y = y.float()

                server_model.to(device)
                output = server_model(data)
                loss = criterion(output, y)
                val_loss += loss.item()

                # if cnf['model']['name'] == 'relationNet':
                batch_acc = metrics.accuracy_score(np.argmax(y.detach().cpu().numpy(), axis=-1),
                                                   np.argmax(output.detach().cpu().numpy(), axis=-1))
                # else:
                #     batch_acc = metrics.accuracy_score(y.detach().cpu().numpy(),
                #                                        np.argmax(output.detach().cpu().numpy(), axis=-1))
                val_acc += batch_acc
                str_postfix = f'[Server Valid] Round: {str_rd} Step: {str_st} Batch Acc: {round(batch_acc, 3)} Acc: {round(val_acc / (i + 1), 3)} Batch Loss: {round(loss.item(), 3)} Loss: {round(val_loss / (i + 1), 3)}'
                show = OrderedDict({f'[Train]': str_postfix})
                val_iter.set_postfix(show)
                step = i
            rnd_vl_acc.append(val_acc / (step + 1))
            rnd_vl_loss.append(val_loss / (step + 1))


        # Update reduced global parameter 
        #weight_list_for_iteration_pca[0] =  (pca.transform(np.array(self.flatten(copy.deepcopy(self.model.state_dict()))).reshape(1, -1)))[0]
        
        # Next state
        print(state_list, '\n',len(state_list), '\n',len(state_list[0]))
        next_state = flatten_state(state_list)
        
        # Action representation
        action = np.zeros(len(client_updater))
        action[idx_users] = 1

        
        # Calculate the reward
        
        # Calculate the utility function (score of each participant client)
        
        
        # Calculate the normalized distance between the local weight and the global model weight
        num_param = sum(p.numel() for p in server_model.parameters() if p.requires_grad)

        # The average of the different points received
        average_weights =  copy.deepcopy(server_model)
        
        normalized_distance = 1/num_param * (np.sum((np.array(weight_local_clients) -
                              np.array(myflatten(average_weights)))/np.array(myflatten(average_weights)), axis = 1))
        
        # Utility is a positive number that indicate if the client selected contributed in a good way or not
        # When utility is near 1, the client did contribute in a good way otherwise not
        utility_clients = np.array([])
        
        if(val_acc > Accuracy_global_pervoius):
            # if we had an increase in the F1 score we want to minimaze the distance
            # ie : the client near the global weights are the good one
            utility_clients = np.exp(-np.abs(normalized_distance))
        else:
            utility_clients = 1- np.exp(-np.abs(normalized_distance))
                    
        # Now we will use the same reputation eq as the FedDRL_Reputation, just replacing the accuracies with the utility
        
        reputation_clients_t[idx_users] = (1 - lamb)*reputation_clients_t[idx_users] + lamb*(utility_clients - (((np.array(time_roundt)) - min_latency) / (max_latency - min_latency)))
        
        ###print(reputation_clients_t)

        reputation_list.append(copy.deepcopy(reputation_clients_t))
        reward = np.array(reputation_clients_t[idx_users])
        #print("the reward is ", reward)

        rewards.append(reward)

        #store the transition information   
        
        if (rnd_idx == args.rounds - 1):
            dql.store_transistion(state, action, reward, next_state, done = True)
        else:
            dql.store_transistion(state, action, reward, next_state, done = False)
            
        #update current state to next state
        state = copy.deepcopy(next_state)
        Accuracy_global_pervoius = val_acc

        loss_dql = dql.train(rnd_idx, mode = "Mode1")
        list_loss_DQL.append(loss_dql)

        # Checkpoint
        torch.save({
            # 'model': server_model.state_dict(),
            # 'rnd_resume': rnd_idx + 1,
            # 'rnd_vl_loss': rnd_vl_loss,
            # 'rnd_vl_acc': rnd_vl_acc,
            # 'rnd_cl': rnd_cl,
            # 'Timesum' : time_rounds_sum,
            # 'Reputation' : reputation_list,
            # 'Rewards' : rewards,
            # 'total_time':total_time
            'model': server_model.state_dict(),
                'rnd_resume': rnd_idx + 1,
                'rnd_vl_loss': rnd_vl_loss,
                'rnd_vl_acc': rnd_vl_acc,
                'rnd_cl': rnd_cl,
                'reward':Q_values,
                'q_value':qq,
                'delay': delay,
                'latency':latency,
                'total_time': total_time
        }, root.joinpath('cpt.latest.pt'))

    dataset_fs.close()
    # Plot
    colormap = plt.cm.gist_ncar
    plt.rcParams.update({'font.size': 140})
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, args.epochs))))

    # all_rnd = list(range(len(rnd_cl)))
    # for idx, cl_res in tqdm(zip(all_rnd[-5:], rnd_cl[-5:]), total=len(rnd_cl[-5:])):
    #     plt_save = plots.joinpath(f'{str(idx + 1).zfill(4)}.jpg')
    #     legend_labels = []
    #     fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(200, 80))
    #     for cl in cl_res:
    #         legend_labels.append(cl['name'])
    #         print(np.arange(args.epochs), cl['accuracy'])
    #         ax_acc.plot(np.arange(args.epochs), cl['accuracy'], linewidth=7.0)
    #         ax_loss.plot(np.arange(args.epochs), cl['loss'], linewidth=7.0)
    #     ax_loss.legend(legend_labels)
    #     ax_loss.set_title("Loss")
    #     ax_acc.legend(legend_labels, )
    #     ax_acc.set_title("Accuracy")
    #     fig.savefig(plt_save)
    #     plt.clf()



    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(200, 80))
    ax_acc.plot(np.arange(args.rounds), rnd_vl_acc, linewidth=7.0)
    ax_acc.set_title("Accuracy")
    ax_loss.plot(np.arange(args.rounds), rnd_vl_loss, linewidth=7.0)
    ax_loss.set_title("Loss")
    fig.savefig(root.joinpath('server.jpg'))
    plt.clf()


if __name__ == '__main__':
    parser = ArgumentParser()

    # Path
    parser.add_argument('--save_root', help='Save root path', type=Path, default='./experimental/Run1_cifar1s0_81alphaResfedproxDDQN50round50client5selection')
    parser.add_argument('--ds-config', dest='ds_config', help='Dataset Configuration (.py)', type=Path,
                        default='./config/protonet-cifar.py')#'./config/protonet-cifar.py'
    parser.add_argument('--verbose', help='Log more details', action='store_true', default=True)
    parser.add_argument('--seed', help='Random State', type=int, default=2023)
    parser.add_argument('--device', help='Device Cuda, Cpu', type=str, default='cuda', choices=['cpu', 'cuda'])

    # Hyperparameter
    parser.add_argument('--batch', help='Batch size', type=int, default=4)
    parser.add_argument('--lr', help='Learning rate', type=float, default=1e-4)

    parser.add_argument('--epochs', help='No.Epochs', type=int, default=5)
    parser.add_argument('--rounds', help='No.Rounds', type=int, default=50)
    parser.add_argument('--loss', help='Loss function', type=str, default='MSE')
    parser.add_argument('--agg', help='Aggregation Strategy', type=str, choices=list(LocalUpdater.keys()),
                        default='FedProx')#FedAvg #FedPerAvg #FedProx

    # Optimizer
    parser.add_argument('--opt', help='Optimizer type', default='SGD', choices=list(RegisterOptim.keys()))
    parser.add_argument('--opt-weight-decay', '--opt_weight_decay', help='Optimizer weight decay', type=float,
                        default=0.)
    parser.add_argument('--opt-momentum', '--opt_momentum', help='Optimizer momentum', type=float, default=0.)
    parser.add_argument('--opt-prox-mu', '--opt_prox_mu', help='FedProx opt mu', type=float, default=0.)

    # Augmentation

    # Color
    parser.add_argument('--color', help='Enable ColorJitter', action='store_true', default=True)
    parser.add_argument('--color-br', dest='color_br', help='Color Brightness Beta', type=float, default=.4)
    parser.add_argument('--color-co', dest='color_co', help='Color Contrast Alpha', type=float, default=0.4)
    parser.add_argument('--color-st', dest='color_st', help='Color Stature', type=float, default=0.4)

    # Crop
    parser.add_argument('--img-size', dest='img_size', help='Image Size', type=int, default=80)
    parser.add_argument('--crop-x', '--crop_x', help='Crop on x axis', type=float, default=1.)
    parser.add_argument('--crop-y', '--crop_y', help='Crop on y axis', type=float, default=1.)

    # Horizontal Flip
    parser.add_argument('--h-flip', dest='h_flip', help='Enable Horizontal Flip', action='store_true', default=True)
    parser.add_argument('--h-flip-p', dest='h_flip_p', help='Horizontal Flip Probability', type=float, default=.5)

    # Vertical Flip
    parser.add_argument('--v-flip', dest='v_flip', help='Enable Vertical Flip', action='store_true', default=False)
    parser.add_argument('--v-flip-p', dest='v_flip_p', help='Vertical Flip Probability', type=float, default=.5)

    # Rotation
    parser.add_argument('--rotate', help='Enable rotation', action='store_true', default=False)
    parser.add_argument('--rotate-angle', dest='rotate_angle', help='Rotation Angle', type=float, default=25.)

    # Shear
    parser.add_argument('--shear', help='Enable shearing', action='store_true', default=False)

    # Dataset Split
    parser.add_argument('--iid', help='Enable iid', action='store_true', default=False)# True for IID setting
    parser.add_argument('--balance', help='Enable Balance for IID', action='store_true', default=True)
    parser.add_argument('--alpha', help='Dirichlet alpha', type=float, default=0.8)#1e-1

    # Client
    parser.add_argument('--client-select', '--client_select', help='Select Client', type=str, default='selected',
                        choices=['random', 'selected'])
    parser.add_argument('--client-frac', '--client_frac', help='The fraction of clients', type=float, default=0.2)
    parser.add_argument('--n-clients', '--n_clients', help='No.Clients', type=int, default=50)

    # Load lists containing clients' names, cores, frequencies, and bandwidths from respective pickle files
    with open("clients_info/names_list.pkl", "rb") as file:
        names = pickle.load(file)

    with open("clients_info/cores_list.pkl", "rb") as file:
        cores= pickle.load(file)

    with open("clients_info/frequency_list.pkl", "rb") as file:
        frequency_list = pickle.load(file)

    frequency = []
    min_freq = []
    max_freq = []
    
    for i in range(100):
        frequency.append(np.mean(frequency_list[i]))
        min_freq.append(np.min(frequency_list[i]))
        max_freq.append(np.max(frequency_list[i]))
    
    with open("clients_info/bandwidth_list.pkl", "rb") as file:
        bandwidth_list = pickle.load(file)
        
    bandwidth = []
    min_bw = []
    max_bw = []
    for i in range(100):
        bandwidth.append(np.mean(bandwidth_list[i]))
        min_bw.append(np.min(bandwidth_list[i]))
        max_bw.append(np.max(bandwidth_list[i]))


    main(names, cores, frequency, bandwidth, min_freq, max_freq, min_bw, max_bw, args=parser.parse_args())
