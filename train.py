import torch
import numpy as np
import argparse
import time
from os.path import join as pjoin

import util
from engine import trainer
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--city', type=str, default='hangzhou', help='')
parser.add_argument('--tinterval', type=int, default=10, help='')

parser.add_argument('--device',type=int,default=1,help='')
parser.add_argument('--adjtype',type=str,default='transition',help='adj type')

parser.add_argument('--gcn_bool', type=int, default=1, help='whether to add graph convolution layer')
parser.add_argument('--aptonly', type=int, default=0, help='whether only adaptive adj')
parser.add_argument('--addaptadj', type=int, default=1, help='whether add adaptive adj')
parser.add_argument('--randomadj', type=int, default=0, help='whether random initialize adaptive adj')

parser.add_argument('--seq_length',type=int,default=3,help='output dimension?')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.003,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0006,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=200,help='')
parser.add_argument('--print_every',type=int,default=10,help='')
parser.add_argument('--seed',type=int,default=1111,help='random seed')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
args = parser.parse_args()

# data path and adjacency matrix path
outflow_path = f'./final_data/{args.city}/outflow{args.tinterval}'
adj_path = f'./final_data/{args.city}/{args.city}_adj_mx.csv'
save_path = f'./experiment/{args.city}/outflow{args.tinterval}'

log_path = pjoin('./result', 'train', args.city, f'{args.tinterval}')
logger = util.Logger(pjoin(log_path, 'test.log'))
logger.write(f'\nTesting configs: {args}')
# use tensorboard to draw the curves.
train_writer = SummaryWriter(pjoin('./result', 'train', args.city, f'{args.tinterval}'))
val_writer = SummaryWriter(pjoin('./result', 'val', args.city, f'{args.tinterval}'))

num_nodes = 165 if args.city == 'shenzhen' else 81


def main():
    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # load data
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    adj_mx = util.load_adj(adj_path, args.adjtype)
    dataloader = util.load_dataset(outflow_path, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    engine = trainer(scaler, args.in_dim, args.seq_length, num_nodes, args.nhid, args.dropout,
                     args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj, adjinit)

    logger.write("start training...")
    his_loss =[]
    val_time = []
    train_time = []
    for i in range(1, args.epochs+1):
        # learning rate schedule
        if i % 10 == 0:
            lr = max(0.000002, args.learning_rate * (0.9 ** (i // 10)))
            for g in engine.optimizer.param_groups:
                g['lr'] = lr

        # train
        train_mae = []
        train_rmse = []
        train_mape = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            # NOTE: B, T, V, F, F=2, but we noly need speed for label: y[:, 0, ...]
            metrics = engine.train(trainx, trainy[:,0,:,:])
            train_mae.append(metrics[0])
            train_rmse.append(metrics[1])
            train_mape.append(metrics[2])
        # log results of training set.
        mtrain_mae = np.mean(train_mae)
        mtrain_rmse = np.mean(train_rmse)
        mtrain_mape = np.mean(train_mape) * 100
        train_writer.add_scalar('train/mae', mtrain_mae, i)
        train_writer.add_scalar('train/rmse', mtrain_rmse, i)
        train_writer.add_scalar('train/mape', mtrain_mape, i)

        # validation
        with torch.no_grad():
            valid_mae = []
            valid_mape = []
            valid_rmse = []
            s1 = time.time()
            for _, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
                testx = torch.Tensor(x).to(device)
                testx = testx.transpose(1, 3)
                testy = torch.Tensor(y).to(device)
                testy = testy.transpose(1, 3)
                metrics = engine.eval(testx, testy[:,0,:,:])
                valid_mae.append(metrics[0])
                valid_rmse.append(metrics[1])
                valid_mape.append(metrics[2])
            # log results of validation set.
            s2 = time.time()
            val_time.append(s2-s1)
            mvalid_mae = np.mean(valid_mae)
            mvalid_mape = np.mean(valid_mape) * 100
            mvalid_rmse = np.mean(valid_rmse)
            his_loss.append(mvalid_mae)
            val_writer.add_scalar('val/mae', mvalid_mae, i)
            val_writer.add_scalar('val/rmse', mvalid_rmse, i)
            val_writer.add_scalar('val/mape', mvalid_mape, i)

        t2 = time.time()
        train_time.append(t2-t1)
        if i%args.print_every == 0:
            logger.write(f'Epoch: {i:03d}, MAE: {mtrain_mae:.2f}, RMSE: {mtrain_rmse:.2f}, MAPE: {mtrain_mape:.2f}, Valid MAE: {mvalid_mae:.2f}, RMSE: {mvalid_rmse:.2f}, MAPE: {mvalid_mape:.2f}')
        torch.save(engine.model.state_dict(), save_path+"_epoch_"+str(i)+"_"+str(round(mvalid_mae,2))+".pth")

    logger.write("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    # logger.write("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(save_path+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))

    logger.write("Training finished")
    logger.write(f"The valid loss on best model is {str(round(his_loss[bestid],4))}")

    # test
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    with torch.no_grad():
        t1 = time.time()
        for _, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1,3)
            preds = engine.model(testx).transpose(1,3)
            outputs.append(preds.squeeze())

        t2 = time.time()
        logger.write(f'Inference time: {t2-t1:.4f}')
        yhat = torch.cat(outputs, dim=0)
        yhat = yhat[:realy.size(0),...]

        # calculate metrics and save predictions
        preds = []
        reals = []
        logger.write('Step i, Test MAE, Test RMSE, Test MAPE')
        for i in range(args.seq_length):
            # prediction of step i
            pred = scaler.inverse_transform(yhat[:,:,i])
            real = realy[:,:,i]
            metrics = util.metric(pred.cpu().detach().numpy(), real.cpu().detach().numpy())
            logger.write(f'{metrics[0]:.2f}, {metrics[1]:.2f}, {metrics[2]*100:.2f}')

            preds.append(pred.tolist())
            reals.append(real.tolist())

    reals = np.array(reals)
    preds = np.array(preds)
    np.save(f'test_{args.city}_{args.tinterval}.npy', np.array(reals))
    np.save(f'test_{args.city}_{args.tinterval}.npy', np.array(preds))
    torch.save(engine.model.state_dict(), save_path+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    logger.write("Total time spent: {:.4f}".format(t2-t1))
