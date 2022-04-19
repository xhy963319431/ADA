import os
import time
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
# from tensorboardX import SummaryWriter

import model
import evaluate
import data_utils
from loss import loss_function,loss_function_new

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', 
	type = str,
	help = 'dataset used for training, options: ml1m,nf',
	default = 'ml1m')
parser.add_argument('--model', 
	type = str,
	help = 'model used for training. options: GMF, NeuMF-end',
	default = 'GMF')
parser.add_argument('--drop_rate', 
	type = float,
	help = 'drop rate',
	default = 0.2)
parser.add_argument('--num_gradual', 
	type = int, 
	default = 30000,
	help='how many epochs to linearly increase drop_rate')
parser.add_argument('--exponent', 
	type = float, 
	default = 1, 
	help='exponent of the drop rate {0.5, 1, 2}')
parser.add_argument("--lr", 
	type=float, 
	default=0.001, 
	help="learning rate")
parser.add_argument("--dropout", 
	type=float,
	default=0.0,  
	help="dropout rate")
parser.add_argument("--batch_size", 
	type=int, 
	default=1024, 
	help="batch size for training")
parser.add_argument("--epochs", 
	type=int,
	default=10,
	help="training epoches")
parser.add_argument("--factor_num", 
	type=int,
	default=32, 
	help="predictive factors numbers in the model")
parser.add_argument("--num_layers", 
	type=int,
	default=3, 
	help="number of layers in MLP model")
parser.add_argument("--num_ng", 
	type=int,
	default=1, 
	help="sample negative items for training")
parser.add_argument("--out", 
	default=True,
	help="save model or not")
parser.add_argument("--gpu", 
	type=str,
	default="1",
	help="gpu card ID")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True

torch.manual_seed(2019) # cpu
torch.cuda.manual_seed(2019) #gpu
np.random.seed(2019) #numpy
random.seed(2019) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

def worker_init_fn(worker_id):
    np.random.seed(2019 + worker_id)

data_path = '../data/{}/'.format(args.dataset)
model_path = './models/{}/'.format(args.dataset)
print("arguments: %s " %(args))
print("config model", args.model)
print("config data path", data_path)
print("config model path", model_path)

############################## PREPARE DATASET ##########################

train_data, user_pos, user_num ,item_num, train_mat, train_data_time = data_utils.load_all(args.dataset, data_path)

# construct the train and test datasets
train_dataset = data_utils.NCFData(
		train_data, item_num, train_mat, args.num_ng, 0, train_data_time)

train_loader = data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

print("data loaded! user_num:{}, item_num:{} train_data_len:{}".format(user_num, item_num, len(train_data)))
print("########################## Data denoising,please wait. #########################")
########################### CREATE MODEL #################################
if args.model == 'NeuMF-pre': # pre-training. Not used in our work.
	GMF_model_path = model_path + 'GMF.pth'
	MLP_model_path = model_path + 'MLP.pth'
	NeuMF_model_path = model_path + 'NeuMF.pth'
	assert os.path.exists(GMF_model_path), 'lack of GMF model'
	assert os.path.exists(MLP_model_path), 'lack of MLP model'
	GMF_model = torch.load(GMF_model_path)
	MLP_model = torch.load(MLP_model_path)
else:
	GMF_model = None
	MLP_model = None

model = model.NCF(user_num, item_num, args.factor_num, args.num_layers, 
						args.dropout, args.model, GMF_model, MLP_model)

model.cuda()

if args.model == 'NeuMF-pre':
	optimizer = optim.SGD(model.parameters(), lr=args.lr)
else:
	optimizer = optim.Adam(model.parameters(), lr=args.lr)


# define drop rate schedule
def drop_rate_schedule(iteration):

	drop_rate = np.linspace(0, args.drop_rate**args.exponent, args.num_gradual)
	if iteration < args.num_gradual:
		return drop_rate[iteration]
	else:
		return args.drop_rate

########################### TRAINING #####################################
count, best_hr = 0, 0
best_loss = 1e9
goal_data = []
for epoch in range(args.epochs):
		
	model.train() # Enable dropout (if have).
	train_loader.dataset.ng_sample()

	for user, item, label ,time in train_loader:
		user = user.cuda()
		item = item.cuda()
		label = label.float().cuda()
		time = time
		model.zero_grad()
		prediction = model(user, item)
		if(epoch==9):
			loss,ind_update = loss_function_new(prediction, label, drop_rate_schedule(count))
			for i in ind_update:
				if(int(label[i])==1):
					goal_data.append([user[i],item[i],time[i]])
		else:
			loss = loss_function(prediction, label, drop_rate_schedule(count))
		loss.backward()
		optimizer.step()

		count += 1

print("############################## Training End. ##############################")
# Output the denoisied data.
filePath = r"../data/output/{}_example.txt".format(args.dataset)
with open(filePath, "w+", encoding='utf-8') as d:
    file_data = "," + "uid" + "," + "iid" + "," + "ts" + "\n"
    for i,x in enumerate(goal_data):
        file_data += str(i) +','+ str(int(x[0])) + "," + str(int(x[1])) + "," + str((x[2])) + "\n"
    d.write(file_data)
d.close()

