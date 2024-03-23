from get_args import get_args
from utils import load_data,accuracy
from models import GCN
import torch
import torch.optim as optim
import time

def model_train():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    features,labels,adj,train_index,valid_index,test_index = load_data()

    model = GCN(args.input_dim,args.hidden_dim,args.output_dim,args.num_layers,args.dropout).to(args.device)

    optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    criterion = torch.nn.NLLLoss()

    features = features.to(args.device)
    labels = labels.to(args.device)
    adj = adj.to(args.device)
    train_index = train_index.to(args.device)
    valid_index = valid_index.to(args.device)
    test_index = test_index.to(args.device)

    best_model = None
    best_acc = 0

    start = time.perf_counter()
    for epoch in range(1,args.epochs+1):
        train_loss,train_acc = train(model,features,labels,adj,train_index,optimizer,criterion)
        valid_loss,valid_acc = eval(model,features,labels,adj,valid_index,criterion)
        if valid_acc > best_acc:
            best_model = model
            best_acc = valid_acc
        print(f'Epoch:{epoch:03d} train_loss:{train_loss:.4f} train_acc:{100*train_acc:.2f}% '
              f'valid_loss:{valid_loss:.4f} test_valid:{100*valid_acc:.2f}%')

    end = time.perf_counter()
    print('Optimization Finished!')
    print("Total time elapsed：{:.4f}s".format(end - start))
    print(f'best_valid_acc:{100*best_acc:.2f}%')
    test_loss,test_acc = eval(best_model,features,labels,adj,test_index,criterion)
    print(f'Test set result:\n'
          f'test_loss:{test_loss:.4f}\n'
          f'test_acc:{100*test_acc:.2f}%\n')

def train(model,features,labels,adj,train_index,optimizer,criterion):
    model.train()
    output = model(features,adj)
    train_loss = criterion(output[train_index],labels[train_index])
    train_acc = accuracy(output[train_index],labels[train_index])
    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return train_loss.item(),train_acc

def eval(model,features,labels,adj,data_index,criterion):
    model.eval()
    with torch.no_grad():
        output = model(features,adj)
        test_loss = criterion(output[data_index],labels[data_index])
        test_acc = accuracy(output[data_index],labels[data_index])
    return test_loss,test_acc

if __name__ == '__main__':
    model_train()
