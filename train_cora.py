import torch
import torch_geometric as pyg
from gcn import GCN
from gin import GIN
# from gat import GAT
from tqdm.auto import tqdm
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device {device}')

def main():
    dataset = pyg.datasets.Planetoid(root='/tmp/Cora', name='Cora').to(device)
    d_in, d_lat, d_out = dataset.num_node_features, 64, dataset.num_classes
    gcn = GCN(d_in, d_lat, d_out).to(device)
    gin = GIN(d_in, d_lat, d_out).to(device)
    # gat = GAT(d_in, d_lat, d_out)
    acc_list = []
    for model in [gcn, gin, ]: # add gat
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        model.train()
        for epoch in tqdm(range(300)):
            optimizer.zero_grad()
            out = model(dataset)
            loss = F.nll_loss(out[dataset.train_mask], dataset.y[dataset.train_mask])
            loss.backward()
            optimizer.step()
        model.eval()
        pred = model(dataset).argmax(dim=1)
        correct = (pred[dataset.test_mask] == dataset.y[dataset.test_mask]).sum()
        acc = int(correct) / int(dataset.test_mask.sum())
        acc_list.append(acc)
    print(acc_list)


if(__name__ == '__main__'):
    main()