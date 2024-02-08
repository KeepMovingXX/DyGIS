import torch
from models import LogisticRegression
from utils import get_acc_score

def link_detection_evaluation(train_embedding, test_embedding, labels_all, device, z_dim, h_dim, numclass, seq_len, spilt_len):
    with torch.no_grad():
        train_embedding = torch.stack(train_embedding)
        test_embedding = torch.stack(test_embedding)
        labels_all = torch.tensor(labels_all,dtype=torch.long).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    finetune_model = LogisticRegression(z_dim, z_dim, numclass, 0.5)
    finetune_model = finetune_model.to(device)
    optimizer1 = torch.optim.Adam(finetune_model.parameters(), lr=0.1)

    for epo in range(300):
        nc_loss = 0
        finetune_model.train()
        optimizer1.zero_grad()
        for t in range(labels_all.shape[1]- spilt_len):
            labels_t = labels_all[:,t]
            logits = finetune_model(train_embedding[t])
            nc_loss += criterion(logits, labels_t)
        nc_loss.backward()
        optimizer1.step()
        print( 'epoch: ', epo, nc_loss.item())
        finetune_model.eval()
        with torch.no_grad():
            test_pred_list = []
            test_label_list = []
            for tt in range(spilt_len):
                test_logits = finetune_model(test_embedding[tt])
                test_pred = test_logits.argmax(dim=1)
                test_pred_list.append(test_pred)
            test_labels = labels_all[:,(seq_len - spilt_len):].t()
            acc = get_acc_score(test_labels,torch.stack(test_pred_list))
            print(acc)
    return acc
