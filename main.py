import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from pokemon import Pokemon
from resnet import ResNet18
batchsz = 32
lr = 1e-3
epochs = 10
device = torch.device('cuda')
torch.manual_seed(1234)
train_db = Pokemon('pokemon', 224, mode='train')
val_db = Pokemon('pokemon', 224, mode='val')
test_db = Pokemon('pokemon', 224, mode='test')
train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=4)
val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=2)
test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=2)
def evaluate(model, loader):
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct = torch.eq(pred, y).sum().float().item()
    return correct / total
def main():
    model = ResNet18(5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
        if epoch % 2 == 0:
            val_acc = evaluate(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best.mdl')
    print('best acc:',best_acc,'best epoch:', best_epoch)
    model.load_state_dict(torch.load('best.mdl'))
    print('loaded from ckpt')
    test_acc = evaluate(model, test_loader)
    print('test acc:', test_acc)
if __name__ == '__main__':
    main()
