from model import Model

from phoenix_datasets import PhoenixVideoTextDataset
from torch.utils.data import DataLoader

import torch.optim as optim

def main():
    dtrain = PhoenixVideoTextDataset(
        root="data/phoenix-2014-multisigner",
        split="train",
        p_drop=0.5,
        random_drop=True,
        random_crop=True,
        base_size=[256, 256]
        crop_size=[224, 224],
    )

    vocab = dtrain.vocab

    print("Vocab", vocab)

    dl = DataLoader(dtrain, collate_fn=dtrain.collate_fn)

    net = Model(vocab_size=len(vocab), 512)
    criterion = nn.CTCLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)

    for epoch in range(20):
        for batch in dl:
            video = batch["video"]
            label = batch["label"]
            signer = batch["signer"]

            optimizer.zero_grad()
            outputs = net([video, signer])
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            print('[%d] loss: %.3f' % (epoch + 1, loss.item()))
            
    print('Finsihed Training')
    PATH = './sign_language.pth'
    torch.save(net.state_dict(), PATH)

if __name__ == '__main__':
    main()