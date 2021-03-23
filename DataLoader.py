from phoenix_datasets import PhoenixVideoTextDataset
from torch.utils.data import DataLoader

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

    for batch in dl:
        video = batch["video"]
        label = batch["label"]
        signer = batch["signer"]

        assert len(video) == len(label)

        print(len(video))
        print(video[0].shape)
        print(label[0].shape)
        print(signer)

        break

if __name__ == '__main__':
    main()