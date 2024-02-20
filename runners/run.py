import torch as th
from nn.utils.train_utils import train_epoch, val_epoch
from nn.datasets.iterators import PAIGDataset
from nn.network.physics_net import PhysicsNet


if __name__ == '__main__':
    epochs = 500
    autoencoder_loss = 3.0

    device = 'cuda' if th.cuda.is_available() else 'cpu'

    npz_file = '../data/datasets/spring_color/color_spring_vx8_vy8_sl12_r2_k4_e6.npz'

    train_dataset = PAIGDataset(npz_file, split='train')
    train_loader = th.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)

    val_dataset = PAIGDataset(npz_file, split='val')
    val_loader = th.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=True)

    model = PhysicsNet('spring_color', train_dataset[0][0].shape,
                       seq_len=12, input_steps=4, pred_steps=6, device=device)
    optimizer = th.optim.RMSprop(model.parameters(), lr=3e-4, alpha=0.9, eps=1e-10)
    # optimizer = th.optim.SGD(model.parameters(), lr=3e-4)
    scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.75 * epochs), gamma=0.2)
    model = model.to(device)

    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer, autoencoder_loss, device)
        val_epoch(model, val_loader, autoencoder_loss, device)

        scheduler.step()


