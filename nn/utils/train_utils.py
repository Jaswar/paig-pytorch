import torch as th
import torch.nn.functional as F


def compute_loss(target, output,
                 recons_target, recons_out,
                 autoencoder_loss, pred_steps):
    loss = th.square(target - output).sum(dim=[2, 3, 4])
    pred_loss = th.mean(loss[:, pred_steps:])
    extrap_loss = th.mean(loss[:, pred_steps:])

    recons_loss = th.square(recons_target - recons_out).sum(dim=[2, 3, 4]).mean()

    train_loss = pred_loss
    if autoencoder_loss > 0.0:
        train_loss += autoencoder_loss * recons_loss

    eval_losses = [pred_loss, extrap_loss, recons_loss]
    return train_loss, eval_losses


def train_epoch(model, train_loader, optimizer, autoencoder_loss, device):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        output = model(batch)

        target = batch[:, model.input_steps:]
        recons_target = batch[:, :model.input_steps+model.pred_steps]
        train_loss, eval_losses = compute_loss(target, output,
                                               recons_target, model.recons_out,
                                               autoencoder_loss, model.pred_steps)

        optimizer.zero_grad()
        train_loss.backward()
        th.nn.utils.clip_grad_value_(model.parameters(), 1)
        optimizer.step()

        total_loss = train_loss.item()

    print(f'--train-- loss: {total_loss}')


def val_epoch(model, val_loader, autoencoder_loss, device):
    model.eval()
    total_loss = 0
    for batch_idx, batch in enumerate(val_loader):
        batch = batch.to(device)
        output = model(batch)

        target = batch[:, model.input_steps:]
        recons_target = batch[:, :model.input_steps + model.pred_steps]
        train_loss, eval_losses = compute_loss(target, output,
                                               recons_target, model.recons_out,
                                               autoencoder_loss, model.pred_steps)

        total_loss += train_loss.item()

    print(f'--val-- loss: {total_loss / len(val_loader)}')

