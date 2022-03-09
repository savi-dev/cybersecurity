from os import makedirs, walk
from os.path import join, dirname, realpath

import time
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

from util import get_model_type
from model import AutoEncoder


class TensorDataset(Dataset):
    def __init__(self, root, tensor_dirs, time_name):
        self.data = []
        for tensor_dir in tensor_dirs:
            for root, _, files in walk(join(root, tensor_dir)):
                for f in files:
                    time_stamp = int(f[6:-3])
                    if get_model_type(datetime.utcfromtimestamp(time_stamp)) == time_name:
                        self.data.append(join(root, f))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.load(self.data[idx]).unsqueeze(0)


def train(model, train_loader, save_dir, num_epochs=30, learning_rate=1e-3):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()  # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    min_epoch_loss = None
    total_loss = []
    model.train()
    model.to(device)
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        for data in train_loader:
            tensor = data.to(device)
            recon = model(tensor)
            loss = criterion(recon, tensor)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss /= len(train_loader)
        
        print('Epoch:{}, Loss:{:.4f}, time:{:.3f}'.format(epoch + 1, epoch_loss, time.time() - epoch_start_time))
        total_loss.append(epoch_loss)

        if min_epoch_loss is None or epoch_loss < min_epoch_loss:
            min_epoch_loss = epoch_loss
            save_dict = {
                "model": model.state_dict(),
                "epoch": epoch,
                "min_epoch_loss": min_epoch_loss,
                "total_loss": total_loss,
                "optimizer": optimizer.state_dict()
            }
            print("Best model saved with loss:", min_epoch_loss, "at", epoch+1, "epoch")
            makedirs(save_dir, exist_ok=True)
            torch.save(save_dict, join(save_dir, "best_model.pt"))


def main(file_dir):
    # hyper parameter definition
    tensor_dirs = [f"june_week{x}_tensors_10min_port53413" for x in [2, 3, 4]]
    save_dir = "june_week_2to4_{}_10min_port53413_model"
    batch_size = 20
    num_epochs = 400
    learning_rate = 1e-3
    random_seed = 42
    for name in ["weekday_morning", "weekday_evening", "weekend_morning", "weekend_evening"]:
        print("Training for", name)
        # Fix the randomness
        torch.manual_seed(random_seed)

        model = AutoEncoder()  # initialize model
        dataset = TensorDataset(file_dir, tensor_dirs, name)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # load tensors

        train(model, train_loader, join(file_dir, save_dir.format(name)), num_epochs, learning_rate)

        
if __name__ == '__main__':
    file_dir_name = dirname(realpath(__file__))
    main(file_dir_name)
