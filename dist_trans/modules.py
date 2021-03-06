import torch
import torch.nn.functional as F 
import numpy as np 
import os


class NeuralNetwork():
    """
    Network class that wraps model related functions (e.g., training, evaluation, etc)
    """
    def __init__(self, model, criterion, optimizer, device):
        """
        Args:
            model: a deep neural network model (sent to device already)
            criterion: loss function
            optimizer: training optimizer
            device: training device
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device


    def train_model(self, data):
        """
        Train the model
        Args:
            data: training dataset generated by DataLoader
        Return batch-wise training loss
        """
        self.model.train()
        training_loss = 0

        for batch, sample in enumerate(data):
            img = sample[0]
            mask = sample[1]
            img = img.to(self.device, dtype=torch.float)
            mask = mask.to(self.device, dtype=torch.float)
            # Forward
            out = self.model(img)
            # Calculate loss
            if out.shape != mask.shape:
                shrink_sz = ((mask.shape[2]-out.shape[2])//2, (mask.shape[3]-out.shape[3])//2)
                mask = mask[:, :, shrink_sz[0]:-shrink_sz[0], shrink_sz[1]:-shrink_sz[1]]
            loss = self.criterion(out, mask)
            training_loss += loss.item()
            # Zero the parameter gradients
            self.optimizer.zero_grad()                    
            # Backward
            loss.backward()
            # Update weights
            self.optimizer.step()

        batch_loss = training_loss/(batch+1)
        return batch_loss


    def eval_model(self, data):
        """
        Evaluate the model
        Args:
            data: evaluation dataset generated by DataLoader
        Return batch-wise evaluation loss
        """
        self.model.eval()
        eval_loss = 0

        for batch, sample in enumerate(data):
            with torch.no_grad():  # Disable gradient computation
                img = sample[0]
                mask = sample[1]
                img = img.to(self.device, dtype=torch.float)
                mask = mask.to(self.device, dtype=torch.float)
                out = self.model(img)
                if out.shape != mask.shape:
                    shrink_sz = ((mask.shape[2]-out.shape[2])//2, (mask.shape[3]-out.shape[3])//2)
                    mask = mask[:, :, shrink_sz[0]:-shrink_sz[0], shrink_sz[1]:-shrink_sz[1]]
                loss = self.criterion(out, mask)
                eval_loss += loss.item()

        batch_loss = eval_loss/(batch+1)
        return batch_loss


    def save_model(self, path, epoch, entire=False):
        """
        Save the model to disk
        Args:
            path: directory to save the model
            epoch: epoch that model is saved
            entire: if save the entire model rather than just save the state_dict
        """
        if not os.path.exists(path):
            os.mkdir(path)
        if entire:
            torch.save(self.model, path+"/whole_model_epoch_{}.pt".format(epoch))
        else:
            torch.save({'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'criterion': self.criterion},
                        path+"/model_ckpt_{}.pt".format(epoch))
    

    def test_model(self, checkpoint, img, input_sz, step):
        """
        Test the model on new data
        Args:
            checkpoint: saved checkpoint
            img: testing data in [x, y] (network input is [batch, channel, x, y])
            input_sz: network input size in (x,y) 
            step: moving step in size (x,y)
        """

        ckpt = torch.load(checkpoint)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        gap = ((input_sz[0]-step[0])//2, (input_sz[1]-step[1])//2)
        
        out = np.zeros(img.shape, dtype=img.dtype)
        for row in range(0, img.shape[0]-input_sz[0], step[0]):
            for col in range(0, img.shape[1]-input_sz[1], step[1]):
                # Generate 
                patch_img = np.zeros((1, 1, input_sz[0], input_sz[1]), dtype=img.dtype)
                patch_img[0,0,:,:] = img[row:row+input_sz[0], col:col+input_sz[1]]
                patch_img = torch.from_numpy(patch_img).float()
                patch_img = patch_img.to(self.device)
                # Apply model
                patch_out = self.model(patch_img)
                patch_out = patch_out.cpu()
                patch_out = patch_out.detach().numpy()
                out[row+gap[0]:row+input_sz[0]-gap[0], col+gap[1]:col+input_sz[1]-gap[1]] = patch_out[0,0,:,:]
                    
        return out