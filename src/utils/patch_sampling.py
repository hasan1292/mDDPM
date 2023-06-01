
import torch
import numpy as np
class BoxSampler():
    """
    Sample patches from an image
    """
    def __init__(self, cfg):
        self.patch_size = cfg.get('patch_size',16)
        self.stride = self.patch_size # default stride is patch size
        self.overlap = cfg.get('overlap',False) # default is no overlap
    
    

    def sample_single_box(self, image):
        """
        sample a random bounding box from an image
        Args:
            image (torch.tensor): 2D image of shape [batch, channel, height, width]
        Returns:
            bounding box (torch.tensor): bounding box of shape [batch, x_min, x_max, y_min, y_max]
        """
        # get image size
        batch_size, channel, height, width = image.shape

        # checks
        if isinstance(self.patch_size, int):
            self.patch_size = [self.patch_size, self.patch_size]
        if self.patch_size[1] > height or self.patch_size[0] > width:
            raise ValueError('Patch size is larger than image size')
        # sample random box
        x_min = torch.randint(0, width , (batch_size, 1))
        y_min = torch.randint(0, height , (batch_size, 1))
        x_max = x_min + self.patch_size[0]
        y_max = y_min + self.patch_size[1]
        
        # create bounding box
        box = torch.stack((x_min,y_min,x_max,y_max),dim=1)
        return box
    
    def sample_grid(self, image):
        """
        sample a grid of bounding boxes from an image
        Args:
            image (torch.tensor): 2D image of shape [batch, channel, height, width]
        Returns:
            bounding box (torch.tensor): bounding box of shape [batch, num_boxes, x_min, x_max, y_min, y_max]
        """
        # get image size
        batch_size, channel, height, width = image.shape

        # checks
        if isinstance(self.patch_size, int):
            self.patch_size = [self.patch_size, self.patch_size]
        if self.patch_size[1] > height or self.patch_size[0] > width:
            raise ValueError('Patch size is larger than image size')

        # sample random box
        x_min = torch.arange(0, width, self.stride).repeat(batch_size,1)
        y_min = torch.arange(0, height, self.stride).repeat(batch_size,1)

        if self.overlap : # adjust the grid to equally distribute the patches
            n_y = len(y_min[0,:])
            n_x = len(x_min[0,:])
            for i in range(n_y):
                y_min[:,i] = (i*((height-self.patch_size[1])/(np.int32(n_y-1))))
            for i in range(n_x):
                x_min[:,i] = (i*((width-self.patch_size[0])/(np.int32(n_x-1))))

        x_max = x_min + self.patch_size[0]
        y_max = y_min + self.patch_size[1]
        box = [] # list of boxes
        for i in range(y_min.shape[1]):
            for j in range(x_min.shape[1]):
                box.append(torch.stack((x_min[:,j],y_min[:,i],x_max[:,j],y_max[:,i]),dim=1))

        # create bounding box
        box = torch.stack((box),dim=1)
        return box


    def sample_grid_cut(self, image): # get grid without overlap..
        """
        sample a grid of bounding boxes from an image
        Args:
            image (torch.tensor): 2D image of shape [batch, channel, height, width]
        Returns:
            bounding box (torch.tensor): bounding box of shape [batch, num_boxes, x_min, x_max, y_min, y_max]
        """
        # get image size
        batch_size, channel, height, width = image.shape

        # checks
        if isinstance(self.patch_size, int):
            self.patch_size = [self.patch_size, self.patch_size]
        if self.patch_size[1] > height or self.patch_size[0] > width:
            raise ValueError('Patch size is larger than image size')

        # sample random box
        x_min = torch.arange(0, width, self.stride).repeat(batch_size,1)
        y_min = torch.arange(0, height, self.stride).repeat(batch_size,1)

        x_max = x_min + self.patch_size[0]
        y_max = y_min + self.patch_size[1]
        box = [] # list of boxes
        for i in range(y_min.shape[1]):
            for j in range(x_min.shape[1]):
                box.append(torch.stack((x_min[:,j],y_min[:,i],x_max[:,j],y_max[:,i]),dim=1))

        # create bounding box
        box = torch.stack((box),dim=1)
        return box
    
    
    
    
    
    
    
    
    
    
