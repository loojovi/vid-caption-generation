from PIL import Image
import torch 
import numpy as np
import glob 
import shutil
import torch.nn as nn

# global variables
train_data_dir = 'data/train/train/'

class FrameDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, transform, feature_extractor):
        'Initialization'
        self.transform = transform
        self.labels = labels
        self.list_IDs = list_IDs
        self.feature_extractor = feature_extractor

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        path2frames = glob.glob(train_data_dir + ID + '/*.jpg')
        path2frames.sort() 
        # use subset of frames to train model - e.g. using frame 1, 4, 7, etc. in order to reduce training time
        # this method utilizes the temporal correlation between neighbor frames
        path2frames = path2frames[np.random.randint(0,3)::3]
        
        frames = []
        for p2f in path2frames:
            frame = Image.open(p2f)
            frame = self.transform(frame).unsqueeze(0)
            frame = self.feature_extractor(frame)
            frames.append(frame)
        
        # Load data and get label
        X = torch.stack(frames).squeeze()
        y = self.labels[ID]

        return X, y

# Defining network
class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.n_objects = 35
        self.n_relationships = 82
        self.in_features = 512
        self.hidden_features = 400
        self.num_layers = 2

        self.dropout = nn.Dropout(p=0.5)
        self.lstm_encoder = nn.LSTM(input_size = self.in_features, 
                                    hidden_size = self.hidden_features,
                                    num_layers = self.num_layers) 
        self.lstm_decoder = nn.LSTM(input_size = self.n_objects + self.n_relationships, 
                                    hidden_size = self.hidden_features, 
                                    num_layers = self.num_layers) 
        self.linear_object_1 = nn.Linear(self.hidden_features, self.n_objects)
        self.linear_relationship = nn.Linear(self.hidden_features, self.n_relationships)
        self.linear_object_2 = nn.Linear(self.hidden_features, self.n_objects)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]
        x = torch.transpose(x, 1,0)
        x = self.dropout(x)
        x, (h, c) = self.lstm_encoder(x) # h,c with shape (1,batch,hidden_features) to be fed into decoder

        h = self.dropout(h)
        c = self.dropout(c)
        
        in1 = torch.zeros((1, batch_size, self.n_objects + self.n_relationships)).to(device)
        out1, (h1, c1) = self.lstm_decoder(in1, (h, c))
        out1 = self.linear_object_1(out1).squeeze(dim=0)
        in2 = self.relu(out1)
       
        in2 = torch.cat((in2, torch.zeros(batch_size, self.n_relationships).to(device)), dim=1).unsqueeze(0)
        out2, (h2, c2) = self.lstm_decoder(in2, (h1, c1))
        out2 = self.linear_relationship(out2).squeeze(dim=0)
        in3 = self.relu(out2)
        
        in3 = torch.cat((torch.zeros(batch_size, self.n_objects).to(device), in3), dim=1).unsqueeze(0)
        out3, (h3, c3) = self.lstm_decoder(in3, (h2, c2))
        out3 = self.linear_object_2(out3).squeeze(dim=0)
            
        return out1, out2, out3 

# helper functions
def softmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps)

def cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector. 
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    p = softmax(X)
    # We use multidimensional array indexing to extract 
    # softmax probability of the correct label for each sample.
    # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
    log_likelihood = -np.log(p[range(m),y])
    loss = np.sum(log_likelihood) 
    return loss

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    
    # return model, optimizer, epoch value, min validation loss 
    train_loss_list = checkpoint['train_loss_list']
    valid_loss_list = checkpoint['valid_loss_list']
    train_loss_it = checkpoint['train_loss_it']

    return model, optimizer, checkpoint['epoch'], valid_loss_min.item(), train_loss_list, valid_loss_list, train_loss_it

def extract_test_images(transform, feature_extractor, path2frames):
    frames = []
    for p2f in path2frames:
        frame = Image.open(p2f)
        frame = test_transformer(frame).unsqueeze(0)
        frame = feature_extractor(frame)
        frames.append(frame)
    return torch.stack(frames).squeeze().unsqueeze(0)