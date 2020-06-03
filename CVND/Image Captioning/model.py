import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # Word embedding layer
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        
        #lstm layer
        self.lstm = nn.LSTM(input_size = embed_size,
                           hidden_size = hidden_size,
                           num_layers = num_layers,
                           batch_first = True)
        
        # linear layer
        self.hidden2vocab = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        
        embed_layer = self.word_embedding(captions[:, :-1])
        embed_layer = torch.cat((features.unsqueeze(1),embed_layer), 1) 
        lstm_output, _ = self.lstm(embed_layer)
        final = self.hidden2vocab(lstm_output)
        
        return final

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        res = []
        i = 0
        while i < max_len:
            lstm_output, states = self.lstm(inputs, states)
            output = self.hidden2vocab(lstm_output)
            output = output.squeeze(1)
            wordid = output.argmax(dim = 1)
            res.append(wordid.item())
            
            if wordid == 1:
                break
            
            inputs = self.word_embedding(wordid)
            inputs = inputs.unsqueeze(1)
            
            i += 1
        return res
            
        
        
        