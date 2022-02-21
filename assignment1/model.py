import torch
import torch.nn as nn
import zipfile
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        ckpt = torch.load(path)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])


def load_embedding(vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """
    v_size = len(vocab)
    emb_vectors = {}
    emb = np.zeros((v_size,emb_size),dtype=np.float64)
    unknown_words = []

    with open(emb_file, 'r') as f:
        for line in f:
            ln = line.split()
            val = np.array(ln[1:], dtype=np.float64)
            emb_vectors[ln[0]] = val
    print("size of vocab = ",len(vocab)) 
    for word in vocab.word2id:
        id = vocab[word]
        if word in emb_vectors:
            emb[id][:] = emb_vectors[word]
        else:
            if word == '<pad>' or word == '<unk>':
                continue
            
            unknown_words.append(word)
            #print("Embedding does not exist for word = ",word)

    #print("length of unknown words",len(unknown_words))
    #if len(unknown_words)>0:
    #    print(unknown_words)
    #    sys.exit()
    
    unk_id = vocab['<unk>']
    emb[unk_id][:] = np.mean(emb,axis=0)
    print("No of unknown words = ",len(unknown_words))
    for word in unknown_words:
        id = vocab[word]
        emb[id][:] = emb[unk_id]

    return emb


class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.n_vocab = len(vocab)
        self.n_embed = args.emb_size
        self.define_model_parameters()
        self.init_model_parameters()

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None and args.emb_file!="None":
            #print('here')
            self.copy_embedding_from_numpy()

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        """
        self.embedding = torch.nn.Embedding(num_embeddings=self.n_vocab, embedding_dim=self.n_embed,padding_idx=self.vocab['<pad>'])
        if self.args.emb_drop!=0:
            self.embdrop = nn.Dropout(p=self.args.emb_drop)
        
        self.fc1 = nn.Linear(self.n_embed, self.args.hid_size)
        self.z1 = nn.LeakyReLU(0.2)
        self.d1 = nn.Dropout(p=self.args.hid_drop)
        #self.b1 = nn.BatchNorm1d(self.n_embed)

        self.fc2 = nn.Linear(self.args.hid_size,self.args.hid_size)
        self.z2 = nn.LeakyReLU(0.2)
        self.d2 = nn.Dropout(p=self.args.hid_drop)
        #self.b2 = nn.BatchNorm1d(300)
        
        self.fc3 = nn.Linear(self.args.hid_size,self.args.hid_size)
        self.z3 = nn.LeakyReLU(0.2)
        #self.b2 = nn.BatchNorm1d(300)
        self.d3 = nn.Dropout(p=self.args.hid_drop)
    

        self.final = nn.Linear(self.args.hid_size,self.tag_size)
        return

    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        """
        
        for name, p in self.named_parameters():
            p.data.uniform_(-0.08,0.08)
            #torch.nn.init.xavier_uniform(p.data)
        
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                #torch.nn.init.xavier_uniform(m.weight)
                m.weight.data.uniform_(-0.08,0.08)
                #m.bias.data.fill_(0.1)
                #print("here")
                m.bias.data.uniform_(-0.08,0.08)
        """
        return
        

    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        """
        emb = load_embedding(self.vocab, self.args.emb_file, self.args.emb_size)
        self.embedding.weight = nn.Parameter(torch.from_numpy(emb).float())
        #self.embedding.weight.requires_grad=False
        
        return

    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)  
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """
        
        #IMPLEMENTATION OF WORD DROPOUT

        if self.args.word_drop!=0:

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            probs = torch.bernoulli((1-self.args.word_drop)*torch.ones(x.shape[0],x.shape[1]))
            probs = probs==1
            tot = probs.sum(1).unsqueeze(-1).expand(-1,x.shape[1])
            probs = torch.where(tot!=0,probs,True).to(device)
            x = torch.where(probs==1,x,0)

        out = torch.count_nonzero(x,dim=1).to(torch.float)
        out = out.unsqueeze(-1).expand(-1,self.args.emb_size)+1e-5
 
        x = self.embedding(x)
        x = torch.sum(x,1)
        x = torch.div(x,out)

        if self.args.emb_drop!=0:
            x = self.embdrop(x)
        
        x = self.fc1(x)
        x = self.z1(x)
        
        #x = self.b1(x)
        
        x = self.d1(x)
        
        x = self.fc2(x)
        x = self.z2(x)
        
        #x = self.b2(x)
        x = self.d2(x)

        #x = self.fc3(x)
        #x = self.z3(x)
        #x = self.d3(x)

        x = self.final(x)
        return x






class LSTMModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(LSTMModel, self).__init__(args, vocab, tag_size)
        self.n_vocab = len(vocab)
        self.n_embed = args.emb_size
        self.define_model_parameters()
        self.init_model_parameters()

        print("Using model LSTM")
        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None and args.emb_file!="None":
            #print('here')
            self.copy_embedding_from_numpy()

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        """
        self.embedding = torch.nn.Embedding(num_embeddings=self.n_vocab, embedding_dim=self.n_embed,padding_idx=self.vocab['<pad>'])
        if self.args.emb_drop!=0:
            self.embdrop = nn.Dropout(p=self.args.emb_drop)
        
        self.fc1 = nn.Linear(self.n_embed*2, self.args.hid_size)
        self.z1 = nn.LeakyReLU(0.2)
        self.d1 = nn.Dropout(p=self.args.hid_drop)
        #self.b1 = nn.BatchNorm1d(self.n_embed)

        self.fc2 = nn.Linear(self.args.hid_size,self.args.hid_size)
        self.z2 = nn.LeakyReLU(0.2)
        self.d2 = nn.Dropout(p=self.args.hid_drop)
        #self.b2 = nn.BatchNorm1d(300)
        
        self.fc3 = nn.Linear(self.args.hid_size,self.args.hid_size)
        self.z3 = nn.LeakyReLU(0.2)
        #self.b2 = nn.BatchNorm1d(300)
        self.d3 = nn.Dropout(p=self.args.hid_drop)
    
        self.lstm = torch.nn.LSTM(300,300,2,batch_first=True,bidirectional=True,dropout=0.2)

        self.final = nn.Linear(self.args.hid_size,self.tag_size)
        return

    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        """
        
        for name, p in self.named_parameters():
            p.data.uniform_(-0.08,0.08)
            #torch.nn.init.xavier_uniform(p.data)
        
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                #torch.nn.init.xavier_uniform(m.weight)
                m.weight.data.uniform_(-0.08,0.08)
                #m.bias.data.fill_(0.1)
                #print("here")
                m.bias.data.uniform_(-0.08,0.08)
        """
        return
        

    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        """
        emb = load_embedding(self.vocab, self.args.emb_file, self.args.emb_size)
        self.embedding.weight = nn.Parameter(torch.from_numpy(emb).float())
        #self.embedding.weight.requires_grad=False
        
        return

    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)  
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """

        x = self.embedding(x)
        x,_ = self.lstm(x)
        
        x = torch.mean(x,1)

        x = self.fc1(x)
        x = self.z1(x)
        
        #x = self.b1(x)
        
        x = self.d1(x)
        
        #x = self.fc2(x)
        #x = self.z2(x)
        
        #x = self.b2(x)
        #x = self.d2(x)

        #x = self.fc3(x)
        #x = self.z3(x)
        #x = self.d3(x)

        x = self.final(x)
        return x


