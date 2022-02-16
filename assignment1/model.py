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
    
    for word in vocab.word2id:
        id = vocab[word]
        if word in emb_vectors:
            emb[id][:] = emb_vectors[word]
        else:
            if word == '<pad>' or word == '<unk>':
                continue
            
            unknown_words.append(word)
            print("Embedding does not exist for word = ",word)

    #print("length of unknown words",len(unknown_words))
    #if len(unknown_words)>0:
    #    print(unknown_words)
    #    sys.exit()
    
    unk_id = vocab['<unk>']
    emb[unk_id][:] = np.mean(emb,axis=0)
    
    for word in unknown_words:
        id = vocab[word]
        emb[id][:] = emb[unk_id]

    return emb
    #raise NotImplementedError()


class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.n_vocab = len(vocab)
        self.n_embed = args.emb_size
        self.define_model_parameters()
        self.init_model_parameters()

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            self.copy_embedding_from_numpy()

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        """
        self.embedding = torch.nn.Embedding(num_embeddings=self.n_vocab, embedding_dim=self.n_embed)
        self.fc1 = nn.Linear(self.n_embed, 300)
        #self.b1 = nn.BatchNorm1d(300)
        
        self.z1 = nn.ReLU()
        self.fc2 = nn.Linear(300, 300)
        self.z2 = nn.ReLU()
        
        #self.b2 = nn.BatchNorm1d(300)
        #self.fc2 = nn.Linear(self.n_embed, 50)
        #self.fc3 = nn.Linear(300, 5)
        #self.z3 = nn.LeakyReLU(0.2)
        #self.fc3 = nn.Linear(300, 5)
        #self.d1 = nn.Dropout(p=0.333)
        #self.d2 = nn.Dropout(p=0.333)
        #self.embdrop = nn.Dropout(p=0.333)

        self.fc3 = nn.Linear(300, 300)
        self.z3 = nn.ReLU()
        #self.d3 = nn.Dropout(p=0.333)

        self.final = nn.Linear(300,5)
        return
        #raise NotImplementedError()

    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                #torch.nn.init.xavier_uniform(m.weight)
                m.weight.data.uniform_(-0.08,0.08)
                #m.bias.data.fill_(0.1)
                #print("here")
                #m.bias.data.uniform_(-0.08,0.08)
        return
        #raise NotImplementedError()

    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        """
        #emb = load_embedding(self.vocab, self.args.emb_file, self.args.emb_size)
        
        #Uniform Random Initialization
        self.embedding.weight.data.uniform_(-0.08, 0.08)
        
    #    self.embedding.weight.data.copy_(emb)

        # = nn.Parameter(torch.from_numpy(emb).float())
        #self.embedding.weight = nn.Parameter(torch.from_numpy(emb).float())
        #self.embedding.weight.requires_grad=False
        return
        #raise NotImplementedError()

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

        
        #g1 = self.embedding(x) 
        #print("g1type",type(g1))
        #x (batch_size,length_of_sentence)
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #probs = torch.bernoulli(0.7*torch.ones(x.shape[0],x.shape[1]))
        #tot = probs.sum(1).unsqueeze(-1).expand(-1,x.shape[1])
        #probs = probs==1
        #probs = torch.where(tot!=0,probs,True).to(device)
        #(batch_size,length_of_sentence)
        #choose = torch.where(probs==1,x,0)
        #batch_size,length_of_sentence)
        #print("x shape embedding",x.shape)
        
        x = self.embedding(x)

        #(batchsize,len_sentence,emb_size)        
        x = torch.mean(x,1)
        #(batchsize,embsize)

        #emb_out = self.embedding(choose)
        #print(" type1",type(emb_out))
        #batch_size,length_of_sentence,embeddingsize)
        #emb_out = torch.sum(emb_out,1)
        #denom = probs.sum(1).unsqueeze(-1).expand(-1,emb_out.shape[1])
        #print(" type",type(emb_out))
        #print("denom",type(denom))
        #x = torch.div(emb_out,denom)
        #(batch_size,embedding_dim)
        #x = self.embdrop(x)

        x = self.fc1(x)
        x = self.z1(x)
        #x = self.d1(x)
        
        x = self.fc2(x)
        x = self.z2(x)
        #x = self.d2(x)

        #x = self.d2(x)
        #print("shape of x = ",x.shape)
        #x = self.z2(x)
        x = self.fc3(x)
        x = self.z3(x)
        #x = self.d3(x)

        x = self.final(x)
        return x


        raise NotImplementedError()
