
# coding: utf-8

# In[1]:

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import ast
import time
import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

use_cuda = torch.cuda.is_available()


# In[2]:

seperator = " +++$+++ "
movie_conversations_path = "/home/tyler/data/text/cornell_movie_dialogs_corpus/movie_conversations.txt"
movie_lines = "/home/tyler/data/text/cornell_movie_dialogs_corpus/movie_lines_converted.txt"
MAX_LENGTH = 69 # including EOS tag


# In[3]:

def get_lines_to_words(path):
    """
    path: the path to the file with the words for lines
    returns a dictionary mapping from line number to the actually words
    """
    lines_dict = {}
    with open(path, "r") as f:
        for line in f:
            columns = line.split(seperator)
            lines_dict[columns[0]] = columns[-1]
    return lines_dict


# In[4]:

lines_dict = get_lines_to_words(movie_lines)


# In[5]:

def get_context_response_pairs(conversations_path, lines_dict):
    """
    conversations_path: the path to the conversation lines
    lines_dict: the dictionary mapping from lines to words
    returns: list of tuples (context, response)
    
    Code loops over all lines in a conversation taking the first as the 
    context and the next as the response. Thus, loop doesn't need to get to
    the last line.
    """
    with open(conversations_path, "r") as f:
        context_response_tuples = []
        for line in f:
            columns = line.split(seperator)
            convs = ast.literal_eval(columns[-1])
            for i, spoken_line in enumerate(convs[:-1]):
                context = lines_dict[convs[i]]
                response = lines_dict[convs[i+1]]
                context_response_tuples.append((context, response))
        return context_response_tuples


# In[6]:

context_response_tuples = get_context_response_pairs(movie_conversations_path,
                                                    lines_dict)


# In[7]:

def normalizeString(s):
    #put a space between punctuation, so not included in word
    s = s.strip().lower()
    s = re.sub(r"([.!?])", r" \1", s)
    #remove things that are not letters or punctuation
    s = re.sub(r"[^a-zA-Z.!?']+", r" ", s)
    return s

def clean_pairs(list_of_pairs, max_length=MAX_LENGTH-1):
    """
    list_of_pairs: list of context, response pairs as raw text
    max_length: max length of context or response. 99 percentile is 68
    returns list of tuples but each tuple is a list of tokenized words
    """
    pairs = []
    for pair in list_of_pairs:
        context = pair[0]
        response = pair[1]
        context_clean_tokens = normalizeString(context).split(" ")
        response_clean_tokens = normalizeString(response).split(" ")
        if len(context_clean_tokens) > max_length or len(response_clean_tokens) > max_length:
            continue
        pairs.append((context_clean_tokens, response_clean_tokens))
    return pairs


# In[8]:

clean_tuples = clean_pairs(context_response_tuples)


# In[9]:

class Words:
    def __init__(self):
        self.SOS_token = 0
        self.EOS_token = 1
        self.word2index = {}
        self.index2word = {self.SOS_token: "SOS", self.EOS_token: "EOS"}
        self.word2count = {}
        self.n_words = 2
        
    def __addArray(self, array):
        for word in array:
            self.__addWord(word)
            
    def addArrayOfTuples(self, array_of_tuples):
        for pair in array_of_tuples:
            self.__addArray(pair[0])
            self.__addArray(pair[1])
    
    def __addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# In[10]:

words = Words()
words.addArrayOfTuples(clean_tuples)


# In[11]:

class EncoderRNN(nn.Module):
    """
    Simple encoder network that embeds the character and then feeds through a GRU
    """
    def __init__(self, input_size, hidden_size, batch_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, self.batch_size, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden
    
    def initHidden(self):
        result = torch.zeros(1, self.batch_size, self.hidden_size)
        if use_cuda:
            return result.cuda()
        else:
            return result
        
class AttnDecoderRNN(nn.Module):
    """
    Attn Decoder
    1. Need max length because learning which input words to attend to
    And thus need to know the maximum number of words could attend to
    2. The attn_weights tell us how much to weight each input word - in this case French,
       In order to predict the english word.
    """
    def __init__(self, input_size, hidden_size, batch_size,
                 dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(hidden_size, hidden_size)
        # note input and output same size
        self.linear = nn.Linear(hidden_size, input_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.attn_layer = nn.Linear(2 * self.hidden_size, MAX_LENGTH)
        self.out_layer = nn.Linear(self.hidden_size, input_size)
        self.attn_combined_layer = nn.Linear(2 * self.hidden_size, self.hidden_size)
    
    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, self.batch_size, -1)
        embedded = self.dropout(embedded)
        attn = self.attn_layer(torch.cat((embedded[0], hidden[0]),dim=1))
        attn_weights = self.softmax(attn)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs) #shape: bx1xh
        attn_combined = torch.cat((embedded[0], attn_applied[:,0,:]), 1)
        attn_combined = self.relu(self.attn_combined_layer(attn_combined).unsqueeze(0))
        output, hidden = self.gru(attn_combined, hidden)
        output = self.softmax(self.out_layer(output[0]))
        return output, hidden, attn_weights
    
    def initHidden(self):
        result = torch.zeros(1, self.batch_size, self.hidden_size)
        if use_cuda:
            return result.cuda()
        else:
            return result        


# In[12]:

teacher_forcing_ratio = 0.5

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer,
         decoder_optimizer, criterion, batch_size, SOS_token=words.SOS_token, max_length=MAX_LENGTH):
    
    encoder_hidden = encoder.initHidden()
    loss = 0
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    input_length = input_variable.size()[1]
    target_length = target_variable.size()[1]
    
    encoder_outputs = torch.zeros((batch_size, MAX_LENGTH, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    
    # Here we are feeding in the english words to get the final hidden state 
    # for the decoder
    for i in range(input_length):
        encoder_ouput, encoder_hidden = encoder.forward(input_variable[:,i,:], encoder_hidden)
        encoder_outputs[:,i,:] = encoder_ouput[0]
        
    decoder_hidden = encoder_hidden
    decoder_input = torch.LongTensor([[SOS_token]]*batch_size)
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    # Here we take the final hidden state from the encoder
    # And feed it to decoder
    # We also give decoder the word to predict the next word starting with SOS token
    # If use teacher forcing then give it the truth, otherwise give it prediction
    if use_teacher_forcing:
        for i in range(target_length):
            decoder_output, decoder_hidden, attn_weights = decoder.forward(decoder_input, 
                                                                           decoder_hidden,
                                                                          encoder_outputs)

            loss += criterion(decoder_output, target_variable[:,i,0])
            decoder_input = target_variable[:,i,:]
            
    else:
        for i in range(target_length):
            decoder_output, decoder_hidden, attn_weights = decoder.forward(decoder_input, 
                                                                           decoder_hidden,
                                                                           encoder_outputs)
            loss += criterion(decoder_output, target_variable[:,i,0])
            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            print(decoder_input)

                
    loss.backward()
    
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.data[0] / target_length


# In[13]:



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))





def trainIters(encoder, decoder, data_loader, epochs, batch_size, print_every=1000, learning_rate=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
    
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    
    criterion = nn.NLLLoss()
    iter = 1
    n_iters = len(data_loader) * epochs
    for epoch in range(1, epochs + 1):
        for i_batch, sample_batched in enumerate(data_loader):
            
            
            loss = train(sample_batched[0], sample_batched[1], encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion,
                        batch_size)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))
                
            iter = iter + 1
            

# In[ ]:

class CustomDataset(Dataset):
    def __init__(self, data, words, max_length):
        self.data = data
        self.words = words
        self.max_length = max_length

    def __getitem__(self, index):
        row = self.data[index]
        training_pairs = self.tensorFromPair(row)
        return (training_pairs[0], training_pairs[1])
    
    def indexesFromSentence(self, sentence):
        return [self.words.word2index[word] for word in sentence]

    def tensorFromSentence(self, sentence):
        indexes = self.indexesFromSentence(sentence)
        indexes.append(self.words.EOS_token)
        # make it 1 column with number of rows equal to words in sentence
        result = torch.LongTensor(indexes).view(-1, 1)
        pad_amount = self.max_length - result.size(0)
        if pad_amount > 0:
            result = F.pad(result, (0,0,0,pad_amount), value=self.words.EOS_token).data
        result = result.cuda() if use_cuda else result
        return result

    def tensorFromPair(self, pair):
        input_variable = self.tensorFromSentence(pair[0])
        output_variable = self.tensorFromSentence(pair[1])
        return (input_variable, output_variable)
        
    def __len__(self):
        return len(self.data)

batch_size = 128
training_dataset = CustomDataset(clean_tuples, words, MAX_LENGTH) # add 1 for EOS
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True,
                                drop_last=True)


# In[ ]:

hidden_size = 512
encoder = EncoderRNN(words.n_words, hidden_size, batch_size)
decoder = AttnDecoderRNN(words.n_words, hidden_size, batch_size)
encoder.load_state_dict(torch.load("./models/chat_encoder.state"))
decoder.load_state_dict(torch.load("./models/chat_decoder.state"))

if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    
trainIters(encoder, decoder, training_dataloader, 10, batch_size, print_every=50)


# In[ ]:

torch.save(encoder.state_dict(), "./models/chat_encoder2.state")
torch.save(decoder.state_dict(), "./models/chat_decoder2.state")
print("Model Saved")
