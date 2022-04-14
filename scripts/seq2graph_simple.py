#Imports
import torch
from torch import nn

import torch.nn.functional as F


#############
###ENCODER###
#############
class Encoder(nn.Module):
    def __init__(self, config,DEVICE):
        super(Encoder, self).__init__()

        # Defining some parameters
        self.input_size=config.input_size
        self.hidden_dim = config.hidden_dim
        self.n_layers = config.n_layers
        self.type_rnn=config.type_rnn
        
        self.DEVICE=DEVICE


        #Defining the layers
        # RNN Layer
        if self.type_rnn=='GRU' or self.type_rnn=='G':
            self.rnn_layer = nn.GRU(self.input_size, self.hidden_dim, self.n_layers, batch_first=False,bidirectional = True)   
        elif self.type_rnn=='LSTM' or self.type_rnn=='L':
            self.rnn_layer = nn.LSTM(self.input_size, self.hidden_dim, self.n_layers, batch_first=False,bidirectional = True)  
        elif self.type_rnn=='RNN' or self.type_rnn=='R':
            self.rnn_layer = nn.RNN(self.input_size, self.hidden_dim, self.n_layers, batch_first=False,bidirectional = True)
        else: 
            print("Not given correct type of RNN. Given: %s. Using GRU instead."%selg.type_rnn)
            self.rnn_layer = nn.GRU(self.input_size, self.hidden_dim, self.n_layers, batch_first=False,bidirectional = True)   

        
    def forward(self, x):
        
        batch_size = x.size(1)

        #Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn_layer(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer    
        if self.type_rnn == 'LSTM' or self.type_rnn=='L':
            return out, hidden[0][-2:]
        else:
            return out, hidden[-2:]
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(2*self.n_layers, batch_size, self.hidden_dim).to(self.DEVICE) + 0.001
         # We'll send the tensor holding the hidden state to the device we specified earlier as well
        if self.type_rnn == 'LSTM' or self.type_rnn=='L':
            return (hidden,hidden)
        else:
            return hidden
        
        
##############
###ATENTION###
############## 
class Attention(nn.Module):
    def __init__(self,config,DEVICE):
        super(Attention, self).__init__()
        self.input_size = config.input_size
        self.hidden_size_attn = config.hidden_size_attn
        self.rnn_hidden_size = config.rnn_hidden_size
        self.output_size = config.output_size
        self.n_layers=config.n_layers
        self.type_rnn=config.type_rnn
        
        self.DEVICE=DEVICE

        
        
        self.attn = nn.Linear(self.input_size + self.output_size, self.hidden_size_attn)
        self.attn_activation=nn.Linear(self.hidden_size_attn, 1 ,bias=False)
        
        
        # RNN Layer
        if self.type_rnn=='GRU' or self.type_rnn=='G':
            self.rnn_layer = nn.GRU(self.input_size + self.output_size, self.rnn_hidden_size, self.n_layers, batch_first=False,bidirectional = False)   
        elif self.type_rnn=='LSTM' or self.type_rnn=='L':
            self.rnn_layer = nn.LSTM(self.input_size + self.output_size, self.rnn_hidden_size, self.n_layers, batch_first=False,bidirectional = False)  
        elif self.type_rnn=='RNN' or self.type_rnn=='R':
            self.rnn_layer = nn.RNN(self.input_size + self.output_size, self.rnn_hidden_size, self.n_layers, batch_first=False,bidirectional = False)
        else: 
            print("Not given correct type of RNN. Given: %s. Using GRU instead."%self.type_rnn)
            self.rnn_layer = nn.GRU(self.input_size + self.output_size, self.rnn_hidden_size, self.n_layers, batch_first=False,bidirectional = False)   
        
        
        self.activation=nn.Linear(self.rnn_hidden_size,self.output_size)
        
        self.alpha=None
        

    def forward(self, hidden,s_inner_hidden, encoder_outputs):

        seq_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(seq_len,1,1).transpose(0,1)
        encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
        #print(encoder_outputs.shape,H.shape)
        attn_energies = self.score(H,encoder_outputs) # compute attention score
        alpha_attn_energies= F.softmax(attn_energies,dim=1).unsqueeze(1) # normalize with softmax
        self.alpha=alpha_attn_energies
        c_temporal_context_vector=torch.bmm(alpha_attn_energies,encoder_outputs).transpose(0,1)
        s_output,s_inner_hidden=self.rnn_layer(torch.cat([hidden,c_temporal_context_vector],2),s_inner_hidden)
        s_output_to_activation=s_output.transpose(0,1) # [T,B,H] -> [B,T,H]
        #v_output=torch.tanh(self.activation(s_output_to_activation))
        #v_output=F.relu(self.activation(s_output_to_activation))
        v_output=self.activation(s_output_to_activation)
        #print(hidden.shape,s_output_to_activation.shape,c_temporal_context_vector.shape)
        return v_output.transpose(0,1),s_inner_hidden # [B,T,H] -> [T,B,H]

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # [B*T*2H]->[B*T*H]
        energy= self.attn_activation(energy) #[B,T,1]
        energy=energy.transpose(1,2)
        return energy.squeeze(1) #[B*T]
    
    def init_hidden(self, batch_size):
        

        
        s_inner_hidden = torch.zeros(self.n_layers, batch_size, self.rnn_hidden_size).to(self.DEVICE) +0.001
        v_output_hidden= torch.zeros(1, batch_size, self.output_size).to(self.DEVICE) + 0.001
        
        if self.type_rnn == 'LSTM' or self.type_rnn=='L':
            return v_output_hidden,(s_inner_hidden,s_inner_hidden)
        else:
            return v_output_hidden,s_inner_hidden
        
        
        
#################
###DUALPURPOSE###
#################     
class DualPurpose(nn.Module):
    def __init__(self, config,DEVICE):
        super(DualPurpose, self).__init__()
        self.config=config
        self.dualpurpose_return_timeseries=config.dualpurpose.return_timeseries
        self.dualpurpose_output_size=config.dualpurpose.output_size
        
        self.DEVICE=DEVICE
            
        self.encoder=Encoder(self.config.encoder,self.DEVICE)
        self.attention=Attention(self.config.dualpurpose,self.DEVICE)
        

        """
            The problem that alphas are not converging into correct values might be:
                1) Model propagates useful info with *hidden*, and then effect of attention can be leaked to neighbour points in time
                2) Maybe LSTM is needed (output and hidden do not have same value)
                3) Model is to powerfull, no need for attention
                4) Alpha not calculated correctly
                5) Attention only makes sense when all inputs are used
                6) Attention doesn't make sense when there is one dominate effect (look Dataset 1 vs Dataset 2)
        """
        
    def forward(self, model_input):
        
        current_batch_size=model_input.shape[1]
        
        encoder_output,encoder_hidden=self.encoder(model_input)
        
        if self.dualpurpose_return_timeseries:
            v_timeseries=torch.zeros(encoder_output.shape[0],encoder_output.shape[1],self.dualpurpose_output_size,requires_grad=True).to(self.DEVICE)
            v,s=self.attention.init_hidden(current_batch_size)
            for i in range(0,model_input.shape[0]):
                v,s=self.attention(v,s,encoder_output)
                v_timeseries[i,:,:]=v
            return v_timeseries
        else:        
            v,s=self.attention.init_hidden(current_batch_size)
            for i in range(0,model_input.shape[0]):
                v,s=self.attention(v,s,encoder_output)
            return v
        

        
#############
###DECODER###
#############  
class Decoder(nn.Module):
    def __init__(self, config,DEVICE):
        super(Decoder, self).__init__()
        self.decoder_output_size=config.decoder.output_size
        self.decoder_return_timeseries=config.decoder.return_timeseries
        
        self.DEVICE=DEVICE
            
        self.attention=Attention(config.decoder,self.DEVICE)
        
    def forward(self, model_input):
        
        current_batch_size=model_input.shape[1]
        
        self.beta=[]
                
        if self.decoder_return_timeseries:
            v_timeseries=torch.zeros(model_input.shape[0],model_input.shape[1],self.decoder_output_size,requires_grad=True).to(self.DEVICE)
            v,s=self.attention.init_hidden(current_batch_size)
            for i in range(0,model_input.shape[0]):
                v,s=self.attention(v,s,model_input)
                v_timeseries[i,:,:]=v
                self.beta.append(self.attention.alpha)
            self.beta=torch.stack(self.beta)
            return v_timeseries
        else:        
            v,s=self.attention.init_hidden(current_batch_size)
            for i in range(0,model_input.shape[0]):
                v,s=self.attention(v,s,model_input)
                self.beta.append(self.attention.alpha)
            self.beta=torch.stack(self.beta)
            return v
        
        
###############
###SEQ2GRAPH###
############### 
class Seq2Graph_no_decoder(nn.Module):
    def __init__(self,config,DEVICE):
        super(Seq2Graph_no_decoder, self).__init__()
        self.config=config
        self.num_of_timeseries=config.num_of_time_series
        self.DEVICE=DEVICE
        
        for i in range(self.num_of_timeseries):
            setattr(self, "dualpurpose_%d" % i,DualPurpose(self.config,self.DEVICE))
        
        #self.decoder=Decoder(self.config,self.DEVICE)
        
        self.alpha=None
        self.beta=None
        
        #self.final = nn.Linear(config.decoder.output_size,config.decoder.output_size)

        
        
    def forward(self, model_input):
        
        dp_list = torch.zeros(self.num_of_timeseries,model_input.shape[2],
                              model_input.shape[1],requires_grad=True).to(self.DEVICE)
        self.alpha=[]
        for i in range(self.num_of_timeseries):
            dualpurpose_output = getattr(self, "dualpurpose_%d" % i)(model_input[i])
            alpha = getattr(self, "dualpurpose_%d" % i).attention.alpha
            #print(dualpurpose_output.transpose(0,2).shape)
            dp_list[i,:]=dualpurpose_output.transpose(0,2)
            self.alpha.append(alpha)
        self.alpha=torch.stack(self.alpha)
        
        #decoder_input=torch.cat(dp_list,dim=0)
        #print(decoder_input.shape)
        #decoder_input=dp_list
        #output=self.decoder(decoder_input)
        #output=self.final(output)
        
        #self.beta=self.decoder.beta
        return dp_list#output.transpose(0,2).squeeze(0).to(self.DEVICE)
        
class Seq2Graph_test(nn.Module):
    def __init__(self,config,DEVICE):
        super(Seq2Graph_test, self).__init__()
        self.config=config
        self.num_of_timeseries=config.num_of_time_series
        self.len_of_timeseries=config.len_of_time_series
        self.DEVICE=DEVICE
        
        for i in range(self.num_of_timeseries):
            setattr(self, "dualpurpose_%d" % i,DualPurpose(self.config,self.DEVICE))
        
        #self.decoder=Decoder(self.config,self.DEVICE)
        self.decoder=nn.Linear(self.num_of_timeseries*self.len_of_timeseries,
                               self.num_of_timeseries*config.decoder.output_size )
        
        self.alpha=None
        self.beta=None
        
        #self.final = nn.Linear(config.decoder.output_size,config.decoder.output_size)

        
        
    def forward(self, model_input):
        
        dp_list = torch.zeros(self.num_of_timeseries,model_input.shape[2],
                              model_input.shape[1],requires_grad=True).to(self.DEVICE)
        self.alpha=[]
        for i in range(self.num_of_timeseries):
            dualpurpose_output = getattr(self, "dualpurpose_%d" % i)(model_input[i])
            alpha = getattr(self, "dualpurpose_%d" % i).attention.alpha
            #print(dualpurpose_output.transpose(0,2).shape)
            dp_list[i,:]=dualpurpose_output.transpose(0,2)
            self.alpha.append(alpha)
        self.alpha=torch.stack(self.alpha)
        
        #decoder_input=torch.cat(dp_list,dim=0)
        #print(decoder_input.shape)
        decoder_input=dp_list.transpose(0,1).flatten(1)
        output=self.decoder(decoder_input)
        #output=self.final(output)
        
        #self.beta=self.decoder.beta
        return output#.transpose(0,2).squeeze(0).to(self.DEVICE)
        
        
class Seq2Graph(nn.Module):
    def __init__(self,config,DEVICE):
        super(Seq2Graph, self).__init__()
        self.config=config
        self.num_of_timeseries=config.num_of_time_series
        self.DEVICE=DEVICE
        
        for i in range(self.num_of_timeseries):
            setattr(self, "dualpurpose_%d" % i,DualPurpose(self.config,self.DEVICE))
        
        self.decoder=Decoder(self.config,self.DEVICE)
        
        self.alpha=None
        self.beta=None
        
        self.final = nn.Linear(self.num_of_timeseries*config.decoder.output_size, self.num_of_timeseries*config.decoder.output_size)

        
        
    def forward(self, model_input):
        
        dp_list = torch.zeros(self.num_of_timeseries,model_input.shape[2],
                              model_input.shape[1],requires_grad=True).to(self.DEVICE)
        self.alpha=[]
        for i in range(self.num_of_timeseries):
            dualpurpose_output = getattr(self, "dualpurpose_%d" % i)(model_input[i])
            alpha = getattr(self, "dualpurpose_%d" % i).attention.alpha
            #print(dualpurpose_output.transpose(0,2).shape)
            dp_list[i,:]=dualpurpose_output.transpose(0,2)
            self.alpha.append(alpha)
        self.alpha=torch.stack(self.alpha)
        
        #decoder_input=torch.cat(dp_list,dim=0)
        decoder_input=dp_list
        #print(decoder_input.shape)
        output=self.decoder(decoder_input)
        #print(output.shape)
        output=self.final(output.transpose(0,2).squeeze(0))
        
        self.beta=self.decoder.beta
        return output.to(self.DEVICE)
        
