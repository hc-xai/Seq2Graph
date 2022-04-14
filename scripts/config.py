from easydict import EasyDict

config = EasyDict()

config.train_params=EasyDict()

config.encoder=EasyDict()
config.dualpurpose=EasyDict()
config.decoder=EasyDict()

##### Train params #####
config.train_params.n_epochs = 5
config.train_params.lr=0.01
config.train_params.batch_size=64
config.train_params.verbose=0
config.train_params.print_every=100
config.train_params.permute=True

##### Neural Net Arh ##### 

#Global
config.num_of_time_series=3
config.len_of_time_series=5

# Encoder
config.encoder.input_size=1
config.encoder.hidden_dim = 16
config.encoder.n_layers = 1
config.encoder.type_rnn='GRU'    

#DualPurpose 
config.dualpurpose.input_size=2*config.encoder.hidden_dim
config.dualpurpose.hidden_size_attn = 16
config.dualpurpose.rnn_hidden_size=16
config.dualpurpose.output_size=1
config.dualpurpose.n_layers=1
config.dualpurpose.type_rnn='GRU'
config.dualpurpose.return_timeseries=True

#Decoder
config.decoder.input_size=config.len_of_time_series
config.decoder.hidden_size_attn = 16
config.decoder.rnn_hidden_size=16
config.decoder.output_size=1
config.decoder.n_layers=1
config.decoder.type_rnn='GRU'
config.decoder.return_timeseries=True


