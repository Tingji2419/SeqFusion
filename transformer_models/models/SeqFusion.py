import copy
import math
import torch
import torch.nn as nn


class Basic_Block(nn.Module):
    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Basic_Block, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.individual = individual
        self.channels = configs.enc_in
        test_pred_len = configs.test_pred_len
        self.c_block_nums = math.ceil(test_pred_len / self.pred_len) 

        self.blocks = nn.ModuleList()
        if configs.basic_model == 'DLinear':
            from transformer_models.models.DLinear import Model as Base_Model
        elif configs.basic_model == 'PatchTST':
            from transformer_models.models.PatchTST import Model as Base_Model
        else:
            raise NotImplementedError
        for i in range(self.c_block_nums):
            pre_configs = copy.deepcopy(configs)
            pre_configs.enc_in = 1
            basic_model = Base_Model(pre_configs, self.individual)
            self.blocks.append(basic_model)

    def load_pretrained(self, load_model_path):
        # print('Repeat Loading models:', self.c_block_nums, load_model_path)
        for block in self.blocks:
            block.load_state_dict(torch.load(load_model_path))

    def encoder(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        ret = torch.tensor([], device=x.device, dtype=x.dtype)
        for i, block in enumerate(self.blocks):
            # print('x shape:', x.shape)
            output = block(x, x_mark_enc, x_dec, x_mark_dec, mask)
            # print('output shape:', output.shape)
            x = torch.cat([x, output], dim=1)  # [B, seq + pred, D]
            x = x[:, -block.seq_len:, :]
            ret = torch.cat([ret, output], dim=1) if i != 0 else output
            # print('ret shape:', ret.shape)
        # assert 0
        return ret

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Encoder
        return self.encoder(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        return dec_out # [B, L, D]


class Model(nn.Module):
    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = individual
        self.channels = configs.enc_in
        self.test_pred_len = configs.test_pred_len
        self.c_block_nums = math.ceil(self.test_pred_len / self.pred_len)

        self.model_zoo = nn.ModuleList()
        self.zoo_size = len(configs.model_zoo)
        self.use_norm = configs.seq_norm
        self.norm_horizon = configs.norm_horizon

        self.basic_model = configs.basic_model
        self.err_predictor = configs.err_predictor

        for i in range(self.zoo_size):
            pre_configs = copy.deepcopy(configs)
            basic_model = Basic_Block(pre_configs, individual)
            self.model_zoo.append(basic_model)


    def load_pretrained(self, load_model_path):

        assert len(self.model_zoo) == len(load_model_path)
        for i, block in enumerate(self.model_zoo):
            block.load_pretrained(load_model_path[i])


    def encoder(self, x, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        pass


    def err_forecast(self, x_enc):
        if self.err_predictor == 'mean':
            pred = x_enc.mean(1).unsqueeze(1).repeat(1, self.pred_len * self.c_block_nums, 1)
        elif self.err_predictor == 'last':
            pred = x_enc[:,-1,:].unsqueeze(1).repeat(1, self.pred_len * self.c_block_nums, 1)
        elif self.err_predictor == 'seasonal':
            pred = x_enc[:,-7:,:].repeat(1, int(self.pred_len * self.c_block_nums / 7) + 1, 1)[:, :self.pred_len * self.c_block_nums, :]

        return pred

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        data, indices = x_enc
        k, batch_size = indices.shape
        input_len, channels = data.shape[1], data.shape[2]
        output_len = self.pred_len * self.c_block_nums

        # Flatten the indices to [k * batch_size]
        flattened_indices = indices.flatten()
        
        # Norm the seires
        if self.use_norm:
            if self.norm_horizon:
                means = torch.mean(data[:,-self.test_pred_len:,:], dim=1, keepdim=True).detach()
                stdev = torch.sqrt(torch.var(data[:,-self.test_pred_len:,:], dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            else:
                means = torch.mean(data, dim=1, keepdim=True).detach()
                stdev = torch.sqrt(torch.var(data, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            data = data - means
            data /= stdev

        # Repeat data, x_mark_enc, x_dec, and x_mark_dec to match the shape [k * batch_size, ...]
        repeated_data = data.repeat(k, 1, 1)
        repeated_x_mark_enc = x_mark_enc.repeat(k, 1, 1)
        repeated_x_dec = x_dec.repeat(k, 1, 1)
        repeated_x_mark_dec = x_mark_dec.repeat(k, 1, 1)

        # Collect all forecasts
        forecasts = torch.zeros([k * batch_size, output_len, channels]).to(data.device)
        for i, model in enumerate(self.model_zoo):
            # Create a mask for data that corresponds to this model
            model_mask = (flattened_indices == i)
            if model_mask.any():
                # Select data for the current model
                current_data = repeated_data[model_mask]
                current_marks_enc = repeated_x_mark_enc[model_mask]
                current_x_dec = repeated_x_dec[model_mask]
                current_marks_dec = repeated_x_mark_dec[model_mask]

                # Forecast using the current model
                model_output = model(current_data, current_marks_enc, current_x_dec, current_marks_dec)
                # Store output with expanded dimensions to match batch
                forecasts[model_mask] = model_output

        # predictions of err_predictor
        err_model_mask = (flattened_indices == -1)
        if err_model_mask.any():
            current_data = repeated_data[err_model_mask]
            model_output = self.err_forecast(current_data)
            forecasts[err_model_mask] = model_output
        # Reshape forecasts to [k, batch_size, output_len, channels]
        forecasts = forecasts.reshape(k, batch_size, output_len, channels)

        # Average the k outputs for each sample
        averaged_forecasts = forecasts.mean(dim=0)
        if self.use_norm:
            averaged_forecasts = averaged_forecasts * stdev + means
        return averaged_forecasts

    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask, dec_self_mask, dec_enc_mask)
        return dec_out # [B, L, D]

