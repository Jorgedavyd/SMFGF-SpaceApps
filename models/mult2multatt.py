from models.utils import *
from models.df_models import *
import torch.nn.functional as F
from sklearn.metrics import r2_score
from models.base import *
def r2(y_pred, y_true):
    # Calculate the mean of the true values
    mean = torch.mean(y_true)

    # Calculate the total sum of squares
    ss_total = torch.sum((y_true - mean) ** 2)

    # Calculate the residual sum of squares
    ss_residual = torch.sum((y_true - y_pred) ** 2)

    # Calculate R2 score
    r2 = 1 - (ss_residual / ss_total)
    
    return r2

def threshold_predictions(predictions, threshold=-2.6):
    # Convert regression predictions to binary (0 or 1) based on the threshold
    binary_predictions = (predictions <= threshold).float()
    return binary_predictions

def acc(predictions, targets, threshold=-2.6):
    binary_predictions = threshold_predictions(predictions, threshold)
    correct = (binary_predictions == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    return accuracy

def compute_precision(predictions, targets, threshold=-2.6):
    binary_predictions = threshold_predictions(predictions, threshold)
    binary_targets = threshold_predictions(targets, threshold)
    true_positives = ((binary_predictions == 1) & (binary_targets == 1)).sum()
    false_positives = ((binary_predictions == 1) & (binary_targets == 0)).sum()
    try:
        precision = true_positives / (true_positives + false_positives)
    except ZeroDivisionError:   
        precision = torch.tensor([0])
    return precision

def compute_recall(predictions, targets, threshold=-2.6):
    binary_predictions = threshold_predictions(predictions, threshold)
    binary_targets = threshold_predictions(targets, threshold)
    true_positives = ((binary_predictions == 1) & (binary_targets == 1)).sum()
    false_negatives = ((binary_predictions == 0) & (binary_targets == 1)).sum()
    recall = true_positives / (true_positives + false_negatives)
    return recall

def compute_all(predictions, targets, threshold=-2.6):
    precision = compute_precision(predictions, targets, threshold)
    recall = compute_recall(predictions, targets, threshold)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score

class MultiHead2MultiHeadBase(GeoBase):
    def training_step(self, batch, weights:list, encoder_forcing):
        self.encoder_forcing = encoder_forcing
        if encoder_forcing:
            if isinstance(self, MultiHeaded2MultiheadAttentionLSTM):
                #instantiate loss
                loss = 0
                alpha_fc, alpha_mg, alpha_out, alpha_main = weights
                fc1, mg1, f1m, m1m, target = batch #decompose batch
                #encoder forcing for faraday cup
                out_fc_L1, (hn_fc_L1, cn_fc_L1) = self.encoder_fc(fc1)
                out_fc_L2, (hn_fc_L2, cn_fc_L2) = self.encoder_fc(f1m)
                loss += F.mse_loss(out_fc_L1, out_fc_L2)*alpha_fc
                ctx_fc_L1, _ = self.attention_1(out_fc_L1,out_fc_L1, out_fc_L1)
                out_fc_L1 = self.layer_norm_1(ctx_fc_L1+out_fc_L1)
                ctx_fc_L2, _ = self.attention_1(out_fc_L2,out_fc_L2, out_fc_L2)
                out_fc_L2 = self.layer_norm_1(ctx_fc_L2+out_fc_L2)
                #encoder forcing for magnetometer
                out_mg_L1, (hn_mg_L1, cn_mg_L1) = self.encoder_mg(mg1)
                out_mg_L2, (hn_mg_L2, cn_mg_L2) = self.encoder_mg(m1m)
                loss += F.mse_loss(out_mg_L1, out_mg_L2)*alpha_mg
                ctx_mg_L1, _ = self.attention_2(out_mg_L1,out_mg_L1, out_mg_L1)
                out_mg_L1 = self.layer_norm_2(ctx_mg_L1+out_mg_L1)
                ctx_mg_L2, _ = self.attention_2(out_mg_L2,out_mg_L2, out_mg_L2)
                out_mg_L2 = self.layer_norm_2(ctx_mg_L2+ out_mg_L2)
                #concat both encoder outputs
                hn_L1 = torch.cat([hn_fc_L1, hn_mg_L1], dim = -1)
                cn_L1 = torch.cat([cn_fc_L1, cn_mg_L1], dim = -1)
                out_L1 = torch.cat([out_fc_L1, out_mg_L1], dim = -1)
                out_L1, (_,_) = self.decoder(out_L1, hn_L1, cn_L1)
                out_L1 = self.fc(out_L1[:,-1,:])
                hn_L2 = torch.cat([hn_fc_L2, hn_mg_L2], dim = -1)
                cn_L2 = torch.cat([cn_fc_L2, cn_mg_L2], dim = -1)
                out_L2 = torch.cat([out_fc_L2, out_mg_L2], dim = -1)
                out_L2, (_,_) = self.decoder(out_L2, hn_L2, cn_L2)
                out_L2 = self.fc(out_L2[:,-1,:])
                loss += F.mse_loss(out_L1, out_L2)*alpha_out
                loss += F.mse_loss(out_L1, target)*alpha_main
            elif isinstance(self, MultiHeaded2MultiheadAttentionGRU):
                #instantiate loss
                loss = 0
                alpha_fc, alpha_mg, alpha_out, alpha_main = weights
                fc1, mg1, f1m, m1m, target = batch #decompose batch
                #encoder forcing for faraday cup
                out_fc_L1, hn_fc_L1 = self.encoder_fc(fc1)
                out_fc_L2, hn_fc_L2 = self.encoder_fc(f1m)
                loss += F.mse_loss(out_fc_L1, out_fc_L2)*alpha_fc
                ctx_fc_L1, _ = self.attention_1(out_fc_L1,out_fc_L1, out_fc_L1)
                out_fc_L1 = self.layer_norm_1(ctx_fc_L1+out_fc_L1)
                ctx_fc_L2, _ = self.attention_1(out_fc_L2,out_fc_L2, out_fc_L2)
                out_fc_L2 = self.layer_norm_1(ctx_fc_L2+out_fc_L2)
                #encoder forcing for magnetometer
                out_mg_L1, hn_mg_L1 = self.encoder_mg(mg1)
                out_mg_L2, hn_mg_L2 = self.encoder_mg(m1m)
                loss += F.mse_loss(out_mg_L1, out_mg_L2)*alpha_mg
                ctx_mg_L1, _ = self.attention_2(out_mg_L1,out_mg_L1, out_mg_L1)
                out_mg_L1 = self.layer_norm_2(ctx_mg_L1+out_mg_L1)
                ctx_mg_L2, _ = self.attention_2(out_mg_L2,out_mg_L2, out_mg_L2)
                out_mg_L2 = self.layer_norm_2(ctx_mg_L2+ out_mg_L2)
                #concat both encoder outputs
                hn_L1 = torch.cat([hn_fc_L1, hn_mg_L1], dim = -1)
                out_L1 = torch.cat([out_fc_L1, out_mg_L1], dim = -1)
                out_L1, _ = self.decoder(out_L1, hn_L1)
                out_L1 = self.fc(out_L1[:,-1,:])
                hn_L2 = torch.cat([hn_fc_L2, hn_mg_L2], dim = -1)
                out_L2 = torch.cat([out_fc_L2, out_mg_L2], dim = -1)
                out_L2, _ = self.decoder(out_L2, hn_L2)
                out_L2 = self.fc(out_L2[:,-1,:])
                loss += F.mse_loss(out_L1, out_L2)*alpha_out
                loss += F.mse_loss(out_L1, target)*alpha_main
        else:
            fc1, mg1, target = batch
            out = self(fc1, mg1)
            loss = F.mse_loss(out, target)
        return loss
    def validation_step(self, batch):
        if self.encoder_forcing:
            fc, mg,_,_, target = batch #decompose batch
        else:
            fc, mg, target = batch 
        pred = self(fc, mg)        
        #dst index or kp
        loss = F.mse_loss(pred, target)
        r2_ = r2(pred, target)
        precision, recall, f1_score = compute_all(pred, target)
        return {'val_loss': loss.detach(),'r2': r2_.detach(), 'precision': precision, 'recall': recall, 'f1': f1_score}

class EncoderMultiheadAttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers = 1, dropout = 0, bidirectional = True):
        super(EncoderMultiheadAttentionLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size= hidden_size
        #attention
        self.attention = nn.MultiheadAttention(input_size, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(input_size)
        
        #encoder
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, dropout = dropout, bidirectional = bidirectional, batch_first = True)

    def forward(self, x):
        #Multihead attention
        attn_out, _ = self.attention(x,x,x)
        #residual connection and layer_norm
        attn_out = self.layer_norm(attn_out+x)
        #LSTM
        out, (hn,cn) = self.lstm(attn_out)
        #return outputs
        return out, (hn, cn)

class DecoderMultiheadAttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers = 1, dropout = 0, bidirectional = True):
        super(DecoderMultiheadAttentionLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size= hidden_size
        #attention
        self.attention = nn.MultiheadAttention(input_size, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(input_size)
        
        #encoder
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, dropout = dropout, bidirectional = bidirectional, batch_first = True)

    def forward(self, x, hn ,cn):
        #LSTM
        out, (hn,cn) = self.lstm(x, (hn,cn))
        context, _ = self.attention(out, out, out)
        out = self.layer_norm(context+out)
        #return outputs
        return out, (hn, cn)
    

class MultiHeaded2MultiheadAttentionLSTM(MultiHead2MultiHeadBase):
    def __init__(self, encoder_fc, encoder_mg, decoder, num_heads: list, architecture, output_size):
        super(MultiHeaded2MultiheadAttentionLSTM, self).__init__()
        #hidden
        self.input_size = encoder_fc.hidden_size + encoder_mg.hidden_size
        #encoder(LSTMWithMultiHeadAttention)
        self.encoder_fc = encoder_fc
        self.encoder_mg = encoder_mg
        self.decoder = decoder
        #MultiheadAttention
        self.attention_1 = nn.MultiheadAttention(encoder_fc.hidden_size, num_heads[0], batch_first=True)
        self.attention_2 = nn.MultiheadAttention(encoder_mg.hidden_size, num_heads[1], batch_first=True)
        #layer norm with residual connections(AttentionIsAllYouNeed uses several times on arch)
        self.layer_norm_1 = nn.LayerNorm(encoder_fc.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(encoder_mg.hidden_size)
        #Decoder arch
        self.fc = DeepNeuralNetwork(self.input_size, output_size, *architecture)
    def forward(self, fc, mg):
        #encoder for faraday cup
        out_fc, (hn_fc,cn_fc) = self.encoder_fc(fc)
        #Attention mechanism
        attn_fc, _ = self.attention_1(out_fc,out_fc,out_fc)
        #Layer norm and residual
        out_fc = self.layer_norm_1(attn_fc+out_fc)
        #encoder for magnetometer
        out_mg, (hn_mg,cn_mg) = self.encoder_mg(mg)
        #Attention mechanism
        attn_mg, _ = self.attention_2(out_mg,out_mg,out_mg)
        #Layer norm and residual
        out_mg = self.layer_norm_2(attn_mg+out_mg)
        hn = torch.cat([hn_fc, hn_mg], dim = -1)
        cn = torch.cat([cn_fc, cn_mg], dim = -1)
        out = torch.cat([out_fc, out_mg], dim = -1)
        out, (_,_) = self.decoder(out, hn, cn)
        out = self.fc(out[:,-1,:])
        return out


class EncoderMultiheadAttentionGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers = 1, dropout = 0, bidirectional = True):
        super(EncoderMultiheadAttentionGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size= hidden_size
        #attention
        self.attention = nn.MultiheadAttention(input_size, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(input_size)
        
        #encoder
        self.gru = nn.GRU(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, dropout = dropout, bidirectional = bidirectional, batch_first = True)

    def forward(self, x):
        #Multihead attention
        attn_out, _ = self.attention(x,x,x)
        #residual connection and layer_norm
        attn_out = self.layer_norm(attn_out+x)
        #LSTM
        out, hn = self.gru(attn_out)
        #return outputs
        return out, hn

class DecoderMultiheadAttentionGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers = 1, dropout = 0, bidirectional = True):
        super(DecoderMultiheadAttentionGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size= hidden_size
        #attention
        self.attention = nn.MultiheadAttention(input_size, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(input_size)
        
        #encoder
        self.gru = nn.GRU(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, dropout = dropout, bidirectional = bidirectional, batch_first = True)

    def forward(self, x, hn):
        #LSTM
        out, hn = self.gru(x, hn)
        context, _ = self.attention(out, out, out)
        out = self.layer_norm(context+out)
        #return outputs
        return out, hn
    
class MultiHeaded2MultiheadAttentionGRU(MultiHead2MultiHeadBase):
    def __init__(self, encoder_fc, encoder_mg, decoder, num_heads: list, architecture, output_size):
        super(MultiHeaded2MultiheadAttentionGRU, self).__init__()
        #hidden
        self.input_size = encoder_fc.hidden_size + encoder_mg.hidden_size
        #encoder(LSTMWithMultiHeadAttention)
        self.encoder_fc = encoder_fc
        self.encoder_mg = encoder_mg
        self.decoder = decoder
        #MultiheadAttention
        self.attention_1 = nn.MultiheadAttention(encoder_fc.hidden_size, num_heads[0], batch_first=True)
        self.attention_2 = nn.MultiheadAttention(encoder_mg.hidden_size, num_heads[1], batch_first=True)
        #layer norm with residual connections(AttentionIsAllYouNeed uses several times on arch)
        self.layer_norm_1 = nn.LayerNorm(encoder_fc.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(encoder_mg.hidden_size)
        #Decoder arch
        self.fc = DeepNeuralNetwork(self.input_size, output_size, *architecture)
    def forward(self, fc, mg):
        #encoder for faraday cup
        out_fc, hn_fc = self.encoder_fc(fc)
        #Attention mechanism
        attn_fc, _ = self.attention_1(out_fc,out_fc,out_fc)
        #Layer norm and residual
        out_fc = self.layer_norm_1(attn_fc+out_fc)
        #encoder for magnetometer
        out_mg, hn_mg = self.encoder_mg(mg)
        #Attention mechanism
        attn_mg, _ = self.attention_2(out_mg,out_mg,out_mg)
        #Layer norm and residual
        out_mg = self.layer_norm_2(attn_mg+out_mg)
        hn = torch.cat([hn_fc, hn_mg], dim = -1)
        out = torch.cat([out_fc, out_mg], dim = -1)
        out, _ = self.decoder(out, hn)
        out = self.fc(out[:,-1,:])
        return out