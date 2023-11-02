import torch.nn.functional as F
import torch
from models.base import *
class Sing2MultNN(GeoBase):
    def __init__(self, encoder, dst, kp, task = 'regression'):
        super(Sing2MultNN, self).__init__()
        GeoBase.__init__(self, task)
        self.encoder = encoder
        self.fc_dst = dst #multiheaded neural network ##regression
        self.fc_kp = kp   #multiclass
    def forward(self, x):
        out = self.encoder(x)
        dst_out = self.fc_dst(out)
        kp_out = self.fc_kp(out)
        return dst_out, kp_out
    def training_step(self, batch, weights:list, encoder_forcing):
        self.encoder_forcing = encoder_forcing
        if encoder_forcing:
            #instantiate loss
            loss = 0
            alpha_encoder, alpha_out, alpha_main = weights
            l1_data, l2_data, dst, kp = batch #decompose batc
            out_L1 = self.encoder(l1_data)
            out_L2 = self.encoder(l2_data)
            loss += F.mse_loss(out_L1, out_L2)*alpha_encoder
            out_dst_L1 = self.fc_dst(out_L1)
            out_dst_L2 = self.fc_dst(out_L2)
            loss += F.mse_loss(out_dst_L1, out_dst_L2)*alpha_out
            out_kp_L1 = self.fc_kp(out_L1)
            out_kp_L2 = self.fc_kp(out_L2)
            loss += F.mse_loss(out_kp_L1, out_kp_L2)*alpha_out

            loss += F.mse_loss(out_dst_L1, dst) *alpha_main if self.task_type == 'regression' else F.cross_entropy(out_dst_L1, dst)*alpha_main
            loss += F.mse_loss(out_kp_L1, kp) * alpha_main if self.task_type == 'regression' else F.cross_entropy(out_kp_L1, kp)*alpha_main
        else:
            l1_data, dst, kp = batch
            out_dst, out_kp = self(l1_data)
            loss = F.mse_loss(out_dst, dst) if self.task_type == 'regression' else F.cross_entropy(out_dst, dst)
            loss += F.mse_loss(out_kp, kp) if self.task_type == 'regression' else F.cross_entropy(out_kp, kp)
        return loss
    def validation_step(self, batch):
        if self.encoder_forcing:
            l1_data,_, dst, kp = batch #decompose batch
        else:
            l1_data, dst, kp= batch 
        out_dst, out_kp = self(l1_data)        
        #dst index or kp
        loss = F.mse_loss(out_dst, dst) if self.task_type == 'regression' else F.cross_entropy(out_dst, dst)
        loss += F.mse_loss(out_kp, kp) if self.task_type == 'regression' else F.cross_entropy(out_kp, kp)
        if self.task_type == 'regression':    
            r2_ = r2(out_dst,dst)
            r2_ += r2(out_kp,kp)
            return {'val_loss': loss.detach()/2, 'r2': r2_.detach()/2}
        else:
            accuracy, precision, recall, f1_score = compute_all(out_dst, dst)
            accuracy_2, precision_2, recall_2, f1_score_2 = compute_all(out_kp, kp) ##add threshold
            return {'val_loss': loss.detach()/2,'accuracy':(accuracy+accuracy_2)/2, 'precision': (precision + precision_2)/2, 'recall': (recall + recall_2)/2, 'f1': (f1_score + f1_score_2)/2}
            

class Sing2Sing(GeoBase):
    def __init__(self, encoder, fc, task='regression'):
        super(Sing2Sing, self).__init__()
        GeoBase.__init__(self, task)
        self.encoder = encoder
        self.fc = fc 
    def forward(self, x):
        out, _ = self.encoder(x)
        out = self.fc(out[:,-1,:])
        return out
    def training_step(self, batch, weights:list, encoder_forcing):
        self.encoder_forcing = encoder_forcing
        if encoder_forcing:
            #instantiate loss
            loss = 0
            alpha_encoder, alpha_out, alpha_main = weights
            l1_data, l2_data, target = batch #decompose batc
            out_L1, _ = self.encoder(l1_data)
            out_L2, _ = self.encoder(l2_data)
            loss += F.mse_loss(out_L1, out_L2)*alpha_encoder
            out_L1 = self.fc(out_L1[:, -1, :])
            out_L2 = self.fc(out_L2[:, -1, :])
            loss += F.mse_loss(out_L1, out_L2) * alpha_out
            loss += F.mse_loss(out_L1, target)*alpha_main if self.task_type == 'regression' else F.cross_entropy(out_L1, target)*alpha_main
        else:
            l1_data, target = batch
            out= self(l1_data)
            loss = F.mse_loss(out, target)*alpha_main if self.task_type == 'regression' else F.cross_entropy(out, target)*alpha_main
        return loss

    def validation_step(self, batch):
        if self.encoder_forcing:
            l1_data,_, target = batch #decompose batch
        else:
            l1_data, target= batch 
        pred= self(l1_data)        
        #dst index or kp
        loss = F.mse_loss(pred, target) if self.task_type == 'regression' else F.cross_entropy(pred, target)
        if self.task_type == 'regression':            
            r2_ = r2(pred, target)
            return {'val_loss': loss.detach(), 'r2': r2_.detach()}
        else: 
            accuracy, precision, recall, f1_score = compute_all(pred, target)
            return {'val_loss': loss.detach(), 'precision': precision, 'recall': recall, 'f1': f1_score, 'accuracy': accuracy}
class Mult2Sing(GeoBase):
    def __init__(self, encoder_fc, encoder_mg, fc, task = 'regression'):
        super(Mult2Sing, self).__init__()
        GeoBase.__init__(self, task)
        self.encoder_fc = encoder_fc
        self.encoder_mg = encoder_mg
        self.fc = fc #sum of both hiddens at input_size
    def forward(self, fc, mg):
        out_1 = self.encoder_fc(fc)
        out_2 = self.encoder_mg(mg)
        hidden = torch.cat((out_1, out_2), dim = 1)
        out = self.fc(hidden)
        return out
    def training_step(self, batch, weights:list, encoder_forcing):
        self.encoder_forcing = encoder_forcing
        if encoder_forcing:
            #instantiate loss
            loss = 0
            alpha_encoder, alpha_out, alpha_main = weights
            fc1, mg1,f1m, m1m, target = batch #decompose batc
            out_fc_l1 = self.encoder_fc(fc1)
            out_fc_l2 = self.encoder_fc(f1m)
            loss += F.mse_loss(out_fc_l1, out_fc_l2)*alpha_encoder
            out_mg_l1 = self.encoder_mg(mg1)
            out_mg_l2 = self.encoder_mg(m1m)
            loss += F.mse_loss(out_mg_l1, out_mg_l2)*alpha_encoder
            out_L1 = torch.cat([out_mg_l1, out_fc_l1], dim = 1)
            out_L2 = torch.cat([out_mg_l2, out_fc_l2], dim = 1)
            out_L1 = self.fc(out_L1)
            out_L2 = self.fc(out_L2)
            loss += F.mse_loss(out_L1, out_L2)*alpha_out
            loss += F.mse_loss(out_L1, target)*alpha_main if self.task_type == 'regression' else F.cross_entropy(out_L1, target)*alpha_main
        else:
            fc1, mg1, target = batch
            out = self(fc1, mg1)
            loss = F.mse_loss(out, target)*alpha_main if self.task_type == 'regression' else F.cross_entropy(out, target)*alpha_main
        return loss

    def validation_step(self, batch):
        if self.encoder_forcing:
            fc1, mg1, _,_, target = batch #decompose batch
        else:
            fc1, mg1, target= batch 
        pred = self(fc1, mg1)        
        #dst index or kp
        loss = F.mse_loss(pred, target) if self.task_type == 'regression' else F.cross_entropy(pred, target)
        if self.task_type == 'regression':            
            r2_ = r2(pred, target)
            return {'val_loss': loss.detach(), 'r2': r2_.detach()}
        else: 
            accuracy, precision, recall, f1_score = compute_all(pred, target)
            return {'val_loss': loss.detach(), 'precision': precision, 'recall': recall, 'f1': f1_score, 'accuracy': accuracy}

