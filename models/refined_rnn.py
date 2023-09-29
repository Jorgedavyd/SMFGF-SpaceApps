import torch.functional as F
import torch.nn as nn
import torch

from models.rnn_models import get_lr

class RefinedRNN(nn.Module):
    def __init__(self, encoder, fc):
        super(RefinedRNN, self).__init__()
        self.encoder = encoder
        self.fc = fc   #RNN with input size: hidden_size, hidden_size: hidden_size, the encoder will encode min into 1/3 time step and 1/3 of features.
    def forward(self, x):
        out = self.encoder(x)
        out = self.fc(out)
        return out
    def training_step(self, batch, weigths = [0.1,0.1,1]):
        h_weigth, o_weigth, main = weigths
        #initialize loss
        loss = 0
        #from batch to encoder loss
        x_t, x_t_hat, y_t = batch
        h_t = self.encoder(x_t)
        h_t_hat = self.encoder(x_t_hat)
        encoder_loss = F.mse_loss(h_t, h_t_hat)
        #add to overall
        loss+=h_weigth*encoder_loss
        #from batch to output loss
        o_t = self(x_t)
        o_t_hat = self(x_t_hat)
        output_loss = F.mse_loss(o_t, o_t_hat)
        #add to overall
        loss+=o_weigth*output_loss
        #from batch to normal loss
        main_loss = F.mse_loss(o_t, y_t) 
        #add to overall
        loss+=main*main_loss
        return loss, encoder_loss, output_loss, main_loss

    def validation_step(self, batch, weigths = [1,1,1]):
        h_weigth, o_weigth, main = weigths
        #initialize loss
        loss = 0
        #from batch to encoder loss
        x_t, x_t_hat, y_t = batch
        h_t = self.encoder(x_t)
        h_t_hat = self.encoder(x_t_hat)
        encoder_loss = F.mse_loss(h_t, h_t_hat)
        #add to overall
        loss+=h_weigth*encoder_loss
        #from batch to output loss
        o_t = self(x_t)
        o_t_hat = self(x_t_hat)
        output_loss = F.mse_loss(o_t, o_t_hat)
        #add to overall
        loss+=o_weigth*output_loss
        #from batch to normal loss
        main_loss = F.mse_loss(o_t, y_t) 
        #add to overall
        loss+=main*main_loss
        return {'overall_loss': loss.detach(), 'main_loss': main_loss.detach(), 'output_loss': output_loss.detach(), 'encoder_loss': encoder_loss.detach()}

    def validation_epoch_end(self, outputs):
        batch_overall = [x['overall_loss'] for x in outputs]
        epoch_overall = torch.stack(batch_overall).mean()   
        batch_main = [x['main_loss'] for x in outputs]
        epoch_main = torch.stack(batch_main).mean()   
        batch_output = [x['output_loss'] for x in outputs]
        epoch_output = torch.stack(batch_output).mean()  
        batch_encoder = [x['encoder_loss'] for x in outputs]
        epoch_encoder = torch.stack(batch_encoder).mean()   
        return {'val_overall_loss': epoch_overall.item(), 'val_main_loss': epoch_main.item(), 'val_output_loss': epoch_output.item(), 'val_encoder_loss': epoch_encoder.item()}

    def epoch_end(self, epoch, result): # Seguimiento del entrenamiento
        print("Epoch [{}], last_lr: {:.5f}\n,train_overall_loss: {:.4f}, train_main_loss: {:.4f}, train_output_loss: {:.4f}, train_encoder_loss: {:.4f}\nval_overall_loss: {:.4f}, val_main_loss: {:.4f}, val_output_loss: {:.4f}, val_encoder_loss: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_overall_loss'], result['train_main_loss'], result['train_output_loss'], result['train_encoder_loss'], result['val_overall_loss'], result['val_main_loss'], result['val_output_loss'], result['val_encoder_loss']))
    
    def evaluate(self, val_loader):
        self.eval()
        outputs = [self.validation_step(batch) for batch in val_loader]
        return self.validation_epoch_end(outputs)
    
    def fit(self, epochs, max_lr, train_loader, val_loader,
                  weight_decay=0, grad_clip=False, opt_func=torch.optim.Adam):
        torch.cuda.empty_cache()
        history = [] # Seguimiento de entrenamiento

        # Poner el método de minimización personalizado
        optimizer = opt_func(self.parameters(), max_lr, weight_decay=weight_decay)
        # Learning rate scheduler, le da momento inicial al entrenamiento para converger con valores menores al final
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                    steps_per_epoch=len(train_loader))

        for epoch in range(epochs):
            # Training Phase
            self.train()  #Activa calcular los vectores gradiente
            train_losses = []
            encoder_losses = []
            output_losses = []
            main_losses = []
            lrs = [] # Seguimiento
            for batch in train_loader:
                # Calcular el costo
                loss, encoder_loss, output_loss, main_loss = self.training_step(batch)
                #Seguimiento
                train_losses.append(loss)
                encoder_losses.append(encoder_loss)
                output_losses.append(output_loss)
                main_losses.append(main_loss)
                #Calcular las derivadas parciales
                loss.backward()

                # Gradient clipping, para que no ocurra el exploding gradient
                if grad_clip:
                    nn.utils.clip_grad_value_(self.parameters(), grad_clip)

                #Efectuar el descensod e gradiente y borrar el historial
                optimizer.step()
                optimizer.zero_grad()

                # Guardar el learning rate utilizado en el cycle.
                lrs.append(get_lr(optimizer))
                #Utilizar el siguiente valor de learning rate dado OneCycle scheduler
                sched.step()

            # Fase de validación
            result = self.evaluate(val_loader)
            result['train_overall_loss'] = torch.stack(train_losses).mean().item() 
            result['train_main_loss'] = torch.stack(main_losses).mean().item() 
            result['train_output_loss'] = torch.stack(output_losses).mean().item()
            result['train_encoder_loss'] = torch.stack(encoder_losses).mean().item()
            result['lrs'] = lrs 
            self.epoch_end(epoch, result)
            history.append(result)
        return history

