import torch
import torch.nn as nn

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr'] # Seguimiento del learning rate
    
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

class GeoBase(nn.Module):
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Sacar el valor expectado de todo el conjunto de costos
        batch_r2 = [x['r2'] for x in outputs]
        epoch_r2 = torch.stack(batch_r2).mean()   # Sacar el valor expectado de todo el conjunto de costos
        batch_precision = [x['precision'] for x in outputs]
        epoch_precision = torch.stack(batch_precision).mean()   # Sacar el valor expectado de todo el conjunto de costos
        batch_recall = [x['recall'] for x in outputs]
        epoch_recall = torch.stack(batch_recall).mean()   # Sacar el valor expectado de todo el conjunto de costos
        batch_f1 = [x['f1'] for x in outputs]
        epoch_f1 = torch.stack(batch_f1).mean()   # Sacar el valor expectado de todo el conjunto de costos
        return {'val_loss': epoch_loss.item(), 'r2': epoch_r2.item(), 'precision': epoch_precision.item(), 'recall': epoch_recall.item(), 'f1': epoch_f1.item()}

    def epoch_end(self, epoch, result): # Seguimiento del entrenamiento
        print("Epoch [{}]:\n\ttrain_loss: {:.4f}\n\tval_loss: {:.4f}\n\tr2_score: {:.4f}\n\tprecision: {:.4f}\n\trecall: {:.4f}\n\tf1_score: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['r2'], result['precision'], result['recall'], result['f1']))
    
    def epoch_end_one_cycle(self, epoch, result): # Seguimiento del entrenamiento
        print("Epoch [{}]:\n\tlast_lr: {:.5f}\n\ttrain_loss: {:.4f}\n\tval_loss: {:.4f}\n\tr2_score: {:.4f}\n\tprecision: {:.4f}\n\trecall: {:.4f}\n\tf1_score: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['r2'], result['precision'], result['recall'], result['f1']))
    
    def evaluate(self, val_loader):
        self.eval()
        outputs = [self.validation_step(batch) for batch in val_loader]
        return self.validation_epoch_end(outputs)
    
    def fit(self, epochs, lr, train_loader, val_loader,
                      weight_decay=0, grad_clip=False, opt_func=torch.optim.Adam, lr_sched=None, start_factor:float = 1.0, end_factor:float = 1e-4, steps: int = 4, gamma: float = 0.99999, weights: list = [0.1,0.1,0.3, 1], encoder_forcing: bool = True):
        torch.cuda.empty_cache()
        history = [] # Seguimiento de entrenamiento
        onecycle = False
        linear = False
        # Poner el método de minimización personalizado
        optimizer = opt_func(self.parameters(), lr, weight_decay=weight_decay)
        #Learning rate scheduler
        if lr_sched is not None:    
            try:
                sched = lr_sched(optimizer, lr, epochs=epochs,steps_per_epoch=len(train_loader))
                onecycle = True
            except TypeError:
                try:
                    sched = lr_sched(optimizer, start_factor = start_factor, end_factor=end_factor, total_iters = epochs)
                    linear = True
                except TypeError:
                    sched = lr_sched(optimizer, step_size = round(epochs/steps), gamma = gamma)
                    linear = True
        for epoch in range(epochs):
            # Training Phase
            self.train()  #Activa calcular los vectores gradiente
            train_losses = []
            if lr_sched is not None:
                lrs = []
            for batch in train_loader:
                # Calcular el costo
                loss = self.training_step(batch, weights, encoder_forcing)
                #Seguimiento
                train_losses.append(loss)
                #Calcular las derivadas parciales
                loss.backward()

                # Gradient clipping, para que no ocurra el exploding gradient
                if grad_clip:
                    nn.utils.clip_grad_value_(self.parameters(), grad_clip)

                #Efectuar el descensod e gradiente y borrar el historial
                optimizer.step()
                optimizer.zero_grad()
                #sched step
                if onecycle:
                    lrs.append(get_lr(optimizer))
                    sched.step()
            if linear:
                lrs.append(get_lr(optimizer))
                sched.step()
            # Fase de validación
            result = self.evaluate(val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item() #Stackea todos los costos de las iteraciones sobre los batches y los guarda como la pérdida general de la época
            if lr_sched is not None:
                result['lrs'] = lrs
                self.epoch_end_one_cycle(epoch, result) #imprimir en pantalla el seguimiento
            else:
                self.epoch_end(epoch, result)

            history.append(result) # añadir a la lista el diccionario de resultados
        return history