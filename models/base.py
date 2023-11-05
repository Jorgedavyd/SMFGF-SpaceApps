import torch
import torch.nn as nn

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr'] # Seguimiento del learning rate
    
def r2(y_pred, y_true):
    mean = torch.mean(y_true)
    ss_total = torch.sum((y_true - mean) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2

def multiclass_accuracy(predicted, target):
    _, preds = torch.max(predicted, dim=1)
    return torch.sum(preds == target) / len(target)

def multiclass_precision(predicted, target):
    _, preds = torch.max(predicted, dim=1)
    correct = (preds == target).float()
    true_positive = torch.sum(correct)
    false_positive = torch.sum(preds != target)
    precision = true_positive / (true_positive + false_positive + 1e-7)
    return precision

def multiclass_recall(predicted, target):
    _, preds = torch.max(predicted, dim=1)
    correct = (preds == target).float()
    true_positive = torch.sum(correct)
    false_negative = torch.sum(preds != target)
    recall = true_positive / (true_positive + false_negative + 1e-7)
    return recall

def compute_all(predictions, targets):
    accuracy = multiclass_accuracy(predictions, targets)
    precision = multiclass_precision(predictions, targets)
    recall = multiclass_recall(predictions, targets)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
    return accuracy, precision, recall, f1_score

class GeoBase(nn.Module):
    def __init__(self, task: str):
        super(GeoBase, self).__init__()
        self.task_type = task
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        
        if self.task_type == 'regression':
            batch_metric = [x['r2'] for x in outputs]
            epoch_r2 = torch.stack(batch_metric).mean()
            return {'val_loss': epoch_loss.item(), 'r2': epoch_r2.item()}
        else:
            batch_accuracy = [x['accuracy'] for x in outputs]
            epoch_accuracy = torch.stack(batch_accuracy).mean()
            batch_precision = [x['precision'] for x in outputs]
            epoch_precision = torch.stack(batch_precision).mean()
            batch_recall = [x['recall'] for x in outputs]
            epoch_recall = torch.stack(batch_recall).mean()
            batch_f1 = [x['f1'] for x in outputs]
            epoch_f1 = torch.stack(batch_f1).mean()
            return {'val_loss': epoch_loss.item(), 'accuracy': epoch_accuracy.item(), 'precision': epoch_precision.item(), 'recall': epoch_recall.item(), 'f1': epoch_f1.item()}

    def epoch_end(self, epoch, result):
        if self.task_type == 'regression':
            print("Epoch [{}]:\n\ttrain_loss: {:.4f}\n\tval_loss: {:.4f}\n\tr2_score: {:.4f}".format(
                epoch, result['train_loss'], result['val_loss'], result['r2']))
        else:
            print("Epoch [{}]:\n\ttrain_loss: {:.4f}\n\tval_loss: {:.4f}\n\taccuracy: {:.4f}\n\tprecision: {:.4f}\n\trecall: {:.4f}\n\tf1_score: {:.4f}".format(
                epoch, result['train_loss'], result['val_loss'], result['accuracy'], result['precision'], result['recall'], result['f1']))

    def epoch_end_one_cycle(self, epoch, result):
        if self.task_type == 'regression':
            print("Epoch [{}]:\n\tlast_lr: {:.5f}\n\ttrain_loss: {:.4f}\n\tval_loss: {:.4f}\n\tr2_score: {:.4f}".format(
                epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['r2']))
        else:
            print("Epoch [{}]:\n\tlast_lr: {:.5f}\n\ttrain_loss: {:.4f}\n\tval_loss: {:.4f}\n\taccuracy: {:.4f}\n\tprecision: {:.4f}\n\trecall: {:.4f}\n\tf1_score: {:.4f}".format(
                epoch, result['lrs'][-1], result['train_loss'], result['val_loss'],result['accuracy'], result['precision'], result['recall'], result['f1']))
        
    
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