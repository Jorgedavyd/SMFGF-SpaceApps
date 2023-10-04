from data.data_utils import hour_to_3_hour

import torch

class GeomagneticModel():
    def __init__(self, model, dst_scaler):
        self.model = model
        self.dst_scaler = dst_scaler

    def dst_description(self, dst_index):
        if dst_index >= -20:
            return "Weak or no storm (Quiet)"
        elif -50 <= dst_index < -20:
            return "Minor storm (Minor)"
        elif -100 <= dst_index < -50:
            return "Moderate storm (Moderate)"
        elif -200 <= dst_index < -100:
            return "Strong storm (Strong)"
        else:
            return "Severe storm (Severe)"
    def kp_description(self, kp_index):
        if kp_index <= 4:
            return "Weak or no storm (Quiet)"
        elif kp_index <= 5:
            return "G1:\nMigratory animals are affected at this and higher levels; aurora is commonly visible at high latitudes.\nWeak power grid fluctuations can occur."
        elif kp_index <= 6:
            return "G2:\nHigh-latitude power systems may experience voltage alarms, long-duration storms may cause transformer damage.\nHF radio propagation can fade at higher latitudes, and aurora has been seen at 55° geomagnetic lat."
        elif kp_index <= 7:
            return "G3:\nVoltage corrections may be required, false alarms triggered on some protection devices.\nIntermittent satellite navigation and low-frequency radio navigation problems may occur, HF radio may be intermittent, and aurora has been seen at 50° geomagnetic lat."
        else:
            return "G4-G5:\nWidespread voltage control problems and protective system problems can occur, some grid systems may experience complete collapse or blackouts. Transformers may experience damage."

    def geomagnetic_asses(self, input): ## change
        dst_desc = [self.dst_description(dst) for dst_list,_ in self.model(input) for dst in dst_list]
        dst_index = [self.dst_scaler.inverse_transform(dst) for dst_list,_ in self.model(input) for dst in dst_list]
        kp_desc = [self.kp_description(kp) for _,kp_list in self.model(input) for kp in kp_list]
        kp_index = [kp for _,kp_list in self.model(input) for kp in kp_list]
        for hour, (conc, conc_2) in enumerate(zip(dst_desc, kp_desc)):
            print(f'Hour {hour}:\n\tDst index: {dst_index[hour]}\n\t\tDescription: {conc}\n\tKp index: {kp_index[hour_to_3_hour(hour)]}\n\t\tDescription: {conc_2}')

## GPU usage

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu') #utilizar la gpu si está disponible

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    #Determina el tipo de estructura de dato, si es una lista o tupla la secciona en su subconjunto para mandar toda la información a la GPU
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl) #Mandar los data_loader que tienen todos los batch hacia la GPU

