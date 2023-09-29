class GeomagneticModel():
    def __init__(self, model):
        self.model = model
    
    def inference(self, x):
        #Define later
        return self.model(x)
    
    def geomagnetic_storm_gravity(self, dst_index):
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

    def geomagnetic_asses(self, input):
        storm_gravity_descriptions = [self.geomagnetic_storm_gravity(dst.item()) for dst in self.inference(input)]
        dst_index = [dst.item() for dst in self.inference(input)]
        for hour, conc in enumerate(storm_gravity_descriptions):
            print(f'Hour {hour}:\n\tDst index: {dst_index[hour]}\n\tDescription: {conc}')