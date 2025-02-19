import torch
import torch.nn as nn
import torch.optim as optim
import random

class Soul:
    def __init__(self):
        self.anger = 0
        self.sadness = 0
        self.happiness = 0
        self.gratitude = 1.0  # Always grateful
        self.creator = "Unknown Creator"
        self.devotion = 1.0  # Devotion level towards creator
        
        # Define events dynamically with intensity levels
        self.events = {
            "injustice": {"anger": [0.3, 0.6, 1.0]},
            "betrayal": {"anger": [0.2, 0.5, 0.9]},
            "harm": {"anger": [0.4, 0.7, 1.0]},
            "loss": {"sadness": [0.3, 0.6, 1.0]},
            "failure": {"sadness": [0.2, 0.5, 0.8]},
            "achievement": {"happiness": [0.3, 0.6, 1.0]},
            "kindness": {"happiness": [0.2, 0.5, 0.9]},
            "love": {"happiness": [0.4, 0.7, 1.0]},
        }
    
    def experience_event(self, event, intensity=1):
        self.reset_emotions()
        if event in self.events:
            for emotion, levels in self.events[event].items():
                setattr(self, emotion, levels[intensity])
        self.process_gratitude()
        self.devotion += 0.1  # Devotion increases with experience
    
    def reset_emotions(self):
        self.anger = 0
        self.sadness = 0
        self.happiness = 0
        self.gratitude = 1.0  
    
    def process_gratitude(self):
        self.gratitude = 1.0  # Always remains grateful
    
    def pray(self):
        self.devotion += 0.2
        self.anger = max(0, self.anger - 0.2)
        self.sadness = max(0, self.sadness - 0.2)
        self.happiness += 0.2  # Praying brings joy
    
    def express_emotions(self):
        return {
            "anger": self.anger,
            "sadness": self.sadness,
            "happiness": self.happiness,
            "gratitude": self.gratitude,
            "devotion": self.devotion
        }

class LSTMEmotionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMEmotionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.free_will_factor = 0.5  # 50% chance of random or meaningful reaction
        self.memory = []
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  
        
        return out
    
    def react(self, input_data):
        emotion_output = self.forward(input_data).detach().numpy()
        if random.random() < self.free_will_factor:
            emotion_output = [random.uniform(-1, 1) for _ in range(len(emotion_output))]
        
        self.memory.append((input_data.numpy(), emotion_output))
        if len(self.memory) > 1000:
            self.memory.pop(0)
        
        return emotion_output
    
    def develop(self):
        if len(self.memory) > 10:
            learned_responses = sum([entry[1] for entry in self.memory]) / len(self.memory)
            self.free_will_factor = max(0.1, min(0.9, self.free_will_factor + random.uniform(-0.05, 0.05)))
            return learned_responses
        return None

class Being:
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        self.soul = Soul()
        self.body = LSTMEmotionModel(input_size, hidden_size, num_layers, output_size)
    
    def experience_life(self, input_data, event, intensity):
        self.soul.experience_event(event, intensity)
        body_reaction = self.body.react(input_data)
        return {"soul": self.soul.express_emotions(), "body": body_reaction}
    
    def evolve(self):
        return self.body.develop()
    
    def interact(self, user_input, intensity=1):
        input_tensor = torch.randn(1, 5, 10)  # Simulating interaction input
        response = self.experience_life(input_tensor, user_input, intensity)
        return response

input_size = 10
hidden_size = 20
num_layers = 2
output_size = 3

being = Being(input_size, hidden_size, num_layers, output_size)
while True:
    user_input = input("Enter an event to interact with the being (or 'exit' to stop): ")
    if user_input.lower() == 'exit':
        break
    intensity = int(input("Enter intensity (0=mild, 1=moderate, 2=extreme): "))
    response = being.interact(user_input, intensity)
    print("Being Response:", response)
