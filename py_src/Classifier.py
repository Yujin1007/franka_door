import torch.nn as nn
import torch.nn.functional as F

# class Classifier(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(Classifier, self).__init__()
#         self.fc1 = nn.Linear(input_size, 256)
#         self.fc2 = nn.Linear(256, 512)
#         self.fc_ = nn.Linear(512, 256)
#         self.fc4 = nn.Linear(256, output_size)
#         # self.fc1 = nn.Linear(input_size, 256)
#         # self.fc2 = nn.Linear(256, 256)
#         # self.fc_ = nn.Linear(256, 256)
#         # self.fc4 = nn.Linear(256, output_size)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc_(x))
#         x = F.relu(self.fc4(x))
#         return x

class Classifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(Classifier, self).__init__()
        # self.fc1 = nn.Linear(input_size, 64)
        # self.fc2 = nn.Linear(64, 256)
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024,1024)
        self.fc_ = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, output_size)

        # self.dropout1 = nn.Dropout(p=0.1)
        # self.dropout2 = nn.Dropout(p=0.1)
        # self.dropout3 = nn.Dropout(p=0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc_(x))

        x = F.softmax(self.fc4(x),dim=0)
        # x = F.relu(self.fc4(x))
        # x = F.sigmoid(self.fc4(x))
        return x
