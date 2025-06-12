import torch.nn as nn
import torch.nn.functional as F

class Cnn(nn.Module):
    def __init__(self, init_bounds=(0.0, 1.0)):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50,10)

        #for m in self.modules():
            #if isinstance(m, (nn.Conv2d, nn.Linear)):
                #nn.init.uniform_(m.weight, init_bounds[0], init_bounds[1])
                #if m.bias is not None:
                    #nn.init.uniform_(m.bias, init_bounds[0], init_bounds[1])
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)
