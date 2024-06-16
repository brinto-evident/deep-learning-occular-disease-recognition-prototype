import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from main import get_color_distortion, RandomGaussianBlur
import datasets
import timeit
from sklearn import metrics
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using %s device.' % (device))

# Hyper Parameters
BATCH_SIZE  = 32
LEARNING_RATE = 0.001
NUM_CLASSES = 2

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])
# build the augmentations
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Compose([
        get_color_distortion(),
        RandomGaussianBlur(),
        ]),
    transforms.ToTensor(),
    normalize,
    ])

# Step 2: Load the data
train_dataset = datasets.ODIR5K('train', transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

test_dataset = datasets.ODIR5K('test', transform)
val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Step 3: Define the teacher model
# teacher_model = models.resnet50(pretrained=True)
# teacher_model.fc = nn.Linear(teacher_model.fc.in_features, NUM_CLASSES)  # Replace num_classes with the number of disease classes
# teacher_model = teacher_model.to(device)
# teacher_model.eval()

teacher_model = torch.hub.load('facebookresearch/swav', 'resnet50')
num_features = teacher_model.fc.in_features
teacher_model.fc = nn.Sequential(
        nn.Linear(num_features, NUM_CLASSES),
        nn.Sigmoid())

if os.path.exists('model'):
    teacher_model.load_state_dict(torch.load('./model/checkpoint.pth', map_location=device))
    print('model state has loaded.')

teacher_model = teacher_model.to(device)
teacher_model.eval()

start_time = timeit.default_timer()
y_true = torch.FloatTensor()
y_pred = torch.FloatTensor()
test_loss = 0
optim = torch.optim.Adam(teacher_model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()
for index, (images, labels) in enumerate(val_loader, 1):
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = teacher_model(images)

    loss = criterion(outputs, labels)
    test_loss += loss.item()

    y_true = torch.cat((y_true, labels.cpu()))
    y_pred = torch.cat((y_pred, outputs.detach().cpu()))

    print('\rtest batch %3d/%3d' % (index, len(val_loader)), end='')
    print(' --- loss %6.4f' % (test_loss / index), end='')
    print(' --- %5.1fsec' % (timeit.default_timer() - start_time), end='')

aucs = [metrics.roc_auc_score(y_true[:, i], y_pred[:, i]) for i in range(NUM_CLASSES)]
auc_classes = ' '.join(['%5.3f' % (aucs[i]) for i in range(NUM_CLASSES)])
print(' --- mean AUC score: %5.3f (%s)' % (np.mean(aucs), auc_classes))

# Step 4: Define the student model
class StudentModel(nn.Module):
    def __init__(self, num_classes):
        super(StudentModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, 2)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)  # Flatten the tensor for the fully connected layer
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

student_model = StudentModel(NUM_CLASSES)
student_model = student_model.to(device)
summary(student_model, input_size=(3, 512, 512))

# Step 5: Define the loss function
def knowledge_distillation_loss(y_student, y_teacher, T=1.0, alpha=0.7):
    """
    Compute knowledge distillation loss
    Args:
        y_student: Output logits from the student model
        y_teacher: Output logits from the teacher model
        T: Temperature parameter for distillation (usually set to 1.0)
        alpha: Weight of the distillation loss relative to the cross-entropy loss
    Returns:
        The combined knowledge distillation loss
    """
    # Apply the temperature scaling to the teacher's logits
    y_soft = nn.functional.softmax(y_teacher / T, dim=1)
    
    # Compute the cross-entropy loss between student's logits and ground truth labels
    cross_entropy_loss = nn.CrossEntropyLoss()(y_student, labels)
    
    # Compute the KL divergence loss between the student's logits and the teacher's softened logits
    kl_div_loss = nn.KLDivLoss()(nn.functional.log_softmax(y_student / T, dim=1), y_soft)
    
    # Combine the losses with the specified weighting
    loss = (1 - alpha) * cross_entropy_loss + alpha * kl_div_loss
    
    return loss

# Step 6: Train the student model with distillation
optimizer = torch.optim.Adam(student_model.parameters(), lr=LEARNING_RATE)
NUM_EPOCHS = 10

train_loss = 0
for epoch in range(NUM_EPOCHS):
    student_model.train()
    for index, (images, labels) in enumerate(train_loader, 1):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        # Forward pass for the student model
        outputs_student = student_model(images)
        print(outputs_student.shape)
        # Forward pass for the teacher model
        with torch.no_grad():
            outputs_teacher = teacher_model(images)
            print(outputs_teacher.shape)
        
        # Calculate the knowledge distillation loss
        loss = knowledge_distillation_loss(outputs_student, outputs_teacher)
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()

        print('\repoch %3d/%3d batch %3d/%3d' % (epoch+1, NUM_EPOCHS, index, len(train_loader)), end='')
        print(' --- loss %6.4f' % (train_loss / index), end='')
        print(' --- %5.1fsec' % (timeit.default_timer() - start_time), end='')

# # Step 7: Evaluate the student model
# student_model.eval()
# correct = 0
# total = 0

# with torch.no_grad():
#     for images, labels in val_loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = student_model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# accuracy = 100 * correct / total
# print(f'Validation Accuracy of the student model: {accuracy:.2f}%')
