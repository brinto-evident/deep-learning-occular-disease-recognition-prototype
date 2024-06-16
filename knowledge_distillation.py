import os
import timeit
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn import metrics
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using %s device.' % (device))

ROOT_DIR = 'dataset'

# Hyperparameters
BATCH_SIZE  = 32
LEARNING_RATE = 0.001
NUM_CLASSES = 4
num_epochs = 12
temperature = float(2.0)

# build the augmentations
class RandomGaussianBlur(object):
    def __call__(self, img):
        do_it = np.random.rand() > 0.5
        if not do_it:
            return img
        sigma = np.random.rand() * 1.9 + 0.1
        return cv2.GaussianBlur(np.asarray(img), (23, 23), sigma)

def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Compose([
        get_color_distortion(),
        RandomGaussianBlur(),
        ]),
    transforms.ToTensor(),
    normalize,
    ])

# Define dataset and dataloaders
dataset = datasets.ODIR5K(ROOT_DIR, transform)
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# Load ResNet-150
resnet150 = torch.hub.load('pytorch/vision:v0.11.1', 'resnet152', pretrained=True)

num_features = resnet150.fc.in_features
resnet150.fc = nn.Sequential(
        nn.Linear(num_features, NUM_CLASSES),
        nn.Sigmoid())

resnet150 = resnet150.to(device)

# Load pre-trained ResNet-50 as the student
resnet50 = torch.hub.load('pytorch/vision:v0.11.1', 'resnet50', pretrained=True)

num_features50 = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
        nn.Linear(num_features50, NUM_CLASSES),
        nn.Sigmoid())

student = resnet50.to(device)


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet150.parameters(), lr=LEARNING_RATE)

# Train the teacher model
def train_teacher():
    resnet150.train()
    for epoch in range(num_epochs):
        start_time = timeit.default_timer()
        y_true = torch.FloatTensor()
        y_pred = torch.FloatTensor()
        train_loss = 0
        for index, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device)
            labels = labels.to(device)


            outputs = resnet150(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_true = torch.cat((y_true, labels.cpu()))
            y_pred = torch.cat((y_pred, outputs.detach().cpu()))

            print('\repoch %3d/%3d batch %3d/%3d' % (epoch+1, num_epochs, index, len(train_loader)), end='')
            print(' --- loss %6.4f' % (train_loss / index), end='')
            print(' --- %5.1fsec' % (timeit.default_timer() - start_time), end='')

        aucs = [metrics.roc_auc_score(y_true[:, i], y_pred[:, i]) for i in range(NUM_CLASSES)]
        auc_classes = ' '.join(['%5.3f' % (aucs[i]) for i in range(NUM_CLASSES)])
        print(' --- mean AUC score: %5.3f (%s)' % (np.mean(aucs), auc_classes))

print("Training Teacher Model")

train_teacher()

# Clone ResNet-150 to create the teacher
teacher = resnet150
teacher.load_state_dict(resnet150.state_dict())  # Copy weights

# Define distillation loss (e.g., Mean Squared Error)
distillation_criterion = nn.MSELoss()
optim = torch.optim.Adam(student.parameters(), lr=LEARNING_RATE)

def distillation_loss(y, teacher_scores, temp):
    soft_teacher = nn.functional.softmax(teacher_scores / temp, dim=1)
    soft_student = nn.functional.log_softmax(y / temp**(0.5), dim=1)
    return distillation_criterion(soft_student, soft_teacher)

# Knowledge Distillation training
def train_distillation():
    student.train()
    teacher.eval()

    for epoch in range(num_epochs):
        start_time = timeit.default_timer()
        y_true = torch.FloatTensor()
        y_pred = torch.FloatTensor()
        train_loss = 0
        for index, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device)
            labels = labels.to(device)

            optim.zero_grad()
            outputs_student = student(images)
            outputs_teacher = teacher(images)
            loss_distillation = distillation_loss(outputs_student, outputs_teacher, temperature)
            train_loss += loss_distillation.item()

            loss_distillation.backward()
            optim.step()

            y_true = torch.cat((y_true, labels.cpu()))
            y_pred = torch.cat((y_pred, outputs_student.detach().cpu()))

            print('\repoch %3d/%3d batch %3d/%3d' % (epoch+1, num_epochs, index, len(train_loader)), end='')
            print(' --- loss %6.4f' % (train_loss / index), end='')
            print(' --- %5.1fsec' % (timeit.default_timer() - start_time), end='')

        aucs = [metrics.roc_auc_score(y_true[:, i], y_pred[:, i]) for i in range(NUM_CLASSES)]
        auc_classes = ' '.join(['%5.3f' % (aucs[i]) for i in range(NUM_CLASSES)])
        print(' --- mean AUC score: %5.3f (%s)' % (np.mean(aucs), auc_classes))

train_distillation()

def evaluate(model, dataloader):
    model.eval()
    start_time = timeit.default_timer()
    correct = 0
    total = 0
    y_true = torch.FloatTensor()
    y_pred = torch.FloatTensor()
    val_loss_count = 0

    with torch.no_grad():
        for index, (images, labels) in enumerate(dataloader, 1):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            val_loss = criterion(outputs, labels)
            val_loss_count += val_loss.item()

            y_true = torch.cat((y_true, labels.cpu()))
            y_pred = torch.cat((y_pred, outputs.detach().cpu()))

        aucs = [metrics.roc_auc_score(y_true[:, i], y_pred[:, i]) for i in range(NUM_CLASSES)]
        auc_classes = ' '.join(['%5.3f' % (aucs[i]) for i in range(NUM_CLASSES)])

    return np.mean(aucs)

student_accuracy = evaluate(student, val_loader)
print("Student Model Validation Accuracy: %5.3f"%(student_accuracy))