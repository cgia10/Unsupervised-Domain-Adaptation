# Adapted from https://github.com/fungtion/DANN

import argparse
import os
import numpy as np
import random

import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.optim as optim

import data_train
from model_classifier import CNNModel
from evaluate_classifier import evaluate


# ---------
#  Parser
# ---------

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--val_epoch", type=int, default=1, help="on which epoch to save model and image")
parser.add_argument("--batch_size", type=int, default=40, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--gpu", type=int, default=0, help="GPU number")
parser.add_argument("--continue_epoch", type=int, default=1, help="Epoch number which was reached last time. Used to name saved models/images")
parser.add_argument("--source", type=str, default="MNISTM", help="name of source dataset")
parser.add_argument("--train_on_target", type=str, default="no", help="whether to train on the target dataset or not")
parser.add_argument("--model_name", type=str, default="class_mnistm_svhn_trainOnMnistm.pth.tar", help="name of saved model to load")
parser.add_argument("--data_dir", type=str, default="./data/", help="root path to the training images")
parser.add_argument("--save_dir", type=str, default="./saved_models/", help="directory to save trained model")
opt = parser.parse_args()


# ---------------------
#  Miscellaneous Setup
# ---------------------

# Set naming of model
if opt.continue_epoch == 1:
    opt.continue_epoch = opt.val_epoch
else:
    opt.continue_epoch += opt.val_epoch

# Directory for saved models
if not os.path.exists(opt.save_dir):
    print("Created save directory")
    os.makedirs(opt.save_dir)

# Set up GPU
if torch.cuda.is_available():
    print("GPU found")
    cuda = True
    torch.cuda.set_device(opt.gpu)
else:
    print("GPU not found")
    cuda = False

# Set seed for batch shuffling
manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)


# -------------
#  Dataloaders
# -------------

print("---> preparing dataloaders...")

# Generate data objects for train/test sets for both source and target datasets

# Source=MNISTM, Target=SVHN
if opt.source == "MNISTM":
    
    if opt.train_on_target == "yes":
        train = data_train.SVHN(opt, "train")
        train_set = "SVHN"
    else:
        train = data_train.MNISTM(opt, "train")
        train_set = "MNISTM"
    
    test = data_train.SVHN(opt, "test")
    test_set = "SVHN"
    target_set = "SVHN"
    
# Source=SVHN, Target=MNISTM
elif opt.source == "SVHN":

    if opt.train_on_target == "yes":
        train = data_train.MNISTM(opt, "train")
        train_set = "MNISTM"
    else:
        train = data_train.SVHN(opt, "train")
        train_set = "SVHN"
    
    test = data_train.MNISTM(opt, "test")
    test_set = "MNISTM"
    target_set = "MNISTM"

print("---> source = %s, target = %s, training on %s, testing on %s" % (opt.source, target_set, train_set, test_set))

# Generate dataloaders

# Training dataloader
dataloader_train = torch.utils.data.DataLoader(
    dataset=train,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu
)

# Testing dataloader
dataloader_test = torch.utils.data.DataLoader(
    dataset=test,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu
)


# ----------------------
#  Loss/Model/Optimizer
# ----------------------

# Loss functions
print("---> preparing loss function...")
loss_criteria = torch.nn.CrossEntropyLoss()

# Initialize models
print("---> preparing models...")
my_net = CNNModel()
model_pth = os.path.join(opt.save_dir, opt.model_name)

if os.path.exists(model_pth):
    print("---> found previously saved {}, loading checkpoint...".format(opt.model_name))
    checkpoint = torch.load(model_pth)
    my_net.load_state_dict(checkpoint)

# Optimizer
print("---> preparing optimizer...")
optimizer = optim.Adam(my_net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Move to GPU
if cuda:
    my_net = my_net.cuda()
    loss_criteria = loss_criteria.cuda()


# ----------
#  Training
# ----------

print("---> start training...")
iters = 1
best_acc = 0
best_epoch = 1

for epoch in range(1, opt.n_epochs+1):

    my_net.train()

    for idx, (img, gt) in enumerate(dataloader_train):

        # Create tensors for domain and class labels
        input_img = torch.FloatTensor(opt.batch_size, opt.channels, opt.img_size, opt.img_size)
        class_label = torch.LongTensor(opt.batch_size)
        
        # Move to GPU
        if cuda:
            img = img.cuda()
            gt = gt.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        # Tensor resizing
        input_img.resize_as_(img).copy_(img)
        class_label.resize_as_(gt).copy_(gt)

        # Model output and classification loss
        class_output = my_net(input_data=input_img)
        err = loss_criteria(class_output, class_label)         

        # Backpropagation
        optimizer.zero_grad()
        err.backward()
        optimizer.step()

        iters += 1

        print ("[Epoch: %d/%d] [Batch: %d/%d] [Classification loss: %.4f]" % (epoch, opt.n_epochs, idx, len(dataloader_train), err.data.cpu().numpy()))
        
        
    # ------------------
    #  Acc/Model Saving
    # ------------------

    if epoch % opt.val_epoch == 0:     
        
        # Accuracy on test set
        correct, total, target_acc = evaluate(my_net, dataloader_test)

        if target_acc > best_acc:
            best_acc = target_acc
            best_epoch = epoch

        # Print info
        print("")
        print("********** Current: epoch %d   Total: epoch %d **********" % (epoch, opt.continue_epoch))
        print("ACC: %d/%d correct, acc = %.2f%%" % (correct, total, target_acc))

        # Save model
        if opt.source == "MNISTM":
            if opt.train_on_target == "yes":
                torch.save(my_net.state_dict(), os.path.join(opt.save_dir, "class_mnistm_svhn_trainOnSvhn{}.pth.tar".format(opt.continue_epoch)))
                print("Saving model: class_mnistm_svhn_trainOnSvhn{}.pth.tar".format(opt.continue_epoch))
            else:
                torch.save(my_net.state_dict(), os.path.join(opt.save_dir, "class_mnistm_svhn_trainOnMnistm{}.pth.tar".format(opt.continue_epoch)))
                print("Saving model: class_mnistm_svhn_trainOnMnistm{}.pth.tar".format(opt.continue_epoch))
            
        elif opt.source == "SVHN":
            if opt.train_on_target == "yes":
                torch.save(my_net.state_dict(), os.path.join(opt.save_dir, "class_svhn_mnistm_trainOnMnistm{}.pth.tar".format(opt.continue_epoch)))
                print("Saving model: class_svhn_mnistm_trainOnMnistm{}.pth.tar".format(opt.continue_epoch))
            else:
                torch.save(my_net.state_dict(), os.path.join(opt.save_dir, "class_svhn_mnistm_trainOnSvhn{}.pth.tar".format(opt.continue_epoch)))
                print("Saving model: class_svhn_mnistm_trainOnSvhn{}.pth.tar".format(opt.continue_epoch))

        print("")
        opt.continue_epoch += opt.val_epoch


print("***** FINISHED TRAINING *****")
print("Best acc: %.2f%% on epoch %d" % (best_acc, best_epoch))
