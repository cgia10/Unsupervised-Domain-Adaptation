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
from model_dann import CNNModel
from evaluate_dann import evaluate


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
parser.add_argument("--model_name", type=str, default="dann_mnistm_svhn1.pth.tar", help="name of saved model to load")
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
    # Train sets
    source_train = data_train.MNISTM(opt, "train")
    target_train = data_train.SVHN(opt, "train")

    # Test sets
    target_test = data_train.SVHN(opt, "test")

# Source=SVHN, Target=MNISTM
elif opt.source == "SVHN":
    # Train sets
    source_train = data_train.SVHN(opt, "train")
    target_train = data_train.MNISTM(opt, "train")

    # Test sets
    target_test = data_train.MNISTM(opt, "test")

# Generate dataloaders

# Source dataloader
dataloader_source_train = torch.utils.data.DataLoader(
    dataset=source_train,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu
)

# Target dataloaders
dataloader_target_train = torch.utils.data.DataLoader(
    dataset=target_train,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu
)

dataloader_target_test = torch.utils.data.DataLoader(
    dataset=target_test,
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
    len_dataloader = min(len(dataloader_source_train), len(dataloader_target_train))
    data_source_iter = iter(dataloader_source_train)
    data_target_iter = iter(dataloader_target_train)

    i = 0
    while i < len_dataloader:
        
        optimizer.zero_grad()
        p = float(i + epoch * len_dataloader) / opt.n_epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1


        # ----------------
        #  Source Dataset
        # ----------------

        # Load next source image and ground truth class
        data_source = data_source_iter.next()
        s_img, s_label = data_source

        # Create tensors for domain and class labels
        batch_size = len(s_label)
        input_img = torch.FloatTensor(batch_size, opt.channels, opt.img_size, opt.img_size)
        class_label = torch.LongTensor(batch_size)
        domain_label = torch.zeros(batch_size).long()
        
        # Move to GPU
        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()
            domain_label = domain_label.cuda()

        # Tensor resizing
        input_img.resize_as_(s_img).copy_(s_img)
        class_label.resize_as_(s_label).copy_(s_label)

        # Model output and classification/domain loss
        class_output, domain_output = my_net(input_data=input_img, alpha=alpha)
        err_s_label = loss_criteria(class_output, class_label)         
        err_s_domain = loss_criteria(domain_output, domain_label)     

        
        # ----------------
        #  Target Dataset
        # ----------------

        # Load next target image
        data_target = data_target_iter.next()
        t_img, _ = data_target

        # Create tensors for domain and class labels
        batch_size = len(t_img)
        input_img = torch.FloatTensor(batch_size, opt.channels, opt.img_size, opt.img_size)
        domain_label = torch.ones(batch_size)
        domain_label = domain_label.long()

        # Move to GPU
        if cuda:
            t_img = t_img.cuda()
            input_img = input_img.cuda()
            domain_label = domain_label.cuda()

        # Tensor resizing
        input_img.resize_as_(t_img).copy_(t_img)

        # Model output and domain loss
        _, domain_output = my_net(input_data=input_img, alpha=alpha)
        err_t_domain = loss_criteria(domain_output, domain_label)    


        # -----------------
        #  Backpropagation
        # -----------------

        err = err_t_domain + err_s_domain + err_s_label
        err.backward()
        optimizer.step()


        # ---------------
        #  Training Info
        # ---------------

        iters += 1
        i += 1

        print ("[Epoch: %d/%d] [Batch: %d/%d] [SRC: class_loss %.4f, domain_loss %.4f] [TRGT: domain_loss %.4f]" \
              % (epoch, opt.n_epochs, i, len_dataloader, err_s_label.data.cpu().numpy(),
                 err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
        
        
    # ------------------
    #  Acc/Model Saving
    # ------------------

    if epoch % opt.val_epoch == 0:     
        
        # Accuracy on target test set
        correct, total, target_acc = evaluate(my_net, dataloader_target_test)

        if target_acc > best_acc:
            best_acc = target_acc
            best_epoch = epoch

        # Print info
        print("")
        print("********** Current: epoch %d   Total: epoch %d **********" % (epoch, opt.continue_epoch))
        print("ACC: %d/%d correct, acc = %.2f%%" % (correct, total, target_acc))
        
        # Save model
        if opt.source == "MNISTM":
            torch.save(my_net.state_dict(), os.path.join(opt.save_dir, "dann_mnistm_svhn{}.pth.tar".format(opt.continue_epoch)))
            print("Saving model: dann_mnistm_svhn{}.pth.tar".format(opt.continue_epoch))
        elif opt.source == "SVHN":
            torch.save(my_net.state_dict(), os.path.join(opt.save_dir, "dann_svhn_mnistm{}.pth.tar".format(opt.continue_epoch)))
            print("Saving model: dann_svhn_mnistm{}.pth.tar".format(opt.continue_epoch))

        print("")
        opt.continue_epoch += opt.val_epoch


print("***** FINISHED TRAINING *****")
print("Best acc: %.2f%% on epoch %d" % (best_acc, best_epoch))
