import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # ID of the GPU you want to use

import time
import numpy as np
import argparse
import torch.cuda
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

from net.dataset.dataset import GalaxyDataset
from net.dataset.dataset_split import dataset_split
from net.dataset.dataset_trasform import dataset_transform
from net.model.model_selector import model_selector
from net.reproducibility.reproducibility import reproducibility
from net.test import test
from net.train import train
from net.dataset.dataset_statistics import dataset_statistics
from net.evaluation.plot import plot_training_validation_accuracies, plot_f1
from net.evaluation.plot import plot_loss
from net.evaluation.metrics import accuracy, precision_recall_f1
from net.evaluation.plot import plot_ROC
from net.evaluation.save import save_errors, save_evaluation, save_evaluation_training
from net.evaluation.save import save_experiment_parameters


class FocalLoss(torch.nn.Module):
    """ Focal Loss implementation """
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return torch.mean(focal_loss)


def main():
    """
    Galaxy Morphology Classification with Deep Learning techniques

    University of Cassino and Southern Lazio

    Computing Engineering (LM-32)

    Exam: Machine and Deep Learning

    Professors: Alessandro Bria and Claudio Marrocco

    Students: Alessio Miele and Giulio Russo
    """

    # ========
    # PARSING |
    # ========

    # create the argument parser
    parser = argparse.ArgumentParser()

    # add the arguments to the parser
    parser.add_argument('--file', type=str, help='file name')
    parser.add_argument('--model', type=str, help='model: resnet#, vgg#, densenet#')
    parser.add_argument('--bs', type=int, help='batch size')
    parser.add_argument('--ep', type=int, help='epochs')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--ss', type=float, help='learning rate step size')
    parser.add_argument('--gamma', type=float, help='learning rate gamma')
    parser.add_argument('--m', type=float, help='momentum')
    parser.add_argument('--opt', type=str, help='optimizer: sgd/adam/rms')
    parser.add_argument('--loss', type=str, help='ce: Cross Entropy, f: Focal Loss')
    parser.add_argument('--test', action='store_true', help='Test = include --test, Train = does not include --test')
    parser.add_argument('--statistics', action='store_true', help='Include to evaluate the statistics of the dataset')
    parser.add_argument('--dataset', type=str, help='10/4 classes: images-10-4, if augmented: images-10-4-augmented'
                                                    '3 classes: images-3, if augmented: images-3-augmented')

    # parse the arguments
    args = parser.parse_args()

    # =====
    # PATH |
    # =====

    where_dataset = "dataset"
    dataset_images_path = os.path.join(where_dataset, args.dataset)
    dataset_annotations_path = os.path.join(where_dataset, "annotations")
    dataset_statistics_path = os.path.join(dataset_annotations_path, "statistics.csv")
    dataset_annotation_split_path = os.path.join(dataset_annotations_path, args.file)

    # ===========
    # PARAMETERS |
    # ===========

    file_name = args.file.replace(".csv", '')  # file name without the extension. e.g. 'all.csv' -> 'all'
    batch_size = args.bs
    epochs = args.ep
    learning_rate = args.lr
    lr_step_size = args.ss
    lr_gamma = args.gamma
    momentum = args.m
    test_mode = args.test
    statistics_flag = args.statistics

    # ==============
    # NETWORK MODEL |
    # ==============

    # define the number of the output neurons
    if '10' in file_name:
        num_class = 10
    elif '4' in file_name:
        num_class = 4
    elif '3' in file_name:
        num_class = 3
    else:
        print("ERROR: wrong class number in NETWORK MODEL section in main.py",
              file=sys.stderr)

    # instance the model
    model_name = args.model
    net = model_selector(model_name=model_name,
                         class_number=num_class)

    # ==========
    # OPTIMIZER |
    # ==========

    if args.opt == "sgd":
        optimizer = SGD(net.parameters(),
                        lr=learning_rate,
                        momentum=momentum)
    elif args.opt == "adam":
        optimizer = Adam(net.parameters(),
                         lr=learning_rate)
    elif args.opt == "rms":
        optimizer = RMSprop(net.parameters(),
                            lr=learning_rate)
    else:
        print("ERROR: wrong optimizer in OPTIMIZER section in main.py",
              file=sys.stderr)

    # ==========
    # SCHEDULER |
    # ==========

    scheduler = StepLR(optimizer=optimizer,
                       step_size=lr_step_size,
                       gamma=lr_gamma)

    # ==========
    # CRITERION |
    # ==========

    if args.loss == 'ce':
        criterion = CrossEntropyLoss()
    elif args.loss == 'f':
        # criterion = torchvision.ops.focal_loss # !!! introduced in PyTorch version 1.8.0 but I have 0.14.1 !!!
        criterion = FocalLoss()
    else:
        print("ERROR: wrong loss in CRITERION section in main.py",
              file=sys.stderr)

    # ==============
    # EXPERIMENT ID |
    # ==============

    experiment_ID = \
        "%s-%s|epochs=%d|batch_size=%d|learning_rate=%.4f|step_size=%d|gamma=%.1f|momentum=%.1f|criterion=%s" \
        "|optimizer=%s" \
        % (model_name, file_name, epochs, batch_size, learning_rate, lr_step_size, lr_gamma, momentum,
           type(criterion).__name__, type(optimizer).__name__)

    # create the directory of the experiment
    experiment_path = os.path.join("experiments", experiment_ID)
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    # best model
    results_path = os.path.join(experiment_path, experiment_ID + ".tar")

    # .csv with all the parameters
    hyperparameter_path = os.path.join(experiment_path, experiment_ID + ".csv")

    # resume folder in the experiment folder
    experiment_resume_path = os.path.join(experiment_path, "resume")
    if not os.path.exists(experiment_resume_path):
        os.mkdir(experiment_resume_path)

    # resume path
    resume_path = os.path.join(experiment_resume_path, experiment_ID + ".tar")

    # loss, train-validation accuracies, errors and scores path
    experiment_plot_path = os.path.join(experiment_path, "plot")
    if not os.path.exists(experiment_plot_path):
        os.mkdir(experiment_plot_path)

    loss_path = os.path.join(experiment_plot_path, "loss.png")
    train_validation_accuracies_path = os.path.join(experiment_plot_path,
                                                    "train-validation-accuracy.png")  # accuracy image path
    f1_path = os.path.join(experiment_plot_path, "f1.png")

    # test infos
    errors_path = os.path.join(experiment_path, "errors.csv")  # errors of test
    test_scores_path = os.path.join(experiment_path, "test-scores.csv")  # scores of the predictions
    test_metrics_path = os.path.join(experiment_path, "test-metrics.csv")

    # training debug path
    experiment_training_debug_path = os.path.join(experiment_path, "debug-training")
    if not os.path.exists(experiment_training_debug_path):
        os.mkdir(experiment_training_debug_path)

    experiment_validation_debug_path = os.path.join(experiment_path, "debug-validation")
    if not os.path.exists(experiment_validation_debug_path):
        os.mkdir(experiment_validation_debug_path)

    # save all the experiments parameters
    save_experiment_parameters(path=hyperparameter_path,
                               file_name=file_name,
                               model_name=model_name,
                               epochs=epochs,
                               batch_size=batch_size,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               criterion=criterion,
                               learning_rate=learning_rate,
                               lr_step_size=lr_step_size,
                               lr_gamma=lr_gamma,
                               momentum=momentum)

    # =======
    # DEVICE |
    # =======

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ================
    # REPRODUCIBILITY |
    # ================

    reproducibility(seed=0)

    # =============
    # LOAD DATASET |
    # =============

    dataset = GalaxyDataset(annotations_path=dataset_annotation_split_path,
                            images_path=dataset_images_path,
                            transforms=None)

    # ==============
    # DATASET SPLIT |
    # ==============

    dataset_train, dataset_validation, dataset_test = dataset_split(dataset=dataset,
                                                                    annotations_path=dataset_annotation_split_path)

    # =============
    # DATA LOADERS |
    # =============

    dataloader_train = DataLoader(dataset=dataset_train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=2,
                                  pin_memory=True)

    dataloader_validation = DataLoader(dataset=dataset_validation,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=2,
                                       pin_memory=True)

    dataloader_test = DataLoader(dataset=dataset_test,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=2,
                                 pin_memory=True)

    # ===================
    # DATASET TRANSFORMS |
    # ===================

    # get the dataset statistics if --statistics is specified
    if statistics_flag:
        dataset_statistics(dataset_train)

    # standardization
    transforms_train, transforms_validation, transforms_test = dataset_transform(norm='std',
                                                                                 statistics_path=dataset_statistics_path)
    dataset_train.dataset.transforms = transforms_train
    dataset_validation.dataset.transforms = transforms_validation
    dataset_test.dataset.transforms = transforms_test

    # convert the targets of validation and test set into a unique tensor to evaluate performance
    targets_validation = torch.tensor([sample['label'] for sample in dataset_validation])
    targets_test = torch.tensor([sample['label'] for sample in dataset_test])

    # ==============
    # TRAINING MODE |
    # ==============

    # --test not specified train a new model and save it
    if not test_mode:

        print("\n----------------------------------")
        print("           TRAINING MODE          |")
        print("----------------------------------")

        # reset performance monitors
        losses = []
        train_accuracies = []
        train_f1 = []
        validation_accuracies = []
        validation_f1 = []
        epoch_vector = []

        # move net to device
        net.to(device)

        print("\nExperiment ID: %s " % experiment_ID)

        # keep track of the time of the experiment
        t_experiment = time.time()

        # for each epoch
        for epoch in range(1, epochs + 1):
            print("\nEPOCH: %d" % epoch)

            # keep track of the time of the epoch and training
            t_epoch_start = time.time()

            # =========
            # TRAINING |
            # =========

            # train the network for each epoch
            print("... training ...")
            avg_loss, train_accuracy_value, avg_precision, avg_recall, avg_f1 = train(device=device,
                                                                                      dataset=dataset_train,
                                                                                      dataloader=dataloader_train,
                                                                                      tot_batch=round(
                                                                                          len(dataset_train) / batch_size),
                                                                                      net=net,
                                                                                      optimizer=optimizer,
                                                                                      criterion=criterion,
                                                                                      scheduler=scheduler,
                                                                                      epoch_number=epoch,
                                                                                      path=experiment_training_debug_path)

            print(
                "... end training >>> Average Loss: %.2f | Training accuracy: %.2f %% | Average Precision: %.2f | "
                "Average Recall: %.2f | Average F1: %.2f | Time: %d s "
                "\n "
                % (avg_loss, train_accuracy_value, avg_precision, avg_recall, avg_f1, time.time() - t_epoch_start))

            # save all
            train_metrics_path = os.path.join(experiment_training_debug_path, "e=" + str(epoch) + "|metrics.csv")
            save_evaluation_training(
                path=train_metrics_path,
                accuracy=train_accuracy_value,
                precision=avg_precision,
                recall=avg_recall,
                f1=avg_f1)

            # ===========
            # VALIDATION |
            # ===========

            # keep track of the time of the validation
            t_validation_start = time.time()

            # validate the model every epoch with the test function
            print("... validation ...")
            validation_scores_path = os.path.join(experiment_validation_debug_path,
                                                  "e=" + str(epoch) + "|validation-scores.csv")
            predictions_validation = test(device=device,
                                          dataset=dataset_validation,
                                          dataloader=dataloader_validation,
                                          tot_batch=round(len(dataset_validation) / batch_size),
                                          net=net,
                                          path=validation_scores_path)

            print("... end validation >>>  Time: %d s" % (time.time() - t_validation_start))

            # Accuracy on validation
            validation_accuracy_value, validation_accuracy_per_class = accuracy(predictions=predictions_validation,
                                                                                targets=targets_validation,
                                                                                print_flag=True)

            # Precision, Recall, F1 score on validation
            validation_precision_mean, validation_recall_mean, validation_f1_mean, \
            validation_precision_per_class, validation_recall_per_class, validation_f1_per_class \
                = precision_recall_f1(
                y_true=targets_validation,
                y_pred=predictions_validation,
                print_flag=True)

            # save all
            validation_metrics_path = os.path.join(experiment_validation_debug_path,
                                                   "e=" + str(epoch) + "|validation-metrics.csv")
            save_evaluation(path=validation_metrics_path,
                            accuracy_value=validation_accuracy_value,
                            accuracy=validation_accuracy_per_class,
                            precision_value=validation_precision_mean,
                            recall_value=validation_recall_mean,
                            f1_value=validation_f1_mean,
                            precision=validation_precision_per_class,
                            recall=validation_recall_per_class,
                            f1=validation_f1_per_class)

            # ========
            # METRICS |
            # ========

            losses.append(avg_loss)
            train_accuracies.append(train_accuracy_value)
            train_f1.append(avg_f1)
            validation_accuracies.append(validation_accuracy_value)
            validation_f1.append(validation_f1_mean)
            epoch_vector.append(epoch)

            # =====
            # PLOT |
            # =====

            # plot loss
            plot_loss(criterion=criterion,
                      epoch_vector=epoch_vector,
                      losses=losses,
                      epochs=epochs,
                      path=loss_path)

            # plot train and validation accuracies average
            plot_training_validation_accuracies(epoch_vector=epoch_vector,
                                                train_accuracies=train_accuracies,
                                                validation_accuracies=validation_accuracies,
                                                epochs=epochs,
                                                path=train_validation_accuracies_path)

            # plot F1 score
            plot_f1(epoch_vector=epoch_vector,
                    f1_training=train_f1,
                    f1_validation=validation_f1,
                    epochs=epochs,
                    path=f1_path)

            # ===========
            # BEST MODEL |
            # ===========

            # save model if validation performance has improved
            if (epoch - 1) == np.argmax(validation_f1):
                torch.save({
                    'net': net,
                    'accuracy': max(validation_f1),
                    'epoch': epoch,
                }, results_path)

            # =======
            # RESUME |
            # =======

            torch.save({
                'net': net,
                'criterion': criterion,
                'scheduler': scheduler,
                'optimizer': optimizer,
                'epoch': epoch
            }, resume_path)

        # show the time at the end of the experiment
        elapsed_time = time.time() - t_experiment
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\ntime: {int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds")

    # ==========
    # TEST MODE |
    # ==========

    # --test specified load the saved model with the corresponding parameters and test it
    else:

        print("\n----------------------------------")
        print("             TEST MODE            |")
        print("----------------------------------")

        # load test_mode model
        pretrained_model = torch.load(results_path, map_location=lambda storage, loc: storage)
        net = pretrained_model['net']

        print("\nLoaded test_mode model: %s " % experiment_ID)
        print("Best F1: %.2f at epoch %d " % (pretrained_model['accuracy'].item(), pretrained_model['epoch']))

        # move net to device
        net.to(device)

        # =====
        # TEST |
        # =====

        print("\n... testing ...")
        predictions_test = test(device=device,
                                dataset=dataset_test,
                                dataloader=dataloader_test,
                                tot_batch=round(len(dataset_test) / batch_size),
                                net=net,
                                path=test_scores_path)

        # ===========
        # EVALUATION |
        # ===========

        # Accuracy on test
        test_accuracy_value, test_accuracy_per_class = accuracy(predictions=predictions_test,
                                                                targets=targets_test,
                                                                print_flag=True)

        # Precision, Recall, F1 score on test
        test_precision_mean, test_recall_mean, test_f1_mean, \
        test_precision_per_class, test_recall_per_class, test_f1_per_class \
            = precision_recall_f1(
            y_true=targets_test,
            y_pred=predictions_test,
            print_flag=True)

        # save all
        save_evaluation(path=test_metrics_path,
                        accuracy_value=test_accuracy_value,
                        accuracy=test_accuracy_per_class,
                        precision_value=test_precision_mean,
                        recall_value=test_recall_mean,
                        f1_value=test_f1_mean,
                        precision=test_precision_per_class,
                        recall=test_recall_per_class,
                        f1=test_f1_per_class)

        # ROC curve
        # ROC(path_scores=os.path.join("results", "scores.csv"))

        # save errors
        save_errors(path=errors_path,
                    predictions=predictions_test,
                    targets=targets_test,
                    dataset=dataset_test)


if __name__ == '__main__':
    main()
