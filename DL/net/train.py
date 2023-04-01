import torch
import time

from net.evaluation.metrics import precision_recall_f1
from net.evaluation.save import save_batch_training


def print_batch_train(names, inputs, targets, outputs, outputs_max):
    """
    :param names: list with dimension batch_size of the name of the images in the batch
    :param inputs: (batch_size, tensor for each image)tensor that contain the tensors for each image in the batch.
    The tensor for an image is tensor([ [[num, ... xW]] xH ]) xC, so (3, 256, 256) in our dataset
    :param targets: (1, batch_size) tensor that contain
    the ground truth labels for each image in the batch
    :param outputs: tensor with dimension of (batch_size,
    number_of_classes) where each row of the outputs tensor will contain the network's prediction of the corresponding
    image. Class is identified by the column index. The row is a tensor of dimension (1, number_of_classes)
    :param outputs_max: tensor of shape (1, batch_size)
    containing the index (alias class) of the maximum predicted probability for each image in the batch
    :return: print of batch info about names, inputs, targets and relative output and outputmax
    """

    print("----------------------------------------------------------------------------")
    print("    names  |          %s of batch length %d                     | %s ..." % (
        type(names), len(names), names[:5]))
    print("   inputs  | %s of shape %s | ... images tensors ..." % (type(inputs), inputs.shape))
    print(
        "  targets  |          %s of batch length %d             | %s ..." % (type(targets), len(targets), targets[:5]))
    print("   output  |          %s of batch length %d             | ... probabilities for each class of each image in "
          "the batch ... " % (type(outputs), len(outputs)))
    print("output max |          %s of batch length %d             | %s ... " % (type(outputs_max), len(outputs_max),
                                                                                 outputs_max[:5]))



def train(device, dataset, dataloader, tot_batch, net, optimizer, criterion, scheduler, epoch_number, path):
    """
    :param device: device that perform the training
    :param dataset: subset object on which we perform the training
    :param dataloader: dataloader of the dataset to analyze it batch by batch for one epoch
    :param tot_batch: total number of batch
    :param net: network to train
    :param optimizer: evaluate the derivatives
    :param criterion: loss function
    :param scheduler: update the learning rate
    :param epoch_number: number of the epoch
    :param path: path to save the debug info
    :return: average loss, accuracy of the epoch, average f1 score
    """

    # switch to train mode
    net.train()

    # reset performance measures
    loss_sum = 0.0
    correct_sum = 0
    train_precision_sum = 0.0
    train_recall_sum = 0.0
    train_f1_sum = 0.0

    # batch number counter
    batch_number = 1

    # 1 epoch = 1 complete loop over the dataset
    for batch in dataloader:  # dataloader split the data in batch and give us the possibility to loop over the batch

        # keep track of the time of the batch
        t_batch = time.time()

        # get data from dataloader
        names, inputs, targets = batch['filename'], batch['image'], batch['label']

        # move data to device
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        # zero the parameter gradients
        optimizer.zero_grad()  # if not, the new gradient is summed to it

        # forward pass
        outputs = net(inputs)

        # calculate loss
        loss = criterion(outputs, targets)

        # loss gradient backpropagation
        loss.backward()

        # net parameters update
        optimizer.step()

        # accumulate loss
        loss_sum += loss.item()

        # obtain the indices of the classes with the max probability
        outputs_max = torch.argmax(outputs, dim=-1)

        # compare the indices of the classes with the max probability with the targets and count the ones that match
        correct_batch = outputs_max.eq(targets).sum().float()
        correct_sum += correct_batch

        # calculate precision, recall, f1 score with average on the labels
        train_precision_batch, train_recall_batch, train_f1_batch, _, _, _ = precision_recall_f1(y_true=targets,
                                                                                                 y_pred=outputs_max,
                                                                                                 print_flag=False)

        # accumulate metrics
        train_precision_sum += train_precision_batch
        train_recall_sum += train_recall_batch
        train_f1_sum += train_f1_batch

        # save the batch infos
        save_batch_training(path=path,
                            epoch_number=epoch_number,
                            batch_number=batch_number,
                            names=names,
                            targets=targets,
                            outputs_max=outputs_max,
                            outputs=outputs,
                            precision=train_precision_batch,
                            recall=train_recall_batch,
                            f1=train_f1_batch,
                            correct=correct_batch)

        print(">>> epoch %d: batch %d / %d | Loss: %.2f | Precision: %.2f | Recall: %.2f | F1: %.2f | Correct: %d | Time: "
              "%d s "
              % (epoch_number, batch_number, tot_batch, loss, train_precision_batch, train_recall_batch,
                 train_f1_batch, correct_batch, time.time() - t_batch))

        # check the batch
        # print_batch_train(names, inputs, targets, outputs, outputs_max)
        batch_number += 1

    # step learning rate scheduler at the end of the epoch
    scheduler.step()

    # evaluate metrics
    average_loss_epoch = loss_sum / len(dataloader)
    accuracy_epoch = 100. * correct_sum / len(dataset)
    average_precision_epoch = train_precision_sum / len(dataloader)
    average_recall_epoch = train_recall_sum / len(dataloader)
    average_f1_epoch = train_f1_sum / len(dataloader)

    # return average loss, accuracy and average f1
    return average_loss_epoch, accuracy_epoch.cpu().detach().numpy(), average_precision_epoch, average_recall_epoch, average_f1_epoch
