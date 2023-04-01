import csv
import time

import torch


def print_batch_test(names, inputs, outputs, outputs_max):
    """
    :param names: list with dimension batch_size of the name of the images in the batch
    :param inputs: (batch_size, tensor for each image) tensor that contain the tensors for each image in the batch.
    The tensor for an image is tensor([ [[num, ... xW]] xH ]) xC, so (3, 256, 256) in our dataset
    :param outputs:
    tensor with dimension of (batch_size, number_of_classes) where each row of the outputs tensor will contain the
    network's prediction of the corresponding image. Class is identified by the column index. The row is a tensor of
    dimension (1, number_of_classes)
    :param outputs_max: tensor of shape (1, batch_size) containing the index (alias
    class) of the maximum predicted probability for each image in the batch
    :return: print of batch info about names,
    inputs and relative output and outputmax
    """

    print("----------------------------------------------------------------------------")
    print("    names  |          %s of batch length %d                     | %s ... " % (
        type(names), len(names), names[:3]))
    print("   inputs  | %s of shape %s  | ... images tensors ..." % (type(inputs), inputs.shape))
    print(
        "output  |          %s of batch length %d             | ... probabilities for each class of each image in the "
        "batch ... " % (type(outputs), len(outputs)))
    print("output max |          %s of batch length %d             | %s ... " % (type(outputs_max), len(outputs_max),
                                                                                 outputs_max[:5]))


def test(device, dataset, dataloader, tot_batch, net, path):
    """
    :param device: device that perform the training
    :param dataset: subset object on which we perform the training
    :param dataloader: dataloader of the dataset to analyze it batch by batch for one epoch
    :param tot_batch: total number of batch
    :param net: network to train
    :param path: path to save the scores
    :return: predictions tensor for the epoch. This function can be used both for validation and test set,
    since formally they do the same thing just on different
    """

    # switch to test mode
    net.eval()

    # initialize predictions
    predictions = torch.zeros(len(dataset), dtype=torch.int64)
    sample_counter = 0

    # do not accumulate gradients (faster)
    with torch.no_grad():

        # batch number counter
        batch_number = 1

        # define the writer to keep track of the probabilities predicted by the network for each image
        with open(path, "w") as file:
            writer = csv.writer(file)
            # the scores.csv store the filename, the ground truth, the label predicted (max score) and the score
            writer.writerow(['FILENAME', 'CLASS', 'LABEL', 'SCORE'])

            # test all batches
            for batch in dataloader:

                # keep track of the time batch
                t_batch = time.time()

                # get data from dataloader [ignore labels/targets as they are not used in test mode]
                inputs = batch['image']

                # move data to device
                inputs = inputs.to(device, non_blocking=True)

                # forward pass
                outputs = net(inputs)

                # for the batch obtain the indices (alias classes) with the max probability and put it into a tensor
                outputs_max = torch.argmax(outputs, dim=-1)
                for output in outputs_max:
                    predictions[sample_counter] = output
                    sample_counter += 1

                # write the infos on path
                for index in range(len(batch['label'])):
                    writer.writerow([batch['filename'][index], int(batch['label'][index]), int(outputs_max[index]),
                                     float("{:.4f}".format(outputs[index, outputs_max[index]]))])

                print(">>> batch %d / %d | Time: %d s " % (batch_number, tot_batch, time.time() - t_batch))

                # check the batch
                # print_batch_test(batch['filename'], inputs, outputs, outputs_max)
                batch_number += 1

    return predictions
