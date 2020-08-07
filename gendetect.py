import numpy as np
import time
import cv2


def generalDetect(image):
    # load the class labels from disk
    rows = open("lib/model/synset_words.txt").read().strip().split("\n")
    classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

    # our CNN requires fixed spatial dimensions for our input image(s)
    # so we need to ensure it is resized to 224x224 pixels while
    # performing mean subtraction (104, 117, 123) to normalize the input;
    # after executing this command our "blob" now has the shape:
    # (1, 3, 224, 224)
    blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe("lib/proto/bvlc_googlenet.prototxt", "lib/model/bvlc_googlenet.caffemodel")

    # set the blob as input to the network and perform a forward-pass to
    # obtain our output classification
    net.setInput(blob)
    start = time.time()
    preds = net.forward()
    end = time.time()
    print("[INFO] classification took {:.5} seconds".format(end - start))

    # sort the indexes of the probabilities in descending order (higher
    # probabilitiy first) and grab the top-10 predictions
    idxs = np.argsort(preds[0])[::-1][:10]

    res = []
    # loop over the top-10 predictions and display them
    for (i, idx) in enumerate(idxs):
        # draw the top prediction on the input image
        # if i == 0:
        #     text = "Label: {}, {:.2f}%".format(classes[idx],
        #                                        preds[0][idx] * 100)
        res.append({'id': i+1, 'name': classes[idx], 'value': int(preds[0][idx] * 100)})

    return res


# # load the input image from disk
# image = cv2.imread("images/p1.jpg")
#
# res = generalDetect(image)
#
# print(res)

