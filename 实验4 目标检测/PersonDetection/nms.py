import numpy as np
import cv2


#手动实现nms代码
def non_max_suppression_slow(boxes, overlapThresh):
    # If the input is empty, return an empty list
    if len(boxes) == 0:
        return []

    # Initialize the list of picked indexes
    pick = []

    # Get the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute the area of the bounding boxes and sort the bounding boxes by the bottom-right y-coordinate
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        # Grab the last index in the indexes list and add the index value to the list of picked indexes
        # then initialize the suppression list using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        # Loop over all indexes in the indexes list
        for pos in range(0, last):
            # Grab the current index
            j = idxs[pos]

            # Find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # Compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # Compute the ratio of overlap between the computed bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]

            # If there is a large overlap, suppress the current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)

        # Delete all indexes from the index list that are in the suppression list
        idxs = np.delete(idxs, suppress)

    # Return only the bounding boxes that were picked
    return boxes[pick]

# Construct a list of images that will be checked along with their respective bounding boxes
images = [
    ("audrey.jpg", np.array([
        (12, 84, 140, 212),
        (24, 84, 152, 212),
        (36, 84, 164, 212),
        (12, 96, 140, 224),
        (24, 96, 152, 224),
        (24, 108, 152, 236)])),
    ("bksomels.jpg", np.array([
        (114, 60, 178, 124),
        (120, 60, 184, 124),
        (114, 66, 178, 130)])),
    ("gpripe.jpg", np.array([
        (12, 30, 76, 94),
        (12, 36, 76, 100),
        (72, 36, 200, 164),
        (84, 48, 212, 176)]))]

# Loop over the images
for (imagePath, boundingBoxes) in images:
    # Load the image and clone it
    print("[x] %d initial bounding boxes" % (len(boundingBoxes)))
    image = cv2.imread(imagePath)
    orig = image.copy()

    # Loop over the bounding boxes for each image and draw them
    for (startX, startY, endX, endY) in boundingBoxes:
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)

    # Apply non-maxima suppression to the bounding boxes
    pick = non_max_suppression_slow(boundingBoxes, 0.3)
    print("[x] after applying non-maximum, %d bounding boxes" % (len(pick)))

    # Draw the bounding boxes that were picked
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Display the results
    cv2.imshow("Original", orig)
    cv2.imshow("After NMS", image)
    cv2.waitKey(0)