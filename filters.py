import matplotlib.image as mpimg
import cv2
import numpy as np


class Filter:
    def __init__(self, path, coord, h_points, w_points, offset=(0, 0), padding=(0, 0)):
        """ Initialize a filter object with the given parameters
        Args:
            path: String, path to the filter image
            coord: int, the index of the point in the keypoints array that corresponds to
                  the top lefthand corner of the filter
            h_points: tuple of two integers specifing the top and bottom of the filter
            w_points: tuple of two integers specifing the left and right of the filter
            offset: tuple of two integers for x offset and y offset
            padding: tuple of two integers for x padding and y padding
        Return: None
        """
        self.img = mpimg.imread(path, -1)
        self.coord = coord
        self.h_points = h_points
        self.w_points = w_points
        self.offset = offset
        self.padding = padding

    def apply(self, keypts, roi):
        """ Apply the filter given the keypts on the given frame
        Args:
            keypts: the keypoints array
            roi: the face image to which you want to apply the filter
        Return:
            roi: the region of interest with the filter on it
        """

        # use w_points to get the width of the filter
        w = int(abs(
            keypts[self.w_points[0]][0] - keypts[self.w_points[1]][0]
        )) + self.padding[0]

        # use h_points to get the height of the filter
        h = int(abs(
            keypts[self.h_points[0]][1] - keypts[self.h_points[1]][1]
        )) + self.padding[1]

        # resize the filter to the calculated width and height
        filter_img = cv2.resize(
            self.img, (w, h), interpolation=cv2.INTER_CUBIC)

        # get the coordinates with offset
        x = int(keypts[self.coord][0]) + self.offset[0]
        y = int(keypts[self.coord][1]) + self.offset[1]

        # slice the region that needs to be filtered
        ro_filter = roi[y:y + h, x:x + w]

        # get the non-transparent pixles
        non_trans = np.argwhere(filter_img[:, :, 3] > 0)

        if(non_trans.shape[0] > ro_filter.shape[0]*ro_filter.shape[1]):
            return roi

        # replace pixle values
        ro_filter[non_trans[:, 0], non_trans[:, 1], :3] = \
            filter_img[non_trans[:, 0], non_trans[:, 1], :3]
        roi[y:y + h, x:x + w] = ro_filter

        return roi
