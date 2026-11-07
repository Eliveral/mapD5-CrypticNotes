"""Project actual source pixels at their native resolution in one resampling."""
import json
from collections import OrderedDict
from threading import RLock
import cv2
import numpy as np
from .s1_catalog import read_image


class OriginalMap:
    def __init__(self, matcher):
        self.matcher = matcher
        self.catalog = matcher.catalog
        registration = self.catalog.root.parent.parent / 'config' / 's1_hd_registration.json'
        self.hd_to_template = json.loads(registration.read_text(encoding='utf-8'))
        self.recent = OrderedDict()
        self.lock=RLock()

    def source(self, floor):
        with self.lock:return self._source(floor)

    def _source(self,floor):
        key = floor['id']
        if key in self.recent:
            self.recent.move_to_end(key)
            return self.recent[key]
        source_id = 'hd_' + floor['id'].removeprefix('s1_')
        image = read_image(self.catalog.path(self.catalog.by_id[source_id]))
        matrix = np.float64(self.hd_to_template[floor['id']])
        backdrop = np.median(np.stack([image[0,0],image[0,-1],image[-1,0],image[-1,-1]]), axis=0)
        color=backdrop.astype(int)
        empty=cv2.inRange(image,tuple(map(int,color-3)),tuple(map(int,color+3)))
        image=cv2.bitwise_and(image,image,mask=cv2.bitwise_not(empty))
        self.recent[key]=(image,matrix,source_id)
        if len(self.recent)>3:self.recent.popitem(last=False)
        return image, matrix, source_id

    def render(self, result, size):
        image, matrix, source_id = self.source(result['floor'])
        transform = np.vstack([result['matrix'], [0,0,1]]) @ np.vstack([matrix, [0,0,1]])
        result['display_source'] = source_id
        return cv2.warpAffine(image, transform[:2], tuple(size), flags=cv2.INTER_LINEAR)
