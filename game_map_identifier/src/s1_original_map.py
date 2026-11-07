"""Project actual source pixels at their native resolution in one resampling."""
import json
import cv2
import numpy as np
from .s1_catalog import read_image


class OriginalMap:
    def __init__(self, matcher):
        self.matcher = matcher
        self.catalog = matcher.catalog
        registration = self.catalog.root.parent.parent / 'config' / 's1_hd_registration.json'
        self.hd_to_template = json.loads(registration.read_text(encoding='utf-8'))
        self.last = None

    def source(self, floor):
        key = floor['id']
        if self.last and self.last[0] == key:
            return self.last[1:]
        source_id = 'hd_' + floor['id'].removeprefix('s1_')
        image = read_image(self.catalog.path(self.catalog.by_id[source_id]))
        matrix = np.float64(self.hd_to_template[floor['id']])
        backdrop = np.median(np.stack([image[0,0],image[0,-1],image[-1,0],image[-1,-1]]), axis=0)
        keep = np.max(np.abs(image.astype(np.int16)-backdrop.astype(np.int16)), axis=2) > 3
        image[~keep] = 0
        self.last = (key, image, matrix, source_id)
        return image, matrix, source_id

    def render(self, result, size):
        image, matrix, source_id = self.source(result['floor'])
        transform = np.vstack([result['matrix'], [0,0,1]]) @ np.vstack([matrix, [0,0,1]])
        result['display_source'] = source_id
        return cv2.warpAffine(image, transform[:2], tuple(size), flags=cv2.INTER_LINEAR)
