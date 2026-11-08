"""Small UI appearance check: map presence is independent of map-key parity."""
from __future__ import annotations
import json
import cv2
import numpy as np
from .s1_catalog import read_image


def edge_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.Canny(gray, 25, 60)


class MapVisibility:
    def __init__(self, root):
        self.root = root
        self.templates = {}
        self.height = 688
        index = json.loads((root/'index.json').read_text(encoding='utf-8'))
        for entry in index['templates']:
            im = read_image(root/entry['file'])
            self.templates[entry['id']] = [edge_image(cv2.resize(im, None, fx=s, fy=s)) for s in (.85, 1., 1.15)]

    def inspect(self, image, fast=False):
        scale = self.height/image.shape[0]
        resized = cv2.resize(image, (round(image.shape[1]*scale), self.height), interpolation=cv2.INTER_AREA)
        width=resized.shape[1]
        left=int(width*.87)
        # Both buttons occupy the right strip. Keep the full-screen edge image
        # lazy for the existing wide fallback when a button is missing.
        hud=edge_image(resized[:int(self.height*.47)+3,left-3:])[:,3:]
        broad=None
        scores = {}
        for name, templates in self.templates.items():
            scores[name] = 0.
            # Stable right-hand controls are independent of map pan/zoom.
            ya, yb = (0., .18) if name == 'close' else (.24, .47)
            region=hud[int(self.height*ya):int(self.height*yb)]
            for t in templates:
                if min(t.shape) < 5 or t.shape[0] > region.shape[0] or t.shape[1] > region.shape[1]:
                    continue
                _, value, _, _ = cv2.minMaxLoc(cv2.matchTemplate(region, t, cv2.TM_CCOEFF_NORMED))
                scores[name] = max(scores[name], float(value))
            if not fast and scores[name] < (.50 if name == 'close' else .48):
                if broad is None:
                    broad=edge_image(resized)[:int(self.height*.84),int(width*.65):]
                for t in templates:
                    if t.shape[0] <= broad.shape[0] and t.shape[1] <= broad.shape[1]:
                        _, value, _, _ = cv2.minMaxLoc(cv2.matchTemplate(broad, t, cv2.TM_CCOEFF_NORMED))
                        scores[name] = max(scores[name], float(value))
        return dict(visible=scores.get('close', 0) >= .50 and scores.get('zoom_plus', 0) >= .48, scores=scores)
