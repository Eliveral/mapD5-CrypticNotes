"""Keep the map icon independent of the legend and previously painted pixels."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.s1_blackboard import board_point
from src.s1_live_match import map_area


def draw_board(frame,center):
    x,y=center
    for a,b in [((-7,-7),(7,-8)),((-7,-7),(-7,2)),((7,-8),(7,8)),
                ((-8,2),(8,2)),((-5,3),(-8,8)),((4,3),(7,7))]:
        cv2.line(frame,(x+a[0],y+a[1]),(x+b[0],y+b[1]),(214,191,172),2,cv2.LINE_AA)


def test_map_board_ignores_legend_and_hidden_pixels():
    frame=np.full((1377,2560,3),(70,85,100),np.uint8)
    draw_board(frame,(465,970))
    assert board_point(frame,map_area) is None
    draw_board(frame,(1550,840))
    point=board_point(frame,map_area)
    assert point is not None and np.linalg.norm(point-[1550,840])<3
    mask=np.full(frame.shape[:2],255,np.uint8)
    mask[825:855,1535:1565]=0
    assert board_point(frame,map_area,mask) is None
    draw_board(frame,(1630,866))
    assert board_point(frame,map_area) is None
