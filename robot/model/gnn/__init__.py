#!/usr/bin/env python
# coding=utf-8

from .simple import GNNNet, GEATNet
from .utils import MySequential
from .itergnn import Baseline as IterBaseline, GenBaseline as IterGenBaseline
from .itergnn import ResBaseline as ResIterBaseline, ResGenBaseline as ResIterGenBaseline
from .itergnn import CatBaseline as CatIterBaseline, CatGenBaseline as CatIterGenBaseline
from .edge_itergnn import EdgeBaseline as IterEdgeBaseline
from .pooling import TopKPooling, SoftmaxPooling, TopKGraphPooling, IdentityPooling
from .pooling import KTopPooling, KTopNewPooling, KTopNewGraphPooling
from .cnn import CNN
from .build_graph import Image2Graph

