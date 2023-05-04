import graph_tools as gt
from loguru import logger
import numpy as np
import pickle as pkl

# this code refers https://github.com/luckiezhou/DynamicTriad/blob/master/core/dataset/dataset_utils.py
# this code refers https://github.com/luckiezhou/DynamicTriad/blob/master/core/graph/graph.h
# this code refers https://github.com/luckiezhou/DynamicTriad/blob/master/core/graph/_test_graph.cpp

class Dataset():
	def __init__(self, graphs, time, step=1, stride=1, load_feature=False):
		self.graphs = []
		self._vertices = set()
		print("hi,time",time)
		for t in range(time):
			self.graphs.append(self.load_mygraph(graphs[t],graphs[time-1]))

		self.vertices = list(self._vertices)
		self.vertex2index = {n: i for i, n in enumerate(self.vertices)}

	def __len__(self):
		return self.graphs.__len__()

	def __getitem__(self, idx):
		return self.graphs.__getitem__(idx)


	def load_mygraph(self,cur_graph,base_graph):
		graph = gt.Graph(directed=False)
		for n in range(len(base_graph.nodes())):
			self._vertices.add(n)
			graph.add_vertex(n)
		print("cur_graph has edges:",len(cur_graph.edges()))
		for e in cur_graph.edges():
			graph.add_edge(e[0], e[1])
			graph.set_edge_weight(e[0], e[1], 1)
		return graph

	def load_graph(self, filename):
		graph = gt.Graph(directed=False)
		f = open(filename, 'r')

		for line in f.readlines():
			fields = line.split(' ')
			n = fields[0]

			if not n in self._vertices:
				self._vertices.add(n)

			if not graph.has_vertex(n):
				graph.add_vertex(n)

			for v, w in zip(fields[1::2], fields[2::2]):
				w = float(w)

				if v == n:
					logger.warning("loopback edge ({}, {}) detected".format(v, n))

				if not v in self._vertices:
					self._vertices.add(v)

				if not graph.has_vertex(v):
					graph.add_vertex(v)

				if not graph.has_edge(n, v):
					graph.add_edge(n, v)
					graph.set_edge_weight(n, v, w)

		f.close()
		return graph

	'''
	* Input: set of time-stamped graphs
	* Output: a merged graph
	* How to merge (example):
		- If the time length is 16, the time step is 4, and the time stride is 2, then the given dataset is merged as below:

		 merge({ 0, 1, 2, 3}) -> 0
		 merge({ 2, 3, 4, 5}) -> 1
		 merge({ 4, 5, 6, 7}) -> 2
		 merge({ 6, 7, 8, 9}) -> 3
		 merge({ 8, 9,10,11}) -> 4
		 merge({10,11,12,13}) -> 5
		 merge({12,13,14,15}) -> 6
	'''
	def merge(self, graphs):
		ret = gt.Graph(directed=False)

		for g in graphs:
			for v0, v1 in g.edges():
				w = g.get_edge_weight(v0, v1)

				if not ret.has_vertex(v0):
					ret.add_vertex(v0)
				if not ret.has_vertex(v1):
					ret.add_vertex(v1)
				if not ret.has_edge(v0, v1):
					ret.add_edge(v0, v1)
					ret.set_edge_weight(v0, v1, w)
				else:
					new_w = w + ret.get_edge_weight(v0, v1)
					ret.set_edge_weight(v0, v1, new_w)

		return ret

class GraphDataset():
	def __init__(self, graphs, time, step=1, stride=1, load_feature=False):
		self.graphs = []
		self._vertices = set()
		print("hi,time",time)
		for t in range(time):
			self.graphs.append(self.load_mygraph(graphs[t],graphs[time-1]))

		self.vertices = list(self._vertices)
		self.vertex2index = {n: i for i, n in enumerate(self.vertices)}

	def __len__(self):
		return self.graphs.__len__()

	def __getitem__(self, idx):
		return self.graphs.__getitem__(idx)


	def load_mygraph(self,cur_graph,base_graph):
		graph = gt.Graph(directed=False)
		for n in range(len(base_graph.nodes())):
			self._vertices.add(n)
			graph.add_vertex(n)
		print("cur_graph has edges:",len(cur_graph.edges()))
		for e in cur_graph.edges():
			graph.add_edge(e[0], e[1])
			graph.set_edge_weight(e[0], e[1], 1)
		return graph

	def load_graph(self, filename):
		graph = gt.Graph(directed=False)
		f = open(filename, 'r')

		for line in f.readlines():
			fields = line.split(' ')
			n = fields[0]

			if not n in self._vertices:
				self._vertices.add(n)

			if not graph.has_vertex(n):
				graph.add_vertex(n)

			for v, w in zip(fields[1::2], fields[2::2]):
				w = float(w)

				if v == n:
					logger.warning("loopback edge ({}, {}) detected".format(v, n))

				if not v in self._vertices:
					self._vertices.add(v)

				if not graph.has_vertex(v):
					graph.add_vertex(v)

				if not graph.has_edge(n, v):
					graph.add_edge(n, v)
					graph.set_edge_weight(n, v, w)

		f.close()
		return graph

	'''
	* Input: set of time-stamped graphs
	* Output: a merged graph
	* How to merge (example):
		- If the time length is 16, the time step is 4, and the time stride is 2, then the given dataset is merged as below:

		 merge({ 0, 1, 2, 3}) -> 0
		 merge({ 2, 3, 4, 5}) -> 1
		 merge({ 4, 5, 6, 7}) -> 2
		 merge({ 6, 7, 8, 9}) -> 3
		 merge({ 8, 9,10,11}) -> 4
		 merge({10,11,12,13}) -> 5
		 merge({12,13,14,15}) -> 6
	'''
	def merge(self, graphs):
		ret = gt.Graph(directed=False)

		for g in graphs:
			for v0, v1 in g.edges():
				w = g.get_edge_weight(v0, v1)

				if not ret.has_vertex(v0):
					ret.add_vertex(v0)
				if not ret.has_vertex(v1):
					ret.add_vertex(v1)
				if not ret.has_edge(v0, v1):
					ret.add_edge(v0, v1)
					ret.set_edge_weight(v0, v1, w)
				else:
					new_w = w + ret.get_edge_weight(v0, v1)
					ret.set_edge_weight(v0, v1, new_w)

		return ret

