from tensorboardX import SummaryWriter
import numpy as np

class SingleNumberVizer(object):
	
	def __init__(self, tb_writer, tag, interval = 1):
		super(SingleNumberVizer, self).__init__()
		self.tag      = tag
		self.writer   = tb_writer
		self.interval = interval

	def print_result(self, pos , number):

		if self.interval <= 0 : return

		if pos % self.interval == 0:
			self.writer.add_scalar(self.tag, number, pos)

import matplotlib.cm
from skimage import io, img_as_float
from skimage.transform import resize, rescale
import os
import scipy.misc

def get_attimg(img, attmap, cm = matplotlib.cm.ScalarMappable(cmap="jet")):

	h = img.shape[1]
	w = img.shape[2]
	s = attmap.size(-1)
	attmap = attmap.squeeze().view(s, s).numpy()
	# attmap = cm.to_rgba(attmap)[:, :, 0:3]
	attmap = resize(attmap, (h, w), mode='reflect')
	attmap = cm.to_rgba(attmap)[:, :, 0:3]

	img = img.permute(1, 2, 0).numpy()

	return img + attmap

class AttImgVizer(object):
	
	def __init__(self, output_dir, sample_num):
		super(AttImgVizer, self).__init__()

		self.output_dir   = output_dir
		self.sample_num   = sample_num
		self.colormap     = matplotlib.cm.ScalarMappable(cmap="jet")

		if not os.path.isdir(self.output_dir): os.mkdir(self.output_dir)

	def print_result(self, images, attmaps, msgs = 'Sample'):
		batch_size = images.size(0)
		if not isinstance(msgs, list) : msgs = [msgs] * batch_size

		for i_batch in range(min(self.sample_num, batch_size)):

			sample_dir = os.path.join(self.output_dir, 'Sample_' + str(i_batch))
			if not os.path.isdir(sample_dir): os.mkdir(sample_dir)

			scipy.misc.imsave(os.path.join(sample_dir, 'raw_image.jpg'), images[i_batch].permute(1, 2, 0).numpy())

			for i_attlist, caps_attmaps in enumerate(attmaps): # (B,8,14,14)

				for i_attmap, attmap in enumerate(caps_attmaps[i_batch]): # (14,14)

					attimg = get_attimg(images[i_batch], attmap, self.colormap)
					scipy.misc.imsave(os.path.join(sample_dir, 'L' \
					 + str(i_attlist) + '_N' + str(i_attmap) + '.jpg'), attimg)

			print(msgs[i_batch] + '?')

#---------------------------------------------- display tree ------------------------------------------
import torch

class Node(object):
	"""docstring for Node"""
	def __init__(self, childs, weights, height, info):
		super(Node, self).__init__()
		
		self.childs  = childs  # <list>[<node>]
		self.weights = weights # <list>[<float>]
		self.height  = height  # <int> start from 0 (leafnode)
		self.info    = info    # <dict> {'n':<int> 'h':<int> 'w':<int>}

class RouteTree(object):
	"""docstring for Tree"""
	def __init__(self, routing_weights):
		super(RouteTree, self).__init__()
		"""
		Args:
			`routing_weights` : <list(TreeHeight)>[ <Tensor>(Nin,Hin,Win,Nout,Hout,Wout) ]
		"""
		self.rw = routing_weights 
		self.height = len(routing_weights) + 1 # <int> tree height
		self.roots = [] # <list>[<node>]

		active_nodes = [{} for _ in range(self.height)] # <dict>{'<tuple>(height, n, h, w)' : node}

		def build_self_to_root(height, n, h, w):
			
			# if built, return
			if (n, h, w) in active_nodes[height]: return 

			# build node and regist
			self_node = Node(childs=[], weights=[], height=height, info={'n': n, 'h': h, 'w': w}) 
			active_nodes[height][(n, h, w)] = self_node

			# if no father, return
			if height == self.height - 1:  
				self.roots.append(self_node)
				return

			# get father pos and build father to root
			rw = self.rw[height][n, h, w] # <Tensor>(Nout,Hout,Wout)
			Nf, Hf, Wf = rw.size()
			_, pos = rw.view(-1).max(dim=0)
			nf, hf, wf = (pos[0] // (Hf * Wf)), ((pos[0] // Wf) % Hf), (pos[0] % Wf)
			build_self_to_root(height + 1, nf, hf, wf)

			# add son to father
			active_nodes[height + 1][(nf, hf, wf)].childs.append(self_node)
			active_nodes[height + 1][(nf, hf, wf)].weights.append(routing_weights[height][n, h, w, nf, hf, wf])

		# build from each leaf node
		N0, H0, W0, _, _, _ = self.rw[0].size()
		for n in range(N0):
			for h in range(H0):
				for w in range(W0):
					build_self_to_root(0, N0 - n - 1, h, w)

	def _layout(self):
		# get a layout policy: <dict>{'edges': , 'nodes': <list>[ <list>(<int>, <int>) ] }

		layer_size = [(w.size(0), w.size(1), w.size(2)) for w in self.rw]
		layer_size.append((self.rw[-1].size(3), self.rw[-1].size(4), self.rw[-1].size(5)))
		node_nums = [w.size(0) * w.size(1) * w.size(2) for w in self.rw]
		node_nums.append(self.rw[-1][0,0,0].numel()) # <list>[<int>]

		# get nodes
		def getx(n):
			if n == 1: return torch.FloatTensor([0])
			return torch.linspace(-n, n, steps=n).type(torch.FloatTensor)

		xs = torch.cat((getx(n) for n in node_nums))
		ytmp = torch.linspace(-1, 1, steps=self.height).type(torch.FloatTensor)
		ys = torch.cat((torch.FloatTensor([ytmp[h]] * node_nums[h]) for h in range(self.height)))

		nodes = torch.stack([xs,ys], 1)
		# end

		# get edges
		edges = []
		def heightnhw2pos(height, n, h, w):
			N, H, W = layer_size[height]
			return sum(node_nums[0:height]) + (n * H * W) + (h * W) + w

		def dfs(rt):
			rt_pos = heightnhw2pos(rt.height, rt.info['n'], rt.info['h'], rt.info['w'])

			for i in range(len(rt.childs)):
				c = rt.childs[i]
				c_pos = heightnhw2pos(c.height, c.info['n'], c.info['h'], c.info['w'])
				edges.append( (rt_pos, c_pos, rt.weights[i]) )
				dfs(c)

		dfs(self.roots[0])
		# end

		# get v_labels
		v_labels = []
		for height in range(self.height):
			N, H, W = layer_size[height]
			for n in range(N):
				for h in range(H):
					for w in range(W):
						v_labels.append('{},{},{}'.format(n, h, w))
		# end

		return {'edges': edges, 'nodes': nodes, 'v_labels': v_labels}

	def show(self, out_dir, msg): #debug
		import plotly
		import plotly.graph_objs as go
		import matplotlib.cm as cm

		lay = self._layout()

		nr_vertices = len(lay['nodes'])
		v_label = lay['v_labels']
		# v_label = list(map(str, range(nr_vertices)))

		position = {k: lay['nodes'][k] for k in range(nr_vertices)}
		
		Y = [lay['nodes'][k][1] for k in range(nr_vertices)]
		M = max(Y)

		E = lay['edges'] # list of edges
		colormap = cm.ScalarMappable(cmap="jet")

		L = len(position)
		Xn = [position[k][0] for k in range(L)]
		Yn = [position[k][1] for k in range(L)]
		lines = []
		for edge in E:
		    r, g, b = colormap.to_rgba(edge[2], norm=False)[0:3]
		    lines.append(go.Scatter(x=[position[edge[0]][0],position[edge[1]][0], None],
		                   		y=[position[edge[0]][1],position[edge[1]][1], None],
		                   		mode='lines',
		                   		line=dict(color='rgb({},{},{})'.format(r*255,g*255,b*255), width=1),
		                   		hoverinfo='none'
		                   		))
		labels = v_label

		dots = go.Scatter(x=Xn,
		                  y=Yn,
		                  mode='markers',
		                  name='',
		                  marker=dict(symbol='dot',
		                                size=25, 
		                                color='#6175c1',    #'#DB4551', '#6175c1' 
		                                line=dict(color='rgb(50,50,50)', width=1)
		                                ),
		                  text=labels,
		                  hoverinfo='text',
		                  opacity=0.8
		                  )

		def make_annotations(pos, text, font_size=13, font_color='rgb(250,250,250)'):
		    L=len(pos)
		    if len(text)!=L:
		        raise ValueError('The lists pos and text must have the same len')
		    annotations = go.Annotations()
		    for k in range(L):
		        annotations.append(
		            go.Annotation(
		                text=labels[k], # or replace labels with a different list for the text within the circle  
		                x=pos[k][0], y=position[k][1],
		                xref='x1', yref='y1',
		                font=dict(color=font_color, size=font_size),
		                showarrow=False)
		        )
		    return annotations

		axis = dict(showline=False, # hide axis line, grid, ticklabels and  title
		            zeroline=False,
		            showgrid=False,
		            showticklabels=False,
		            )

		layout = dict(title= 'Parsed-Tree like Capsule',  
		              annotations=make_annotations(position, v_label),
		              font=dict(size=12),
		              showlegend=False,
		              xaxis=go.XAxis(axis),
		              yaxis=go.YAxis(axis),          
		              margin=dict(l=0, r=0, b=10, t=0),
		              hovermode='closest',
		              plot_bgcolor='rgb(0,0,0)'          
		              )

		data=go.Data(lines+[dots])
		fig=dict(data=data, layout=layout)
		fig['layout'].update(annotations=make_annotations(position, v_label))
		plotly.offline.plot(fig, filename=os.path.join(out_dir, msg + '.html'))

class CapsTreeVizer(object):
	
	def __init__(self, output_dir, sample_num):
		super(CapsTreeVizer, self).__init__()

		self.output_dir   = output_dir
		self.sample_num   = sample_num

		if not os.path.isdir(self.output_dir): os.mkdir(self.output_dir)

	def print_result(self, routing_weights, msgs = ''):
		batch_size = routing_weights[0].size(0)

		for i_batch in range(min(self.sample_num, batch_size)):

			sample_dir = os.path.join(self.output_dir, 'Sample_' + str(i_batch))
			if not os.path.isdir(sample_dir): os.mkdir(sample_dir)

			rw = [ rw_h[i_batch][:, None, None, :, None, None] for rw_h in routing_weights ]

			tree = RouteTree(rw)
			lay  = tree.show(sample_dir, msgs[i_batch])

# if __name__ == '__main__':

# 	routing_weights = [ #torch.randn(16,1,1,4,1,1), \
# 						# torch.randn(4,1,1,1,1,1), \
# 						torch.randn(4,1,1,5,1,1) ]
	
# 	tree = RouteTree(routing_weights)

# 	lay = tree.show()