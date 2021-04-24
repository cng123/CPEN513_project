import tkinter as tk
import math

CELL_WIDTH=50
CELL_HEIGHT=50
CHASM_WIDTH=75

class Renderer(tk.Frame):
	def __init__(self, parent, *args, **kwargs):
		tk.Frame.__init__(self, parent, *args, **kwargs)

	def load(self, num_cells, nets, sol):
		self.sol = [(sol[0][0], sol[0][1], 0)]

		# only keep iterations with cost updates
		for i, _ in enumerate(sol[:-1]):
			if sol[i] != sol[i+1]:
				self.sol.append((sol[i+1][0], sol[i+1][1], i + 1))

		self.nets = nets
		self.width = math.ceil(math.sqrt(num_cells/2))
		self.height = math.ceil(num_cells/2/self.width)

		self.canvas = tk.Canvas(
			self, 
			height=self.height*CELL_HEIGHT, 
			width=self.width*CELL_WIDTH*2 + CHASM_WIDTH)
		self.canvas.grid(row=0, column=0)

		# render cell grid
		for i in range(self.width + 1):
			self.canvas.create_line(
				i*CELL_WIDTH,
				0,
				i*CELL_WIDTH,
				self.height*CELL_HEIGHT)
			self.canvas.create_line(
				CHASM_WIDTH + self.width*CELL_WIDTH + i*CELL_WIDTH,
				0,
				CHASM_WIDTH + self.width*CELL_WIDTH + i*CELL_WIDTH,
				self.height*CELL_HEIGHT)
		for i in range(self.height + 1):
			self.canvas.create_line(
				0,
				i*CELL_HEIGHT,
				self.width*CELL_WIDTH,
				i*CELL_HEIGHT)
			self.canvas.create_line(
				CHASM_WIDTH + self.width*CELL_WIDTH,
				i*CELL_HEIGHT,
				CHASM_WIDTH + 2*self.width*CELL_WIDTH,
				i*CELL_HEIGHT)

		# render channel between partitions
		self.canvas.create_rectangle(
			self.width*CELL_WIDTH,
			0,
			CHASM_WIDTH + self.width*CELL_WIDTH,
			self.height*CELL_HEIGHT,
			fill="grey")

		# render frame for cost and buttons
		self.stats = tk.Frame(self, height=self.height*CELL_HEIGHT, width=50)
		self.stats.grid(row=0, column=1)
		self.cost = tk.StringVar()
		tk.Label(self.stats, textvariable=self.cost, width=20).grid(row=0)
		
		self.start = tk.Button(
			self.stats, 
			text="Start", 
			width=20, 
			command=self.start_callback,
			state=tk.DISABLED if len(self.sol) == 1 else tk.NORMAL)

		self.restart = tk.Button(
			self.stats, 
			text="Restart", 
			width=20, 
			command=self.restart_callback,
			state=tk.DISABLED)

		self.start.grid(row=1)
		self.restart.grid(row=2)
		
		# keep track of label id and coord for rendering
		self.left_text   = []
		self.left_coord  = []
		self.right_text  = []
		self.right_coord = []

		# render text to label cells
		for i in range(self.width):
			for j in range(self.height):
				x = ((self.width - i - 1)*CELL_WIDTH 
					+ (self.width - i)*CELL_WIDTH)/2

				y = (j*CELL_HEIGHT + (j + 1)*CELL_HEIGHT)/2
				
				self.left_text.append(self.canvas.create_text(x, y))
				self.left_coord.append((x,y))
				x = (i*CELL_WIDTH + (i + 1)*CELL_WIDTH)/2
				self.right_text.append(
					self.canvas.create_text(
						CHASM_WIDTH + self.width*CELL_WIDTH + x, 
						y))
				self.right_coord.append(
					(CHASM_WIDTH + self.width*CELL_WIDTH + x,y))

		self.net_lines = []
		self.curr_frame = 0
		self.render_frame(0)

	# callback for when run button is pressed
	def start_callback(self):
		# render next frame
		self.curr_frame += 1
		self.render_frame(self.curr_frame)
		self.start.configure(state=tk.DISABLED)

		# if last frame is rendered, stop and enable restart button
		if self.curr_frame == (len(self.sol) - 1):
			self.restart.configure(state=tk.NORMAL)
		# else, keep rendering next frame after some delay
		else:
			self.after(1, self.start_callback)

	# callback for when restart button is pressed
	def restart_callback(self):
		# render first frame again
		self.curr_frame = 0
		self.render_frame(self.curr_frame)
		self.start.configure(state=tk.NORMAL)
		self.restart.configure(state=tk.DISABLED)
	
	def render_frame(self, id):
		left = []
		right = []
		cell_to_loc = {}

		sol_string = self.sol[id][0]
		sol_cost   = self.sol[id][1]
		sol_it     = self.sol[id][2]

		# update text label
		self.cost.set("Cost:\n{}\n(iteration:{})\n".format(sol_cost, sol_it))

		# group sol based on partition
		for i,s in enumerate(sol_string):
			if s == "L":
				cell_to_loc[i] = len(left)
				left.append(i)
			else:
				cell_to_loc[i] = len(right)
				right.append(i)

		# render nodes on left partition by changing text
		for i, id in enumerate(self.left_text):
			self.canvas.itemconfigure(id, text=left[i] if i < len(left) else "")
		
		# render nodes on right partition by changing text
		for i, id in enumerate(self.right_text):
			self.canvas.itemconfigure(
				id, 
				text=right[i] if i < len(right) else "")

		# rerender nets
		for l in self.net_lines:
			self.canvas.delete(l)
		self.net_lines.clear()

		for n in self.nets:
			left  = []
			right = []

			# for each node in a net, group into left and right partitions
			for nn in n:
				if sol_string[nn] == "L":
					left.append(nn)
				else:
					right.append(nn)
			
			# connect all left nodes (just one after another with single wire)
			for l in zip(left[:-1], left[1:]):
				start_coord = self.left_coord[cell_to_loc[l[0]]]
				end_coord   = self.left_coord[cell_to_loc[l[1]]]
				self.net_lines.append(
					self.canvas.create_line(
						start_coord[0], 
						start_coord[1], 
						end_coord[0],
						end_coord[1]))
			# connect all right nodes (just one after another with single wire)
			for r in zip(right[:-1], right[1:]):
				start_coord = self.right_coord[cell_to_loc[r[0]]]
				end_coord   = self.right_coord[cell_to_loc[r[1]]]
				self.net_lines.append(
					self.canvas.create_line(
						start_coord[0], 
						start_coord[1], 
						end_coord[0],
						end_coord[1]))

			# if this net incurs cost, connect arbitrary left and right node 
			# with single red wire
			if left and right:
				start_coord = self.left_coord[cell_to_loc[left[0]]]
				end_coord   = self.right_coord[cell_to_loc[right[0]]]
				self.net_lines.append(
					self.canvas.create_line(
						start_coord[0], 
						start_coord[1], 
						end_coord[0],
						end_coord[1],
						fill="red",
						width=3))
