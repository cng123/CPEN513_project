import tkinter as tk
import math

BOX_WIDTH=50
BOX_HEIGHT=50
CHASM_WIDTH=75

class HeatmapRenderer(tk.Frame):
	def __init__(self, parent, *args, **kwargs):
		tk.Frame.__init__(self, parent, *args, **kwargs)

	def load(self, sol):
		self.sol = sol

		self.canvas = tk.Canvas(
			self, 
			height=5*BOX_HEIGHT, 
			width=10*BOX_WIDTH)
		self.canvas.grid(row=0, column=0)

		self.box_ids  = []
		self.text_ids = []

		# render cell grids
		for i in range(10):
			for j in range(5):
				self.box_ids.append(
					self.canvas.create_rectangle(
						i*BOX_WIDTH,
						j*BOX_HEIGHT,
						(i+1)*BOX_WIDTH,
						(j+1)*BOX_HEIGHT))
				self.text_ids.append(
					self.canvas.create_text(
						(i + 1/2)*BOX_WIDTH,
						(j + 1/2)*BOX_HEIGHT))

		self.stats = tk.Frame(self, height=5*BOX_HEIGHT, width=50)
		self.stats.grid(row=0, column=1)
		
		self.it           = tk.StringVar()
		self.it_best_cost = tk.StringVar()
		self.worse_init   = max(sol[0])
		self.best_final   = min(sol[-1])

		tk.Label(self.stats, textvariable=self.it, width=20).grid(row=0)
		tk.Label(
			self.stats, textvariable=self.it_best_cost, width=20).grid(row=1)
		tk.Label(
			self.stats, 
			text="Best cost:\n{}\n".format(self.best_final), 
			width=20).grid(row=2)
		
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

		self.start.grid(row=3)
		self.restart.grid(row=4)
		
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
			self.after(10, self.start_callback)

	# callback for when restart button is pressed
	def restart_callback(self):
		# render first frame again
		self.curr_frame = 0
		self.render_frame(self.curr_frame)
		self.start.configure(state=tk.NORMAL)
		self.restart.configure(state=tk.DISABLED)
	
	def render_frame(self, id):
		self.it.set("Iteration: {}\n".format(id))
		it_best = min(self.sol[id])
		self.it_best_cost.set("Best Cost\nthis Iteration: {}\n".format(it_best))

		# update heatmap
		for i in zip(self.box_ids, self.text_ids, self.sol[id]):
			# scale color by x^(1/4) 
			# (or else color gets too intense too quickly)
			perc = 1/(pow(self.worse_init - self.best_final, 0.25))
			perc_shifted = perc*pow((i[2] - self.best_final), 0.25)
			self.canvas.itemconfigure(
				i[0], 
				fill="#FF{0:02x}{0:02x}".format(
					int(255*perc_shifted), int(255*perc_shifted)))
			self.canvas.itemconfigure(i[1], text=str(i[2]))
