import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from scipy.ndimage.filters import gaussian_filter1d

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2]) 

dataset = 'mnist'
cycleworld_size = 6
learning_rate = 0.001
fpt_lr = 0.001
output_lr = 0.001
buffer_length_series = [100,1000]
updates_per_step_series = [1,5,10]
state_updates_per_step_series = [0,5,10] 
time_steps_series = [1,3,5]
exp = False
alpha = 0.5
hybrid = True
steps = 499
buffer = False
value = 1.0
name = ''
comp = False

# configs = [(1000,1,0)]#,(1000,5,0),(1000,3,0),(1000,1,0)]
configs = [(1000,1,0),(1000,5,0),(1000,10,0)]
# configs = [(1000,1,1),(1000,3,3),(1000,5,5)]
# configs = [(1000,15,0),(1000,20,0),(1000,25,0),(1000,30,0),(1000,35,0)]
# configs = [(1000,1,1),(1000,10,10),(1000,15,15)]
# configs = [(1000,15,0),(1000,20,0),(1000,25,0),(1000,30,0)]#,(1000,15,0)]
# configs = [(1000,1,0),(1000,5,0),(1000,5,5),(1000,10,0)]
# configs = [(1000,1,1),(1000,5,5),(1000,10,10)]
# configs = [(1000,1,0),(1000,10,0),(1000,10,10),(1000,15,15),(1000,20,20)]

def plot_error_bars(x,data,num_runs,color):
	mean = np.mean(data,axis = 0)
	std = np.std(data,axis=0)
	# plt.fill_between(x,(mean+std),(mean-std),alpha=0.15,color=color)


plt.figure(figsize=(10, 6), dpi=80)
ax = plt.gca()
if dataset == 'cycleworld':
	bptt_pathname = 'results-final/results-cw/'
	pathname = 'results-final/results-cw/'
	learning_rate = 0.01
	fpt_lr = 0.01
	output_lr = 0.01
	data_name = '{}_cw_'.format(cycleworld_size)
	# plt.title('{}-CycleWorld'.format(cycleworld_size))
	plt.ylim(70,110)
	plt.xlabel('Steps')
	x = list(range(100, 100000, 100))
	# plt.ylabel('Good predictions in last {} steps'.format(100))
	if exp == True:
		end_name_fpt = 'exp_{}.npy'.format(alpha)
	else:
		end_name_fpt = '.npy'
	end_name_bptt = '.npy'
	baseline = 90*np.ones_like(x)
	time_steps_series = []
	configs = []
	
	
	# configs = [(1000,1,1),(1000,2,2),(1000,3,3),(1000,5,5),(1000,10,10)]
	# configs = [(1000,1,10),(1000,2,20),(1000,3,30),(1000,5,50),(1000,10,100)]
	# configs = [(100,50,0)]#,(1000,1000,0)]
	# configs = [(1000,1,0),(1000,3,0),(1000,5,0),(1000,7,0),(1000,10,0)]
	# configs = [(1000,1,1),(1000,3,3),(1000,5,5),(1000,7,7),(1000,10,10)]
	if cycleworld_size == 6:
		time_steps_series = [1,2,6]
		configs = [(1000,1,0),(1000,2,0),(1000,6,0)]
	else:
		time_steps_series = [1,3,5,10]
		configs = [(1000,1,0),(1000,3,0),(1000,5,0),(1000,10,0)]
	# run = 3
	# configs = [(1000,1,0),(1000,3,0),(1000,5,0),(1000,10,0)]
	# configs = [(1000,1,1),(1000,3,3),(1000,5,5),(1000,10,10)]
	# configs = [(1000,1,2),(1000,3,6),(1000,5,10),(1000,10,20)]
	# configs = [(100,1,0),(100,1,0),(10000,10,0),(10000,10,0)]
	plt.plot(x,baseline,color='black',label='baseline')

elif dataset == 'stochastic_dataset':
	bptt_pathname = 'results-final/results-sd/'
	pathname = 'results-final/results-sd/'
	data_name = 'sd_'
	# plt.title('Stochastic Dataset')
	plt.xlabel('Steps')
	# plt.ylabel('Cross-Entropy Loss')
	x = list(range(100, 10000, 100))
	plt.ylim(0.4,0.7) 	
	if exp == True:
		end_name_fpt = 'exp_{}_loss.npy'.format(alpha)
	else:
		end_name_fpt = '_loss.npy'
	end_name_bptt = '_loss.npy'
	time_steps_series = [1,5,10,15]

	# time_steps_series = []	
	configs = []
	configs = [(1000,1,0),(1000,5,0),(1000,10,0),(1000,15,0)]
	# configs = [(1000,1,1),(1000,3,3),(1000,5,5),(1000,10,10)]
	# configs = [(1000,15,0),(1000,20,0),(1000,30,0),(1000,50,0),(1000,100,0)]
	# configs = [(10000,100,0)]#,(10000,1000,0)]#,(1000,1000,0)]
	# configs = [(1000,15,0),(1000,20,0)]#,(1000,30,0)]#,(1000,30,0)]
	# configs = [(1000,1,1),(1000,3,3),(1000,5,5),(1000,10,10)]
	# configs = [(1000,1,2),(1000,3,6),(1000,5,10),(1000,10,20)]
	# configs = [(100,1,0),(100,1,0),(10000,10,0),(10000,10,0)]
	# configs = [(100,1,0),(100,10,0),(100,50,0)]

elif dataset == 'ptb':
	learning_rate = 0.0001
	fpt_lr = 0.0001
	output_lr = 0.0001
	bptt_pathname = 'results-final-2/results-ptb/'
	pathname = 'results-final-2/results-ptb/'
	data_name = 'ptb_'
	plt.title('Penn-Tree-Bank')
	plt.xlabel('Steps')
	x = list(range(0, 46400, 100))
	plt.ylim(4,7) 	
	if exp == True:
		end_name_fpt = 'exp_{}_loss.npy'.format(alpha)
	else:
		end_name_fpt = '_loss.npy'
	end_name_bptt = '_loss.npy'
	time_steps_series = [1,5,10,20,35,50]
	configs = [(1000,1,0),(1000,5,0),(1000,10,0),(1000,20,0),(1000,35,0),(1000,50,0)]

elif dataset == 'mnist':
	learning_rate = 0.001
	fpt_lr = 0.001
	output_lr = 0.001
	bptt_pathname = 'results-mnist/'
	pathname = 'results-mnist/'
	data_name = 'mnist_'
	x = list(range(0, 499, 1))
	end_name_fpt = '.npy'
	end_name_bptt = '.npy'
	time_steps_series = [5,10,20,28]
	configs = [(1000,5,0),(1000,10,0),(1000,20,0),(1000,28,0)]

colors = []
for time_steps in time_steps_series:
	filename_bptt = bptt_pathname+'bptt/'+data_name+'lr_{}_bptt_T_{}'.format(learning_rate, time_steps)+end_name_bptt
	if dataset == 'cycleworld':
		bptt = np.load(filename_bptt)*100
	else:
		bptt = np.load(filename_bptt)
	bptt_mean = np.mean(bptt,axis=0)
	color = next(ax._get_lines.prop_cycler)['color']
	colors.append(color)
	bptt_mean = gaussian_filter1d(bptt_mean,sigma=1.0)

	# for i in range(len(bptt_mean)):
	# 	if bptt_mean[i]>100:
	# 		bptt_mean[i] = 100
	# bptt_mean = bptt[run]
	if dataset != 'ptb':
		if buffer == False:
			plt.plot(bptt_mean,label = '{}-BPTT'.format(time_steps),color=color,linestyle='--')
			# plot_error_bars(x,bptt,bptt.shape[0],color)
		print('{}-BPTT: {}'.format(time_steps,np.mean(bptt_mean)))
	else:
		plt.plot(bptt_mean,label = '{}-BPTT'.format(time_steps),color=color)
		print('{}-BPTT: {}'.format(time_steps,bptt_mean[-1]))

count = 0
for config in configs:
	buffer_length,updates_per_step,state_updates_per_step = config
	name = ''
	if hybrid == True:
		if value == 1.0:
			name = name +str(updates_per_step) + '-' 
		else:
			name = name+'HYB-'+str(updates_per_step) + '-'
		if exp == True:
			name = name + 'PER-'
			folder = 'per/'
			end_name = end_name_fpt
		else:
			folder = ''
			end_name = end_name_fpt
		if comp == True:
			pathname = 'results/results-cw-(c)/'
		filename_fpt = pathname+'hybrid/'+folder+data_name+'lr_{}{}_fpt_n_{}_N_{}_s_{}_a{}{}'.format(fpt_lr,output_lr,buffer_length,
			updates_per_step,state_updates_per_step,steps,value)+end_name
		if comp == True:
			pathname = 'results/results-cw-(a)/'
			filename_fpt_1 = pathname+'hybrid/'+folder+data_name+'lr_{}{}_fpt_n_{}_N_{}_s_{}_a{}{}'.format(fpt_lr,output_lr,buffer_length,
				updates_per_step,state_updates_per_step,steps,value)+end_name
	else:
		if exp == True:
			folder = 'per/'
			end_name = 'exp_{}.npy'.format(alpha)
		else:
			folder = ''
			end_name = '.npy'
		if comp == True:
			pathname = 'results/results-cw-(c)/'
		filename_fpt = pathname+'normal/'+folder+data_name+'lr_{}{}_fpt_n_{}_N_{}_s_{}'.format(fpt_lr,output_lr,buffer_length,updates_per_step,state_updates_per_step)+end_name_fpt
		if comp==True:
			pathname = 'results/results-cw-(a)/'
			filename_fpt_1 = pathname+'normal/'+folder+data_name+'lr_{}{}_fpt_n_{}_N_{}_s_{}'.format(fpt_lr,output_lr,buffer_length,updates_per_step,state_updates_per_step)+end_name_fpt
	fpt = np.load(filename_fpt)
	fpt[:,0] = 0
	fpt_mean = np.mean(fpt,axis=0)
	fpt_mean = gaussian_filter1d(fpt_mean,sigma=1.0)

	# fpt_mean = fpt[run]
	if comp == True:
		fpt_1 = np.load(filename_fpt_1)
		fpt_1[:,0] = 0
		fpt_mean_1 = np.mean(fpt_1,axis=0)
	# color = next(ax._get_lines.prop_cycler)['color']
	# colors.append(color)
	color = colors[count]
	color = lighten_color(color,0.75)
	# print('FPP-Buffer-{}-n-{}-k-{}: {}'.format(buffer_length,updates_per_step,state_updates_per_step,np.mean(fpt_mean)))
	if dataset != 'ptb':
		plt.plot(fpt_mean,color=color,label = name+'FPP-Buffer-{}'.format(buffer_length,updates_per_step,state_updates_per_step))
		# plot_error_bars(fpt,fpt.shape[0],color)
		print(name+'FPP-Buffer-{}-n-{}: {},{}'.format(buffer_length,updates_per_step,np.mean(fpt_mean),np.std(fpt_mean)))
		if comp == True:
			plt.plot(x,fpt_mean_1,color=color,linestyle=':')
			plot_error_bars(x,fpt_1,fpt.shape[0],color)
			print(name+'FPP-Buffer-{}-n-{}: {},{}'.format(buffer_length,updates_per_step,np.mean(fpt_mean_1),np.std(fpt_mean)))

	else:		
		plt.plot(fpt_mean,color=color,label = name+'FPP-Buffer-{}-n-{}'.format(buffer_length,updates_per_step))
		print(name+'FPP-Buffer-{}-n-{}: {}'.format(buffer_length,updates_per_step,fpt_mean[-1]))

	count += 1
# steps = 1999
# value =1.0
# count = 0
# for config in configs:
# 	buffer_length,updates_per_step,state_updates_per_step = config
# 	name = ''
# 	if hybrid == True:
# 		if value == 1.0:
# 			name = name +str(updates_per_step) + '-' 
# 		else:
# 			name = name+'HYB-'+str(updates_per_step) + '-'
# 		if exp == True:
# 			name = name + 'PER-'
# 			folder = 'per/'
# 			end_name = end_name_fpt
# 		else:
# 			folder = ''
# 			end_name = end_name_fpt
# 		if comp == True:
# 			pathname = 'results/results-cw-(c)/'
# 		filename_fpt = pathname+'hybrid/'+folder+data_name+'lr_{}{}_fpt_n_{}_N_{}_s_{}_a{}{}'.format(fpt_lr,output_lr,buffer_length,
# 			updates_per_step,state_updates_per_step,steps,value)+end_name
# 		if comp == True:
# 			pathname = 'results/results-cw-(a)/'
# 			filename_fpt_1 = pathname+'hybrid/'+folder+data_name+'lr_{}{}_fpt_n_{}_N_{}_s_{}_a{}{}'.format(fpt_lr,output_lr,buffer_length,
# 				updates_per_step,state_updates_per_step,steps,value)+end_name
# 	else:
# 		if exp == True:
# 			folder = 'per/'
# 			end_name = 'exp_{}.npy'.format(alpha)
# 		else:
# 			folder = ''
# 			end_name = '.npy'
# 		if comp == True:
# 			pathname = 'results/results-cw-(c)/'
# 		filename_fpt = pathname+'normal/'+folder+data_name+'lr_{}{}_fpt_n_{}_N_{}_s_{}'.format(fpt_lr,output_lr,buffer_length,updates_per_step,state_updates_per_step)+end_name_fpt
# 		if comp==True:
# 			pathname = 'results/results-cw-(a)/'
# 			filename_fpt_1 = pathname+'normal/'+folder+data_name+'lr_{}{}_fpt_n_{}_N_{}_s_{}'.format(fpt_lr,output_lr,buffer_length,updates_per_step,state_updates_per_step)+end_name_fpt
# 	fpt = np.load(filename_fpt)
# 	fpt[:,0] = 0
# 	fpt_mean = np.mean(fpt,axis=0)
# 	fpt_mean = gaussian_filter1d(fpt_mean,sigma=1.0)

# 	# fpt_mean = fpt[run]
# 	if comp == True:
# 		fpt_1 = np.load(filename_fpt_1)
# 		fpt_1[:,0] = 0
# 		fpt_mean_1 = np.mean(fpt_1,axis=0)
# 	# color = next(ax._get_lines.prop_cycler)['color']
# 	# colors.append(color)
# 	color = colors[count]
# 	color = lighten_color(color,0.75)
# 	# print('FPP-Buffer-{}-n-{}-k-{}: {}'.format(buffer_length,updates_per_step,state_updates_per_step,np.mean(fpt_mean)))
# 	if dataset != 'ptb':
# 		plt.plot(x,fpt_mean,color=color,linestyle='--',label = name+'FPP-Buffer-{}'.format(buffer_length,updates_per_step,state_updates_per_step))
# 		plot_error_bars(x,fpt,fpt.shape[0],color)
# 		print(name+'FPP-Buffer-{}-n-{}: {},{}'.format(buffer_length,updates_per_step,np.mean(fpt_mean),np.std(fpt_mean)))
# 		if comp == True:
# 			plt.plot(x,fpt_mean_1,color=color,linestyle=':')
# 			plot_error_bars(x,fpt_1,fpt.shape[0],color)
# 			print(name+'FPP-Buffer-{}-n-{}: {},{}'.format(buffer_length,updates_per_step,np.mean(fpt_mean_1),np.std(fpt_mean)))

# 	else:		
# 		plt.plot(fpt_mean,color=color,label = name+'FPP-Buffer-{}-n-{}'.format(buffer_length,updates_per_step))
# 		print(name+'FPP-Buffer-{}-n-{}: {}'.format(buffer_length,updates_per_step,fpt_mean[-1]))

# 	count += 1

if buffer==True:
	##BUFFER BPTT
	count = 0
	for config in configs:
		buffer_length,updates_per_step,state_updates_per_step = config
		name = ''
		name = name+str(updates_per_step) + '-'+'BufferBPTT-'
		if exp == True:
			name = name + 'PER-'
			folder = 'per/'
			end_name = end_name_fpt
		else:
			folder = ''
			end_name = end_name_fpt
			
		filename_fpt = pathname+'buffer_bptt/'+folder+data_name+'lr_{}{}_fpt_n_{}_N_{}_s_{}_a{}{}'.format(fpt_lr,output_lr,buffer_length,
				updates_per_step,state_updates_per_step,steps,1.0)+end_name
			
		fpt = np.load(filename_fpt)
		fpt[:,0] = 0
		fpt_mean = np.mean(fpt,axis=0)
		fpt_mean = gaussian_filter1d(fpt_mean,sigma=1.0)

		# fpt_mean = fpt[run]
		if comp == True:
			fpt_1 = np.load(filename_fpt_1)
			fpt_1[:,0] = 0
			fpt_mean_1 = np.mean(fpt_1,axis=0)
		color = colors[count]
		color = lighten_color(color,0.5)
		# print('FPP-Buffer-{}-n-{}-k-{}: {}'.format(buffer_length,updates_per_step,state_updates_per_step,np.mean(fpt_mean)))
		if dataset != 'ptb':
			plt.plot(x,fpt_mean,color=color,linestyle='dashed',label = name+'Buffer-{}'.format(buffer_length,updates_per_step))
			plot_error_bars(x,fpt,fpt.shape[0],color)
			print(name+'BPTT-Buffer-{}: {},{}'.format(buffer_length,updates_per_step,np.mean(fpt_mean),np.std(fpt_mean)))
			if comp == True:
				plt.plot(x,fpt_mean_1,color=color)
				plot_error_bars(x,fpt_1,fpt.shape[0],color)
				print(name+'BPTT-Buffer-{}: {},{}'.format(buffer_length,updates_per_step,np.mean(fpt_mean_1),np.std(fpt_mean)))

		else:		
			plt.plot(fpt_mean,color=color,linestyle=':',label = name+'Buffer-{}'.format(buffer_length,updates_per_step))
			print(name+'BPTT-Buffer-{}: {}'.format(buffer_length,updates_per_step,fpt_mean[-1]))
		count += 1



# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.77, box.height])
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
if buffer == True:
	name = 'BufferBPTT'
else:
	name = 'BPTT'
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='black', lw=1, linestyle='--'),
                Line2D([0], [0], color='black', lw=1)]

# fig, ax = plt.subplots()
# lines = ax.plot(data)
ax.legend(custom_lines, [name, 'FPP'])
# plt.legend()
plt.show()
# exit()
##Senstivity
plt.figure(figsize=(10, 6), dpi=80)
ax = plt.gca()
if dataset == 'cycleworld':
	bptt_pathname = 'results/results-cw/'
	pathname = 'results/results-cw/'
	learning_rate = 0.01
	fpt_lr = 0.01
	output_lr = 0.01
	data_name = '{}_cw_'.format(cycleworld_size)
	# plt.title('{}-CycleWorld'.format(cycleworld_size))
	# plt.ylim(50,110)
	plt.xlabel('Steps')
	x = list(range(100, 100000, 100))
	# plt.ylabel('Good predictions in last {} steps'.format(100))
	if exp == True:
		end_name_fpt = 'exp_{}.npy'.format(alpha)
	else:
		end_name_fpt = '.npy'
	end_name_bptt = '.npy'
	

elif dataset == 'stochastic_dataset':
	bptt_pathname = 'results/results-sd/'
	pathname = 'results/results-sd/'
	data_name = 'sd_'
	# plt.title('Stochastic Dataset')
	plt.xlabel('Steps')
	# plt.ylabel('Cross-Entropy Loss')
	x = list(range(100, 10000, 100))
	# plt.ylim(0.4,0.7) 	
	if exp == True:
		end_name_fpt = 'exp_{}_loss.npy'.format(alpha)
	else:
		end_name_fpt = '_loss.npy'
	end_name_bptt = '_loss.npy'

elif dataset == 'ptb':
	pathname = 'results/results-ptb/'
	data_name = 'ptb_'
	plt.title('Penn-Tree-Bank')
	plt.xlabel('Steps')
	# x = list(range(0, 10000, 100))
	# plt.ylim(0.4,0.7) 	
	if exp == True:
		end_name_fpt = 'exp_{}_loss.npy'.format(alpha)
	else:
		end_name_fpt = '_loss.npy'
	end_name_bptt = '_loss.npy'

configs = [(100,1,0),(100,10,0),(100,50,0),(1000,1,0),(1000,10,0),(1000,50,0),(10000,1,0),(10000,10,0),(10000,50,0)]
configs = [(100,1,0),(100,10,0),(100,50,0),(1000,1,0),(1000,10,0),(1000,50,0),(10000,1,0),(10000,10,0),(10000,50,0)]
print('Senstivity:')
y = {}
# if hybrid == True:
# 	plt.title('Parameter Senstivity for {}'.format(dataset))
# else:
# 	plt.title('Parameter Senstivity for {}'.format(dataset))
for config in configs:
	buffer_length,updates_per_step,state_updates_per_step = config
	name = ''
	if hybrid == True:
		name = name+'HYB-'
		if exp == True:
			name = name + 'PER-'
			folder = 'per/'
			end_name = end_name_fpt
		else:
			folder = ''
			end_name = end_name_fpt
		filename_fpt = pathname+'hybrid/'+folder+data_name+'lr_{}{}_fpt_n_{}_N_{}_s_{}_a{}{}'.format(fpt_lr,output_lr,buffer_length,
			updates_per_step,state_updates_per_step,steps,value)+end_name
	else:
		if exp == True:
			folder = 'per/'
			end_name = 'exp_{}.npy'.format(alpha)
		else:
			folder = ''
			end_name = '.npy'
		filename_fpt = pathname+'normal/'+folder+data_name+'lr_{}{}_fpt_n_{}_N_{}_s_{}'.format(fpt_lr,output_lr,buffer_length,updates_per_step,state_updates_per_step)+end_name_fpt
	fpt = np.load(filename_fpt)
	fpt[:,0] = 0
	fpt_mean = np.mean(np.mean(fpt))
	print(name+'FPP-Buffer-{}-n-{}-k-{}: {}'.format(buffer_length,updates_per_step,state_updates_per_step,fpt_mean))
	x = ['T=1','T=10','T=50']
	y[config[1]] = fpt_mean
	if config[1] == 50:
		plt.plot(x,y.values(), label= 'Buffer Length= {}'.format(buffer_length))
		y = {}
plt.legend()
plt.show()
	# fpt_1 = np.load(filename_fpt_1)
	# fpt_mean_1 = np.mean(fpt_1,axis=0)
	# color = next(ax._get_lines.prop_cycler)['color']
	# print('FPP-Buffer-{}-n-{}-k-{}: {}'.format(buffer_length,updates_per_step,state_updates_per_step,np.mean(fpt_mean)))
	# plt.plot(x,fpt_mean,color=color,label = name+'FPP-N-{}-n-{}-k-{}'.format(buffer_length,updates_per_step,state_updates_per_step))
	# # plt.plot(x,fpt_mean_1,color=color,linestyle=':',label = 'FPP-Buffer-{}-n-{}-k-{}'.format(buffer_length,updates_per_step,state_updates_per_step))
	# plot_error_bars(x,fpt,fpt.shape[0])

# hybrid = False
# for config in configs:
# 	buffer_length,updates_per_step,state_updates_per_step = config
# 	if hybrid == True:
# 		filename_fpt = 'results/'+data_name+'lr_{}{}_fpt_n_{}_N_{}_s_{}_a{}{}'.format(fpt_lr,output_lr,buffer_length,
# 			updates_per_step,state_updates_per_step,steps,value)+end_name
# 	else:
# 		filename_fpt = 'results/'+data_name+'lr_{}{}_fpt_n_{}_N_{}_s_{}'.format(fpt_lr,output_lr,buffer_length,updates_per_step,state_updates_per_step)+end_name
# 	fpt = np.load(filename_fpt)
# 	fpt_one = fpt[0]
# 	fpt_mean = np.mean(fpt,axis=0)
# 	print('FPP-Buffer-{}-n-{}-k-{}: {}'.format(buffer_length,updates_per_step,state_updates_per_step,np.mean(fpt_mean)))
# 	plt.plot(x,fpt_mean,label = 'FPP-Buffer-{}-n-{}-k-{}'.format(buffer_length,updates_per_step,state_updates_per_step))
# 	plot_error_bars(x,fpt,fpt.shape[0])


