import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.rc('font',family='Helvetica')
plt.rcParams.update({'font.size': 14})

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

dataset = 'stochastic_dataset'
# dataset = 'cycleworld'
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
steps = 1999
bptt = False
value = 1.0
name = ''
comp = False
decay = False
if decay == True:
	value = 1.0

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
	steps = 19999
	bptt_pathname = 'results-iclr-lr/results-cw/'
	pathname = 'results-iclr-lambda/results-cw/'
	lambda_series = [0.0,0.1,0.5,1.0,5.0,10.0]
	learning_rate = 0.01
	fpt_lr = 0.01
	output_lr = 0.01
	data_name = '{}_cw_'.format(cycleworld_size)
	plt.title('{}-CycleWorld'.format(cycleworld_size))
	# plt.ylim(50,110)
	plt.xlabel('Lambda')
	x = list(range(100, 100000, 100))
	plt.ylabel('Incorrect predictions in last {} steps'.format(100))
	if exp == True:
		end_name_fpt = 'exp_{}.npy'.format(alpha)
	else:
		end_name_fpt = '.npy'
	end_name_bptt = '.npy'

	configs = []
	if cycleworld_size == 10:
		configs = [(1000,1,0),(1000,3,0),(1000,5,0),(1000,10,0)]
		time_steps_series = [1,3,5,10]
	else:
		configs = [(1000,1,0),(1000,2,0),(1000,6,0)]
	# configs = [(1000,1,1),(1000,2,2),(1000,3,3),(1000,5,5),(1000,10,10)]
	# configs = [(1000,1,10),(1000,2,20),(1000,3,30),(1000,5,50),(1000,10,100)]
	# configs = [(100,50,0)]#,(1000,1000,0)]
	# configs = [(1000,1,0),(1000,3,0),(1000,5,0),(1000,7,0),(1000,10,0)]
	# configs = [(1000,1,1),(1000,3,3),(1000,5,5),(1000,7,7),(1000,10,10)]

		time_steps_series = [1,2,6]
	

elif dataset == 'stochastic_dataset':
	bptt_pathname = 'results-iclr-lr/results-sd/'
	pathname = 'results-iclr-lambda/results-sd/'
	lambda_series = [0.0,0.1,0.5,1.0,5.0,10.0,15.0]
	data_name = 'sd_'
	plt.title('Stochastic Dataset')
	plt.xlabel('Lambda')
	plt.ylabel('Cross-Entropy Loss')
	x = list(range(100, 10000, 100))
	# plt.xticks([0.0001,0.0003,0.001,0.003,0.001,0.003])
	# plt.ylim(0.4,0.7) 	
	if exp == True:
		end_name_fpt = 'exp_{}_loss.npy'.format(alpha)
	else:
		end_name_fpt = '_loss.npy'
	end_name_bptt = '_loss.npy'

	time_steps_series = [1,5,10,15]
	configs = []
	l = 1000
	configs = [(l,1,0),(l,5,0),(l,10,0),(l,15,0)]

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



# configs = [(100,1,0),(100,10,0),(100,50,0),(1000,1,0),(1000,10,0),(1000,50,0),(10000,1,0),(10000,10,0),(10000,50,0)]
# configs = [(100,1,0),(100,10,0),(100,50,0),(1000,1,0),(1000,10,0),(1000,50,0),(10000,1,0),(10000,10,0),(10000,50,0)]
# print('Senstivity:')
y = np.zeros(len(lambda_series))
x = ['0','0.1','0.5','1','5','10','15']

# if hybrid == True:
# 	plt.title('Parameter Senstivity for {}'.format(dataset))
# else:
# 	plt.title('Parameter Senstivity for {}'.format(dataset))
colors = []
for time_steps in time_steps_series:
	color = next(ax._get_lines.prop_cycler)['color']
	colors.append(color)
	# filename_bptt = bptt_pathname+'bptt/'+data_name+'lr_{}_bptt_T_{}'.format(learning_rate, time_steps)+end_name_bptt
	# if dataset == 'cycleworld': #or 'stochastic_dataset':
	# 	bptt = 100-np.load(filename_bptt)*100
	# else:
	# 	bptt = np.load(filename_bptt)
	# bptt_mean = np.mean(np.mean(bptt,axis=0))
	# # bptt_mean = gaussian_filter1d(bptt_mean,sigma=1.0)
	# y[i] = bptt_mean
		# print(lr[i],y[i])
	# input()
	# plt.plot(x,y,color=color,linestyle=':')
# plt.show()
# exit()
count = 0
for config in configs:
	buffer_length,updates_per_step,state_updates_per_step = config
	for i in range(len(lambda_series)):
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
			filename_fpt = pathname+'hybrid/'+folder+data_name+'lr_{}{}_lamb_{}_fpt_n_{}_N_{}_s_{}_a{}{}'.format(fpt_lr,output_lr,lambda_series[i],buffer_length,
				updates_per_step,state_updates_per_step,499,value)+end_name
		else:
			if exp == True:
				folder = 'per/'
				end_name = 'exp_{}.npy'.format(alpha)
			else:
				folder = ''
				end_name = '.npy'
			filename_fpt = pathname+'normal/'+folder+data_name+'lr_{}{}_fpt_n_{}_N_{}_s_{}'.format(fpt_lr,output_lr,buffer_length,updates_per_step,state_updates_per_step)+end_name_fpt
		
		fpt = np.load(filename_fpt)
		# fpt[:,0] = 0
		fpt_mean = np.mean(np.mean(fpt))
		

		if dataset == 'cycleworld':
			y[i] = 100-fpt_mean
		else:
			y[i] = fpt_mean

		# print(name+'FPP-Buffer-{}-n-{}-k-{}: {}'.format(buffer_length,updates_per_step,state_updates_per_step,fpt_mean))
		# x = ['1','2','4','8','16']
		# print(y)
	plt.plot(x,y,color=colors[count])
	count += 1
# plt.legend()
from matplotlib.lines import Line2D
ax = plt.gca()
custom_lines = [Line2D([0], [0], color='black', lw=1, linestyle='--'),
                Line2D([0], [0], color='black', lw=1)]

custom_lines = [Line2D([0], [0], color='black', lw=1, linestyle='--'),
                Line2D([0], [0], color='black', lw=1),
				Line2D([0], [0], color='blue', lw=1,),
                Line2D([0], [0], color='orange', lw=1),
                Line2D([0], [0], color='green', lw=1),
                Line2D([0], [0], color='red', lw=1)]

# fig, ax = plt.subplots()
# lines = ax.plot(data)
if dataset == 'stochastic_dataset':
	ax.legend(custom_lines, ['BPTT', 'FPP','1','5','10','15'])
elif dataset == 'cycleworld' and cycleworld_size==6:
	ax.legend(custom_lines, ['BPTT', 'FPP','1','2','6'])
elif dataset == 'cycleworld' and cycleworld_size==10:
	ax.legend(custom_lines, ['BPTT', 'FPP','1','3','5','10'])
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


