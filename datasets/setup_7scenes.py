import os

# name of the folder where we download the original 7scenes dataset to
# we restructure the dataset by creating symbolic links to that folder
src_folder = '7scenes_source'
focallength = 525.0

def mkdir(directory):
	"""Checks whether the directory exists and creates it if necessacy."""
	if not os.path.exists(directory):
		os.makedirs(directory)

print("=== Downloading ESAC Initialization Files ===============================")

# our initilization file package include the base folder structure for the 7 scenes
os.system('wget -O 7scenes_init.tar.gz https://heidata.uni-heidelberg.de/api/access/datafile/:persistentId?persistentId=doi:10.11588/data/GSJE9D/RG00C4')

# unpack and delete tar file
os.system('tar -xvzf 7scenes_init.tar.gz')
os.system('rm 7scenes_init.tar.gz')

# download the original 7 scenes dataset for poses and images
mkdir(src_folder)
os.chdir(src_folder)

for ds in ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']:
	
	print("=== Downloading 7scenes Data:", ds, "===============================")

	os.system('wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/' + ds + '.zip')
	os.system('unzip ' + ds + '.zip')
	os.system('rm ' + ds + '.zip')
	
	sequences = os.listdir(ds)

	for file in sequences:
		if file.endswith('.zip'):

			print("Unpacking", file)
			os.system('unzip ' + ds + '/' + file + ' -d ' + ds)
			os.system('rm ' + ds + '/' + file)

	print("Linking files...")

	target_folder = '../7scenes_' + ds + '/'

	def link_frames(split_file, variant):

		# create subfolders
		mkdir(target_folder + variant + '/rgb/')
		mkdir(target_folder + variant + '/poses/')
		mkdir(target_folder + variant + '/calibration/')

		# read the split file
		with open(ds + '/' + split_file, 'r') as f:
			split = f.readlines()	
		# map sequences to folder names
		split = ['seq-' + s.strip()[8:].zfill(2) for s in split]

		for seq in split:
			files = os.listdir(ds + '/' + seq)

			# link images
			images = [f for f in files if f.endswith('color.png')]
			for img in images:
				os.system('ln -s ../../../'+src_folder+'/'+ds+'/'+seq+'/'+img+ ' ' +target_folder+variant+'/rgb/'+seq+'-'+img)

			# link folders
			poses = [f for f in files if f.endswith('pose.txt')]
			for pose in poses:
				os.system('ln -s ../../../'+src_folder+'/'+ds+'/'+seq+'/'+pose+ ' ' +target_folder+variant+'/poses/'+seq+'-'+pose)
			
			# create calibration files
			for i in range(len(images)):
				with open(target_folder+variant+'/calibration/%s-frame-%s.calibration.txt' % (seq, str(i).zfill(6)), 'w') as f:
					f.write(str(focallength))
	
	link_frames('TrainSplit.txt', 'training')
	link_frames('TestSplit.txt', 'test')
