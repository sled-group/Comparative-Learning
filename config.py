
'''
Learning Attributes:
	- Color (6)
	- Material (3)
	- Shape (8)
Additional Attributes:
	- Color (2)
	- Material (1)
	- Shape (3)
Flexibility:
	- Camera angle (6)
	- Lighting (3)
Variability (Only in testing):
	- Size (2) 		[Default: large]
	- Stretch (3) 	[Default: normal]
	- Color shade (2) [Default: base]

Naming convension:
[color]_[material]_[shape]_shade_[]_stretch_[]_scale_[]_brightness_view_[]_[tyimg].png
e.g.
aqua_glass_cone_shade_base_stretch_normal_scale_large_brightness_bright_view_0_-2_3_rgba.png
'''
# Learning attributes:
colors = ['brown', "green", "blue", "aqua", "purple", "red", "yellow", 'white']
materials = ['rubber', 'metal', 'plastic', 'glass']
shapes = ["cube", "cylinder", "sphere", "cone", "torus", "gear",
			"torus_knot", "sponge", "spot", "teapot", "suzanne"]
vocabs = colors+materials+shapes


# Flexibility:
views = ['0_3_2', '-2_-2_2', '-2_2_2',  '1.5_-1.5_3', '1.5_1.5_3', '0_-2_3']
brightness = ['dim', 'normal', 'bright']

# Variability
scale_train = ['large']
stretch_train = ['normal']
shade_train = ['base']

scale_test = ['small', 'medium', 'large']
stretch_test = ['normal', 'x', 'y', 'z']
shade_test = ['base', 'light', 'dark']

# Types of images
tyimgs = ['rgba', 'depth', 'normal', 'object_coordinates', 'segmentation']


dic_train = {"color": colors,
			"material": materials,
			"shape": shapes,
			"view": views,
			'brightness': brightness,
			"scale": scale_train,
			'stretch': stretch_train,
			'shade': shade_train
			}
dic_test = {"color": colors,
			"material": materials,
			"shape": shapes,
			"view": views,
			'brightness': brightness,
			"scale": scale_test,
			'stretch': stretch_test,
			'shade': shade_test
			}
types_learning = ['color', 'material', 'shape']
types_flebility = ['color', 'material', 'shape', 'brightness', 'view']
types_variability = ['scale', 'stretch', 'shade']
types_all = ['color', 'material', 'shape', 'brightness',
				'view', 'shade', 'stretch', 'scale']


# paths and filenames
bn_n_train = "bn_n_train.txt"
bsn_novel_train_1 = "bsn_novel_train_1.txt"
bsn_novel_train_2 = "bsn_novel_train_2.txt"
bsn_novel_train_2_nw = "bsn_novel_train_2_nw.txt"
bsn_novel_train_2_old = "bsn_novel_train_2_old.txt"

bn_n_test = "bn_n_test.txt"
bsn_novel_test_1 = "bsn_novel_test_1.txt"
bsn_novel_test_2_nw = "bsn_novel_test_2_nw.txt"
bsn_novel_test_2_old = "bsn_novel_test_2_old.txt"

bn_test = "bn_test.txt"
bsn_test_1 = "bsn_test_1.txt"
bsn_test_2_nw = "bsn_test_2_nw.txt"
bsn_test_2_old = "bsn_test_2_old.txt"


# train parameters
resize = 224
lr = 5e-4
epochs = 50

sim_batch = 128
gen_batch = 128
batch_size = 32

# model architecture
hidden_dim_clip = 128
latent_dim = 16
