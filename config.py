#%%
from itertools import product

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

#
others = views + brightness + scale_test + stretch_test + shade_test

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

### make dicts for logical traing and testing <--- new 
relations = ['and', 'or', 'not'] 
types_logical = [] 
for i in types_learning:
	for j in relations:
		if j == 'not':
			types_logical.append(j+' '+i)
		else:
			for h in types_learning:
				if h+' '+j+' '+i not in types_logical:
					if j == 'and' and i == h:
						pass
					else:
						types_logical.append(i+' '+j+' '+h)
types_logical_with_learning =  types_logical + types_learning 

dic_train_logical = dic_train.copy()
for rel in types_logical:
	if rel.split(' ')[0] == 'not':
		attr = rel.split(' ')[1]
		dic_train_logical[rel] = [f'not {x}' for x in dic_train[attr]]
	else:
		attr1 = rel.split(' ')[0]
		r = rel.split(' ')[1]
		attr2 = rel.split(' ')[2]
		dic_train_logical[rel] = [f'{x} {r} {y}' for x, y in product(dic_train[attr1], dic_train[attr2]) if x != y]

dic_test_logical = dic_train_logical.copy()
dic_test_logical["scale"] = dic_test["scale"]
dic_test_logical["stretch"] = dic_test["stretch"]
dic_test_logical["shade"] = dic_test["shade"]

all_vocabs = []
logical_vocabs = []
for v in dic_train_logical.values():
	for n in v:
		if n not in others and n not in vocabs:
			logical_vocabs.append(n)
all_vocabs = vocabs + logical_vocabs

# count n of concepts

types_logical_with_learning_1 = types_logical_with_learning[0:2]
types_logical_with_learning_2 = types_logical_with_learning[2:4]
types_logical_with_learning_3 = types_logical_with_learning[4:6]
types_logical_with_learning_4 = types_logical_with_learning[6:8]
types_logical_with_learning_5 = types_logical_with_learning[8:10]
types_logical_with_learning_6 = types_logical_with_learning[10:12] 
types_logical_with_learning_7 = types_logical_with_learning[12:]

# <--- end new

# Objects to remove from train test
remove_from_train = [
    ('yellow', 'metal', 'torus'),
    ('white', 'rubber', 'torus_knot'),
    ('red', 'metal', 'torus'),
    ('blue', 'rubber', 'sponge'),
    ('aqua', 'glass', 'suzanne'),
    ('purple', 'glass', 'spot'),
    ('purple', 'metal', 'teapot'),
    ('green', 'metal', 'cube'),
    ('blue', 'rubber', 'cylinder'),
    ('white', 'plastic', 'torus'),
    ('red', 'rubber', 'spot'),
    ('blue', 'glass', 'spot'),
    ('blue', 'plastic', 'torus_knot'),
    ('aqua', 'metal', 'torus'),
    ('white', 'metal', 'gear'),
    ('green', 'metal', 'gear'),
    ('yellow', 'plastic', 'torus_knot'),
    ('green', 'plastic', 'gear'),
    ('red', 'plastic', 'sphere'),
    ('purple', 'rubber', 'sphere'),
    ('brown', 'metal', 'sponge'),
    ('yellow', 'plastic', 'cylinder'),
    ('red', 'glass', 'cone'),
    ('aqua', 'glass', 'spot'),
    ('yellow', 'rubber', 'torus'),
    ('brown', 'glass', 'suzanne'),
    ('green', 'rubber', 'cylinder'),
    ('red', 'rubber', 'sphere'),
    ('purple', 'plastic', 'cylinder'),
    ('yellow', 'glass', 'sphere'),
    ('blue', 'glass', 'cone'),
    ('purple', 'plastic', 'cube')]

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

bn_train = "bn_train.txt"
bn_test = "bn_test.txt"
bsn_test_1 = "bsn_test_1.txt"
bsn_test_2_nw = "bsn_test_2_nw.txt"
bsn_test_2_old = "bsn_test_2_old.txt"

# train parameters
resize = 224
lr = 1e-3
epochs = 2

sim_batch = 132
gen_batch = 132
batch_size = 33

# model architecture
hidden_dim_clip = 128
latent_dim = 16

