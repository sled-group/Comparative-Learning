#%%
from random	import shuffle 
from statistics import stdev
from config import *
from tqdm import tqdm
from pprint import pprint

def generate_objects():
	objects = [(color, material, shape) for color in colors for material in materials for shape in shapes]
	color_material = [(color,material) for color in colors for material in materials]
	color_shape = [(color,shape) for color in colors for shape in shapes]
	material_shape = [(material,shape) for material in materials for shape in shapes]
	combs = color_material + color_shape + material_shape
	return objects, combs

def check_distribution(object_to_discard):
	discarded_attr = dict()
	for o in object_to_discard:
		for a in o:
			if a in discarded_attr.keys():
				discarded_attr[a] += 1
			else:
				discarded_attr[a] = 1
	return discarded_attr

def check_relative_stdev(attrs_count):
	c = []
	m = [] 
	s = []
	for e in colors:
		c.append(attrs_count[e])
	std_c = stdev(c)
	for e in materials:
		m.append(attrs_count[e])
	std_m = stdev(m)
	for e in shapes:
		s.append(attrs_count[e])
	std_s = stdev(s)
	return std_c + std_m + std_s

def select_objects(objects, combs):
	# Remove objects based on the specified attribute index
	min_std = 1000
	final_keep = []
	final_discard = []
	for iteration in tqdm(range(10000)):
		object_to_keep = []
		object_to_discard = []
		count = [0 for c in combs]
		updated_count = count.copy()
		shuffle(objects)
		for o in objects:
			for i, c in enumerate(combs):
				if c[0] in o and c[1] in o and o not in object_to_keep and count[i] < 2:
					count[i] += 1
				if sum(count) > sum(updated_count):
					object_to_keep.append(o)
					updated_count = count.copy()
				if len(object_to_keep) == 282:
					break
		for o in objects:
			if o not in object_to_keep:
				object_to_discard.append(o)
		
		kept_attrs = check_distribution(object_to_keep)
		discarded_attrs = check_distribution(object_to_discard)
		
		if len(discarded_attrs.keys()) == 23:
			std_kept = check_relative_stdev(kept_attrs)
			std_disc = check_relative_stdev(discarded_attrs)
			std = std_kept + std_disc
			if std < min_std:
				print('std:', std)
				min_std = std
				final_keep = object_to_keep
				final_discard = object_to_discard 
		
	return final_keep, final_discard

objects, combs = generate_objects()
object_to_keep, object_to_discard = select_objects(objects, combs)

print(len(object_to_keep))
print(len(object_to_discard))
#%%
pprint(object_to_discard)
#%%
# OLD
# N: 32, std 4.83251298454236, it 10000 minimize std of keept and discarded 
#remove_from_train = [
#    ('yellow', 'metal', 'torus'),
#    ('white', 'rubber', 'torus_knot'),
#    ('red', 'metal', 'torus'),
#    ('blue', 'rubber', 'sponge'),
#    ('aqua', 'glass', 'suzanne'),
#    ('purple', 'glass', 'spot'),
#    ('purple', 'metal', 'teapot'),
#    ('green', 'metal', 'cube'),
#    ('blue', 'rubber', 'cylinder'),
#    ('white', 'plastic', 'torus'),
#    ('red', 'rubber', 'spot'),
#    ('blue', 'glass', 'spot'),
#    ('blue', 'plastic', 'torus_knot'),
#    ('aqua', 'metal', 'torus'),
#    ('white', 'metal', 'gear'),
#    ('green', 'metal', 'gear'),
#    ('yellow', 'plastic', 'torus_knot'),
#    ('green', 'plastic', 'gear'),
#    ('red', 'plastic', 'sphere'),
#    ('purple', 'rubber', 'sphere'),
#    ('brown', 'metal', 'sponge'),
#    ('yellow', 'plastic', 'cylinder'),
#    ('red', 'glass', 'cone'),
#    ('aqua', 'glass', 'spot'),
#    ('yellow', 'rubber', 'torus'),
#    ('brown', 'glass', 'suzanne'),
#    ('green', 'rubber', 'cylinder'),
#    ('red', 'rubber', 'sphere'),
#    ('purple', 'plastic', 'cylinder'),
#    ('yellow', 'glass', 'sphere'),
#    ('blue', 'glass', 'cone'),
#    ('purple', 'plastic', 'cube')]

# N: 71 std: 6.299084526452528
remove_from_train = [('green', 'rubber', 'gear'),
 ('brown', 'metal', 'torus'),
 ('green', 'plastic', 'teapot'),
 ('red', 'metal', 'spot'),
 ('white', 'rubber', 'gear'),
 ('purple', 'plastic', 'gear'),
 ('brown', 'metal', 'sponge'),
 ('aqua', 'metal', 'cube'),
 ('brown', 'metal', 'teapot'),
 ('yellow', 'metal', 'sponge'),
 ('brown', 'glass', 'torus'),
 ('yellow', 'glass', 'suzanne'),
 ('aqua', 'glass', 'cube'),
 ('yellow', 'metal', 'cone'),
 ('brown', 'plastic', 'gear'),
 ('green', 'plastic', 'torus'),
 ('white', 'glass', 'suzanne'),
 ('red', 'glass', 'sphere'),
 ('blue', 'rubber', 'cube'),
 ('blue', 'plastic', 'suzanne'),
 ('green', 'plastic', 'sponge'),
 ('white', 'metal', 'torus'),
 ('aqua', 'plastic', 'sponge'),
 ('yellow', 'glass', 'sponge'),
 ('white', 'rubber', 'torus_knot'),
 ('brown', 'rubber', 'suzanne'),
 ('purple', 'plastic', 'teapot'),
 ('white', 'rubber', 'sphere'),
 ('yellow', 'glass', 'cylinder'),
 ('brown', 'rubber', 'torus_knot'),
 ('brown', 'glass', 'spot'),
 ('white', 'plastic', 'sponge'),
 ('purple', 'glass', 'cylinder'),
 ('purple', 'plastic', 'spot'),
 ('aqua', 'glass', 'sphere'),
 ('green', 'rubber', 'cone'),
 ('purple', 'glass', 'suzanne'),
 ('blue', 'glass', 'cylinder'),
 ('red', 'glass', 'cube'),
 ('blue', 'metal', 'gear'),
 ('blue', 'glass', 'sphere'),
 ('blue', 'metal', 'teapot'),
 ('aqua', 'metal', 'teapot'),
 ('brown', 'metal', 'spot'),
 ('blue', 'glass', 'torus_knot'),
 ('aqua', 'glass', 'torus_knot'),
 ('purple', 'rubber', 'sponge'),
 ('yellow', 'rubber', 'sponge'),
 ('purple', 'plastic', 'torus_knot'),
 ('white', 'rubber', 'cube'),
 ('blue', 'metal', 'torus'),
 ('blue', 'metal', 'sponge'),
 ('brown', 'plastic', 'sphere'),
 ('aqua', 'plastic', 'cylinder'),
 ('red', 'plastic', 'cone'),
 ('green', 'plastic', 'cube'),
 ('red', 'metal', 'cone'),
 ('purple', 'rubber', 'cylinder'),
 ('red', 'glass', 'cylinder'),
 ('red', 'metal', 'sphere'),
 ('green', 'metal', 'suzanne'),
 ('red', 'plastic', 'torus_knot'),
 ('white', 'rubber', 'teapot'),
 ('blue', 'plastic', 'teapot'),
 ('purple', 'rubber', 'sphere'),
 ('aqua', 'rubber', 'suzanne'),
 ('purple', 'plastic', 'torus'),
 ('aqua', 'glass', 'cone'),
 ('red', 'rubber', 'cylinder'),
 ('yellow', 'rubber', 'suzanne')]

