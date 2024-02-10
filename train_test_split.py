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

#objects, combs = generate_objects()
#object_to_keep, object_to_discard = select_objects(objects, combs)

# N: 32, std 4.83251298454236, it 10000 minimize std of keept and discarded 
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

