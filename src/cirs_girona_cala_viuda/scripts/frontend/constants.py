import numpy as np

# For plotting cones in our figures
# cone_times_ordered = np.array([1372687222.817922, 1372687282.219970, 1372687385.219641, 1372687430.219218, 1372687570.418142, 1372687896.077008, 1372688246.475310, 
#                        1372688413.886930, 1372688477.887347, 1372688594.087488, 1372688650.085620, 1372689028.086277, 1372689150.885724])
# cone_times = np.array([[1372687222.817922, 1372687282.219970, 1372687385.219641, 1372687430.219218, 1372687570.418142, 1372687896.077008, 1372688246.475310], 
#                        [1372688413.886930, 1372688477.887347, 1372688594.087488, 1372688650.085620, 1372689028.086277, 1372689150.885724]])

# cone_times_ordered = np.array([1372687222.817922, 1372687282.219970, 1372687385.219641, 1372687430.219218, 1372687570.418142, 1372687896.077008, 
#                        1372688413.886930, 1372688477.887347, 1372688594.087488, 1372688650.085620, 1372689028.086277, 1372689150.885724])

# These times were shared by Angelos Mallios (dataset creator)
cone_times_ordered = np.array([1372687223.020753194,
                      1372687283.218789086,
                      1372687384.817663679,
                      1372687428.818958879,
                      1372687570.418141745,
                      1372687895.875383276,
                      1372688414.086915358,
                      1372688477.887347035,
                      1372688594.486129917,
                      1372688650.085619758,
                      1372689028.086277101,
                      1372689151.285564780])

# First and second time each cone 1-6 seen
cone_times = np.array([[cone_times_ordered[0], cone_times_ordered[1], cone_times_ordered[2], cone_times_ordered[3], cone_times_ordered[4], cone_times_ordered[9]],
                       [cone_times_ordered[11], cone_times_ordered[8], cone_times_ordered[7], cone_times_ordered[6], cone_times_ordered[5], cone_times_ordered[10]]]).T

# x, y offsets for each cone (ordered) - shared by Angelos Mallios (dataset creator)
# (dim, timestep)
cone_offsets_ordered = np.array([[-0.4572, -0.1083,  -0.3835, 1.2975, -0.0596, 0.3341, -0.4925, -0.0917, -0.6535, -0.3835, 0.1669, 0.5535],
                         [0.0129, -0.2070, 0.0715, 0.2486, -0.0565, -0.0290, -0.0357, -0.1566, -0.1308, 0.0130, -0.1495, -0.0272]])
cone_offsets_ordered[1,:] = -cone_offsets_ordered[1,:]

# (pass#, cone#, dim)
cone_offsets = np.array([[cone_offsets_ordered[:,0], cone_offsets_ordered[:,1], cone_offsets_ordered[:,2], cone_offsets_ordered[:,3], cone_offsets_ordered[:,4], cone_offsets_ordered[:,9]],
                       [cone_offsets_ordered[:,11], cone_offsets_ordered[:,8], cone_offsets_ordered[:,7], cone_offsets_ordered[:,6], cone_offsets_ordered[:,5], cone_offsets_ordered[:,10]]]).T

# Ordering of cone observations is 123455432661
cone_path = [0, 1, 2, 3, 4, 4, 3, 2, 1, 5, 5, 0]


cone_distances = {(0,1): 19.,
                  (1,0): 19.,
                  (1,2): 32.,
                  (2,1): 32.,
                  (2,3): 16.,
                  (3,2): 16.,
                  (0,5): 30.,
                  (5,0): 30.}