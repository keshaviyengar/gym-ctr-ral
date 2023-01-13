import numpy as np
from collections import OrderedDict

three_tube_ctr_systems = {
    # Autonomous steering by Mohsen Khadem
    'system_0': {
        'inner':
            {'length': 431e-3, 'length_curved': 103e-3, 'diameter_inner': 0.7e-3,
             'diameter_outer': 1.10e-3,
             'stiffness': 10.25e+10, 'torsional_stiffness': 18.79e+10, 'x_curvature': 21.3,
             'y_curvature': 0
             },

        'middle':
            {'length': 332e-3, 'length_curved': 113e-3, 'diameter_inner': 1.4e-3,
             'diameter_outer': 1.8e-3,
             'stiffness': 68.6e+10, 'torsional_stiffness': 11.53e+10, 'x_curvature': 13.1,
             'y_curvature': 0
             },

        'outer':
            {'length': 174e-3, 'length_curved': 134e-3, 'diameter_inner': 2e-3, 'diameter_outer': 2.4e-3,
             'stiffness': 16.96e+10, 'torsional_stiffness': 14.25e+10, 'x_curvature': 3.5,
             'y_curvature': 0
             }
    },
    # Learning the FK and IK of a 6-DOF CTR by Grassmann
    'system_1': {
        'inner':
            {'length': 370e-3, 'length_curved': 45e-3, 'diameter_inner': 0.3e-3, 'diameter_outer': 0.4e-3,
             'stiffness': 50e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 15.8, 'y_curvature': 0,
             },

        'middle':
            {'length': 305e-3, 'length_curved': 100e-3, 'diameter_inner': 0.7e-3,
             'diameter_outer': 0.9e-3,
             'stiffness': 50e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 9.27, 'y_curvature': 0,
             },

        'outer':
            {'length': 170e-3, 'length_curved': 100e-3, 'diameter_inner': 1.2e-3,
             'diameter_outer': 1.5e-3,
             'stiffness': 50e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 4.37, 'y_curvature': 0,
             }
    },
    # RViM lab tube parameters
    'system_2': {
        'inner':
            {'length': 309e-3, 'length_curved': 145e-3, 'diameter_inner': 0.7e-3,
             'diameter_outer': 1.1e-3,
             'stiffness': 75e+9, 'torsional_stiffness': 25e+9, 'x_curvature': 13.52, 'y_curvature': 0
             },
        'middle':
            {'length': 275e-3, 'length_curved': 114e-3, 'diameter_inner': 1.4e-3,
             'diameter_outer': 1.8e-3,
             'stiffness': 75e+9, 'torsional_stiffness': 25e+9, 'x_curvature': 11.68, 'y_curvature': 0
             },
        'outer':
            {'length': 173e-3, 'length_curved': 173e-3, 'diameter_inner': 1.83e-3,
             'diameter_outer': 2.39e-3,
             'stiffness': 75e+9, 'torsional_stiffness': 25e+9, 'x_curvature': 10.8, 'y_curvature': 0
             }
    },
    # Unknown tube parameters or where they are from
    'system_3': {
        'inner':
            {'length': 150e-3, 'length_curved': 100e-3, 'diameter_inner': 1.0e-3,
             'diameter_outer': 2.4e-3,
             'stiffness': 5e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 15.82, 'y_curvature': 0},

        'middle':
            {'length': 100e-3, 'length_curved': 21.6e-3, 'diameter_inner': 3.0e-3,
             'diameter_outer': 3.8e-3,
             'stiffness': 5e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 11.8, 'y_curvature': 0},

        'outer':
            {'length': 70e-3, 'length_curved': 8.8e-3, 'diameter_inner': 4.4e-3, 'diameter_outer': 5.4e-3,
             'stiffness': 5.0e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 20.04, 'y_curvature': 0}
    },
}
two_tube_ctr_systems = {
    # Autonomous steering by Mohsen Khadem
    'system_0': {
        'inner':
            {'length': 431e-3, 'length_curved': 103e-3, 'diameter_inner': 0.7e-3,
             'diameter_outer': 1.10e-3,
             'stiffness': 10.25e+10, 'torsional_stiffness': 18.79e+10, 'x_curvature': 21.3,
             'y_curvature': 0
             },
        'outer':
            {'length': 174e-3, 'length_curved': 134e-3, 'diameter_inner': 2e-3, 'diameter_outer': 2.4e-3,
             'stiffness': 16.96e+10, 'torsional_stiffness': 14.25e+10, 'x_curvature': 3.5,
             'y_curvature': 0
             }
    },
    # Learning the FK and IK of a 6-DOF CTR by Grassmann
    'system_1': {
        'inner':
            {'length': 370e-3, 'length_curved': 45e-3, 'diameter_inner': 0.3e-3, 'diameter_outer': 0.4e-3,
             'stiffness': 50e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 15.8, 'y_curvature': 0,
             },
        'outer':
            {'length': 170e-3, 'length_curved': 100e-3, 'diameter_inner': 1.2e-3,
             'diameter_outer': 1.5e-3,
             'stiffness': 50e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 4.37, 'y_curvature': 0,
             }
    },
    # RViM lab tube parameters
    'system_2': {
        'inner':
            {'length': 309e-3, 'length_curved': 145e-3, 'diameter_inner': 0.7e-3,
             'diameter_outer': 1.1e-3,
             'stiffness': 75e+9, 'torsional_stiffness': 25e+9, 'x_curvature': 13.52, 'y_curvature': 0
             },
        'outer':
            {'length': 173e-3, 'length_curved': 173e-3, 'diameter_inner': 1.83e-3,
             'diameter_outer': 2.39e-3,
             'stiffness': 75e+9, 'torsional_stiffness': 25e+9, 'x_curvature': 10.8, 'y_curvature': 0
             }
    },
    # Unknown tube parameters or where they are from
    'system_3': {
        'inner':
            {'length': 150e-3, 'length_curved': 100e-3, 'diameter_inner': 1.0e-3,
             'diameter_outer': 2.4e-3,
             'stiffness': 5e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 15.82, 'y_curvature': 0},
        'outer':
            {'length': 70e-3, 'length_curved': 8.8e-3, 'diameter_inner': 4.4e-3, 'diameter_outer': 5.4e-3,
             'stiffness': 5.0e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 20.04, 'y_curvature': 0}
    },
}
