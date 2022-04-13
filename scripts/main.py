#!/usr/bin/env python3

#### DO NOT CHANGE THESE IMPORTS
import numpy
import time 
import pathlib
####

#### TODO: ADD YOUR IMPORTS HERE

from math import sqrt
from pypointmatcher import pointmatcher as pm, pointmatchersupport as pms
from utils import parse_translation, parse_rotation


PM = pm.PointMatcher
DP = PM.DataPoints
Parameters = pms.Parametrizable.Parameters
####

# Be more verbose (info logging to the console)
is_verbose = True

# Load the config from a YAML file
config_file = "./icp_cfg.yaml"

# Add an initial 3D translation before applying ICP (default: 0,0,0)
init_translation = "0,0,0"
# Add an initial 3D rotation before applying ICP (default: 1,0,0;0,1,0;0,0,1)
init_rotation = "1,0,0;0,1,0;0,0,1"

def main():
    data_list = ["bag", "basketball", "computercluster1", "corner2", "lab1", "sofalong", "sofawhole", "threechair", "threemonitor"]
    # data_list = ["threemonitor"]

    # repo_location = pathlib.Path(__file__).parent.parent.absolute()
    # data_folder = repo_location / 'data'

    data_folder = '../data'

    print(data_folder)

    # for location in data_list:
    for location in data_list:

        path_to_ply1 = data_folder + '/' + location + '/' + 'kinect.ply'
        path_to_ply2 = data_folder + '/' + location + '/' + 'sfm.ply'

        start_time = time.time()
        
        # TODO: Add your code here

        print(path_to_ply1)
        print(path_to_ply2)

        # Load 3D point clouds
        ref = DP(DP.load(path_to_ply1))
        data = DP(DP.load(path_to_ply2))
        test_base = "3D"

        output_base_directory = "results/"
        output_base_file = location

        # Create the default ICP algorithm
        icp = PM.ICP()

        if len(config_file) == 0:
            # See the implementation of setDefault() to create a custom ICP algorithm
            icp.setDefault()
        else:
            # load YAML config
            icp.loadFromYaml(config_file)        

        cloud_dimension = ref.getEuclideanDim()

        assert cloud_dimension == 2 or cloud_dimension == 3, "Invalid input point clouds dimension"

        # Parse the translation and rotation to be used to compute the initial transformation
        translation = parse_translation(init_translation, cloud_dimension)
        rotation = parse_rotation(init_rotation, cloud_dimension)

        init_transfo = numpy.matmul(translation, rotation)
        rigid_trans = PM.get().TransformationRegistrar.create("RigidTransformation")


        if not rigid_trans.checkParameters(init_transfo):
            print("Initial transformations is not rigid, identiy will be used")
            init_transfo = numpy.identity(cloud_dimension + 1)

        initialized_data = rigid_trans.compute(data, init_transfo)

        # Compute the transformation to express data in ref
        T = icp(initialized_data, ref)
        match_ratio = icp.errorMinimizer.getWeightedPointUsedRatio()
        print(f"match ratio: {match_ratio:.6}")

        # Transform data to express it in ref
        data_out = DP(initialized_data)
        icp.transformations.apply(data_out, T)

        # Save files to see the results
        ref.save(f"{output_base_directory + test_base}_{output_base_file}_ref.vtk")
        data.save(f"{output_base_directory + test_base}_{output_base_file}_data_in.vtk")
        data_out.save(f"{output_base_directory + test_base}_{output_base_file}_data_out.vtk")
        
        # print(f"Final {test_base} transformations:\n{T}\n".replace("[", " ").replace("]", " "))

        if is_verbose:
            print(f"{test_base} ICP transformation:\n{T}".replace("[", " ").replace("]", " "))

        print ('Execution time : {}'.format(time.time() - start_time))

if __name__ == "__main__":
    main()
