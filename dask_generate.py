import socket
import os
import sys
import pandas as pd

from dask_jobqueue import SLURMCluster
from dask.distributed import Client
from dask import delayed, compute

from DeepTrap import Locations, utils, BackgroundSubtraction

def start_tunnel():
    """
    Start a juypter session and ssh tunnel to view task progress
    """
    host = socket.gethostname()        
    print("To tunnel into dask dashboard:")
    print("ssh -N -L 8787:%s:8787 -l b.weinstein hpg2.rc.ufl.edu" % (host))
    
    #flush system
    sys.stdout.flush()

def run(config, debug=False):
    #Read and log config file
    
    #use local image copy
    if debug:
        config["train_data_path"] = "tests/data/sample_location"
        config["train_h5_dir"] = "/Users/Ben/Downloads/train/"
        config["test_h5_dir"] = "/Users/Ben/Downloads/test/"
        
    destination_dir = config["train_h5_dir"] 
    #check for image dir
    if not os.path.exists(destination_dir):
        os.mkdir(destination_dir)
            
    #Load train data
    train_df = pd.read_csv('data/train.csv')
    train_df['file_path'] = train_df['id'].apply(lambda x: os.path.join(config["train_data_path"], f'{x}.jpg'))
    train_df = utils.check_images(train_df, config["train_data_path"])
    
    #Sort images into location
    locations  = Locations.sort_locations(train_df)
    
    print("{} locations found".format(len(locations)))
    
    results=[]
    for location in locations:
        location_data = locations[location]
        try:
            message = delayed(Locations.preprocess_location)(location_data, destination_dir)
        except Exception as e:
            message = "{} failed with error {}".format(location,e)
        results.append(message)
    
    #Trigger dask    
    print(compute(*results))
    
    #test data
    test_df = pd.read_csv('data/test.csv')
    test_df['file_path'] = test_df['id'].apply(lambda x: os.path.join(config["test_data_path"], f'{x}.jpg'))
    test_df = utils.check_images(test_df, config["test_data_path"])
    
    destination_dir = config["test_h5_dir"] 
    #check for image dir
    if not os.path.exists(destination_dir):
        os.mkdir(destination_dir)
        
    #Sort images into location
    locations  = BackgroundSubtraction.sort_locations(test_df)
        
    results = []
    for location in locations:
        location_data = locations[location]
        try:
            message = delayed(Locations.preprocess_location)(location_data, destination_dir)
        except Exception as e:
            message = "{} failed with error {}".format(location,e)
        
        results.append(message)
        
    print(compute(*results))

def run_local():
    
    config = utils.read_config()
    
    client = Client()    
    run(config,debug=True)
    
def run_HPC():
        
    #################
    # Setup dask cluster
    #################
    
    config = utils.read_config()
    num_workers = config["num_hipergator_workers"]
    
    #job args
    extra_args=[
        "--error=/home/b.weinstein/logs/dask-worker-%j.err",
        "--account=ewhite",
        "--output=/home/b.weinstein/logs/dask-worker-%j.out"
    ]
    
    cluster = SLURMCluster(
        processes=1,
        queue='hpg2-compute',
        cores=1, 
        memory='10GB', 
        walltime='48:00:00',
        job_extra=extra_args,
        local_directory="/home/b.weinstein/logs/", death_timeout=300)
    
    print(cluster.job_script())
    cluster.adapt(minimum=num_workers, maximum=num_workers)
    
    dask_client = Client(cluster)
        
    #Start dask
    dask_client.run_on_scheduler(start_tunnel)  
    
    run(config, debug=False)

if __name__ == "__main__":
    #Local debugging
    #run_local()
    
    #On Hypergator
    run_HPC()
