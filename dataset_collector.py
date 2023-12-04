from rembg import remove
from PIL import Image 
import numpy as np
import matplotlib.pyplot as plt
import sys
import sim
import cv2
import time
import math
import os








def find_set_number(dataset_folder):
    file_list = os.listdir(dataset_folder)

    max_number = 0

    for file_name in file_list:
        if file_name[0].isdigit():
            number = int(file_name[0])

            if number > max_number:
                max_number = number
    return max_number + 1


def separate_objects(mask):
    # Apply connected component labeling
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Create an array to store individual objects
    objects = []

    # Iterate over each label (excluding background label 0)
    for label in range(1, num_labels):
        # Extract the object region based on the label
        object_region = (labels == label).astype(np.uint8)
        # Add the object region to the list of objects
        objects.append(object_region)

    return objects




def import_obj(clientID,obj_number):
    res, ret_string, ret_float, ret_int, ret_buffer = sim.simxCallScriptFunction(clientID, '/Mycub/Dummy',
                                                                                 sim.sim_scripttype_childscript,
                                                                                 'getParameter',
                                                                                 [obj_number],
                                                                                 [obj_number], [str(obj_number)], b'', sim.simx_opmode_blocking)
    errorCode , Handle = sim.simxGetObjectHandle(clientID,str(obj_number),sim.simx_opmode_blocking)
    
    
    
    #sim.simxSetModelProperty(clientID, Handle, sim.sim_objectspecialproperty_collidable , sim.simx_opmode_oneshot)
    sim.simxSetObjectPosition(clientID, Handle, -1,[0,0,0.25], sim.simx_opmode_blocking)
    orientation = [math.pi/2, 0.0, math.pi/2]  # Adjust the orientation angles as needed
    res = sim.simxSetObjectOrientation(clientID, Handle, -1, orientation, sim.simx_opmode_blocking)
    
    res, ret_string, ret_float, ret_int, ret_buffer = sim.simxCallScriptFunction(clientID, '/Mycub/Dummy',sim.sim_scripttype_childscript,'set_dynamic',[obj_number],[obj_number], [str(obj_number)], b'', sim.simx_opmode_blocking)

    
    

for i in range(1,501):
    sim.simxFinish(-1) 
    clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5) 
    
    if clientID!=-1:
        print ('Connected to remote API server')
    else:
        print ('Failed connecting to remote API server')
        sys.exit('FAILED!')
        
        
    sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
    
    obj_number = i
    dataset_folder = "./dataset/"
    
    
    import_obj(clientID, obj_number)
    
    
    set_name = find_set_number(dataset_folder)
    
    
    errorCode , camHandle = sim.simxGetObjectHandle(clientID,'Vision_sensor',sim.simx_opmode_blocking)
    errorCode1, resolution, raw_image = sim.simxGetVisionSensorImage(clientID, camHandle,0, sim.simx_opmode_blocking)
    if errorCode1 == sim.simx_return_ok:
        print('Vision sensor image retrieved successfully')
    else:
        print('Failed to retrieve vision sensor image')
    sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
    ####simulation end
    
    rgb_image = np.uint8(np.reshape(raw_image, (resolution[1], resolution[0], 3)))
    
    
    
    image  = Image.fromarray(rgb_image)
    
    out_path = dataset_folder+f"{obj_number}_nobg_rgb.png"
    
    raw_out_img = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    
    raw_path = dataset_folder + f"{obj_number}_raw_rgb.png"
    cv2.imwrite(raw_path,raw_out_img)
    
    output = remove(image,alpha_matting_background_threshold=0,alpha_matting_foreground_threshold=200)
    output.save(out_path)
    
    
    img = np.array(output)
    
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create a mask where all non-zero pixels are set to 255 (white)
    mask = np.where(grayscale != 0, 255, 0).astype(np.uint8)
    
    #mask = cv2.GaussianBlur(mask, (3,3), 1)
    mask = cv2.medianBlur(mask,3)
    
    cv2.imwrite(dataset_folder+f"{obj_number}_mask.png",mask)















