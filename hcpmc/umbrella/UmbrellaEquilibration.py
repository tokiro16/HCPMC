# updated for hoomd4 by Jin Wang. 2024/04/16

from hoomd import *
import numpy as np 
import time

def naiveTimeAverage(UW, trials_per_force_accept = 200, close_enough_count_max = 3, close_enough_percent = 0.02, time_average = 20):
    """ Standard brute force method with modification on equilibration criteria and when to do force update"""
    time0 = time.time()
    print("\n\nSTEP1: Starting naive time average equilibration routines\n")
    print(" trials_per_force_accept : " + str(trials_per_force_accept))
    print(" close_enough_count_max : " + str(close_enough_count_max))
    print("\n") 
    UW.disableLogging() 
    UW.runUmbrellaTrials(trials_per_force_accept) 
    curr_order = np.zeros(time_average)
    umbrellatrialnumber = 0
    curr_order +=  UW.getOrder() 

    close_enough_count = 0 
    force_update_number = 0 
    while(close_enough_count < close_enough_count_max):
        time1 = time.time()
        print("\nBeginning another new brute force equilibration iteration     ") 
        print(" OrderDifference = " + str(UW.getOrder() - UW.getOrderTarget())) 
        UW.runUmbrellaTrials(trials_per_force_accept)
        if UW.getAcceptanceRatio() < 0.02:
            UW.simulation.run(UW.getTrajectoryLength())
            UW.forceUpdate()
            print("\nforce_update_number : " + str(force_update_number))
            force_update_number += 1
            UW.resetAcceptanceStatistics()
        curr_order[umbrellatrialnumber % time_average] = UW.getOrder()
        umbrellatrialnumber += 1
        print('This iteration costed {:.1f} seconds. Total costed {:.3f} hours.'.format(time.time()-time1,(time.time()-time0)/3600))
        if ( (abs(UW.getOrderTarget() - np.mean(curr_order)) ) / UW.getOrderTarget() < close_enough_percent ):
            close_enough_count += 1 
            print("\nclose_enough_count upticked to : " + str(close_enough_count)) 


def movingTargetEquil(UW):
    """ Create a moving target towards the true order target """
    print("\n\nSTEP1: Trying moving target less-stupid equilibration routine\n") 
    #Magical constants
    close_enough_distance = 10 
    trials_per_window     = 10 
    trials_per_equilibration_iteration = 200 
    equilibrated       = False 
    order_parameter_offset = 15   #Set order parameter to be "This" far away from where system is at

    #Disable logging, since I'll be tweaking a lot of settings in here.
    UW.disableLogging() 

    #Poll target from UW class
    final_order_target = UW.getOrderTarget() 

    #Reset the system to the current state (bias can be lost easily in statistical noise otherwise)
    run(200) 
    UW.forceUpdate() 
    curr_order = UW.getOrder() 

    while(not equilibrated):
        print("Starting equilibration loop iteration") 
        #Use curr_order to set a new target
        if(curr_order > final_order_target):
            temp_order_target = curr_order - order_parameter_offset 
        elif(curr_order < final_order_target):
            temp_order_target = curr_order + order_parameter_offset 
        else: #curr order = final target
            temp_order_target = final_order_target 
        print(" Current Order: " + str(curr_order) +", Current Target: "+str(temp_order_target)) 
        print(" Actual Target: " + str(final_order_target)) 
        #Trial some with the current bias
        UW.setOrderTarget(temp_order_target) 
        UW.runUmbrellaTrials(trials_per_equilibration_iteration) 
        UW.printAcceptanceStatistics() 
        UW.resetAcceptanceStatistics() 

        #Force update? Maybe not needed
        #Update - statistical noise can damn you to hell
        run(UW.getTrajectoryLength()) 
        UW.forceUpdate() 

        curr_order = UW.getOrder() 

        #Check if we're close enough to call us done.
        if( (abs(final_order_target-curr_order)) < close_enough_distance):
            equilibrated=True 
    #Restore state of the UW class.
    UW.setOrderTarget(final_order_target) 
    UW.resetAcceptanceStatistics() 



#moving target equilibration with sampling of the past N windows, currently working in progress, create new function so the old one doesn't break
def movingTargetEquilNew(UW, close_enough_distance, trials_per_window, trials_per_equilibration_iteration, equilibrated, order_parameter_offset):
    """ Create a moving target towards the true order target """

    print("\n\nSTEP1: Trying moving target less-stupid equilibration routine\n") 
    #Magical constants
    #close_enough_distance = 10 
    #trials_per_window     = 10 
    #trials_per_equilibration_iteration = 200 
    #equilibrated       = False 
    #order_parameter_offset = 15   #Set order parameter to be "This" far away from where system is at

    time_average = 20
    curr_order = numpy.zeros(time_average)
    umbrellatrialnumber = 0
    #Disable logging, since I'll be tweaking a lot of settings in here.
    UW.disableLogging() 

    #Poll target from UW class
    final_order_target = UW.getOrderTarget() 
    close_enough_distance = final_order_target * 0.03 
    #Reset the system to the current state (bias can be lost easily in statistical noise otherwise)
    run(200) 
    UW.forceUpdate() 
    curr_order +=  UW.getOrder() 

    while(not equilibrated):
        print("Starting equilibration loop iteration") 
        #Use curr_order to set a new target
        if(numpy.mean(curr_order) > final_order_target):
            temp_order_target = numpy.mean(curr_order) - order_parameter_offset 
        elif(numpy.mean(curr_order) < final_order_target):
            temp_order_target = numpy.mean(curr_order) + order_parameter_offset 
        else: #curr order = final target
            temp_order_target = final_order_target 
        print(" Current Order: " + str(numpy.mean(curr_order)) +", Current Target: "+str(temp_order_target)) 
        print(" Actual Target: " + str(final_order_target)) 
        #Trial some with the current bias
        UW.setOrderTarget(temp_order_target) 
        UW.runUmbrellaTrials(trials_per_equilibration_iteration) 
        UW.printAcceptanceStatistics() 
        UW.resetAcceptanceStatistics() 

        #Force update? Maybe not needed
        #Update - statistical noise can damn you to hell
        run(UW.getTrajectoryLength()) 
        UW.forceUpdate() 

        curr_order[umbrellatrialnumber % time_average] = UW.getOrder() 
        umbrellatrialnumber += 1 

        #Check if we're close enough to call us done.
        if( (abs(final_order_target-numpy.mean(curr_order))) < close_enough_distance):
            equilibrated=True 
    #Restore state of the UW class.
    UW.setOrderTarget(final_order_target) 
    UW.resetAcceptanceStatistics() 


#moving target equilibration with sampling of the past N windows, currently working in progress, create new function so the old one doesn't break
def movingTargetEquilTest(UW, close_enough_distance, trials_per_window, trials_per_equilibration_iteration, equilibrated, order_parameter_offset, time_average):
    """ Create a moving target towards the true order target """

    print("\n\nSTEP1: Trying moving target less-stupid equilibration routine\n") 
    #Magical constants
    #close_enough_distance = 10 
    #trials_per_window     = 10 
    #trials_per_equilibration_iteration = 200 
    #equilibrated       = False 
    #order_parameter_offset = 15   #Set order parameter to be "This" far away from where system is at

    #time_average = 20
    curr_order = numpy.zeros(time_average)
    umbrellatrialnumber = 0
    #Disable logging, since I'll be tweaking a lot of settings in here.
    UW.disableLogging() 

    #Poll target from UW class
    final_order_target = UW.getOrderTarget() 
    close_enough_distance = final_order_target * 0.03 
    #Reset the system to the current state (bias can be lost easily in statistical noise otherwise)
    run(200) 
    UW.forceUpdate() 
    curr_order +=  UW.getOrder() 

    while(not equilibrated):
        print("Starting equilibration loop iteration") 
        #Use curr_order to set a new target
        if(numpy.mean(curr_order) > final_order_target):
            temp_order_target = numpy.mean(curr_order) - order_parameter_offset 
        elif(numpy.mean(curr_order) < final_order_target):
            temp_order_target = numpy.mean(curr_order) + order_parameter_offset 
        else: #curr order = final target
            temp_order_target = final_order_target 
        print(" Current Order: " + str(numpy.mean(curr_order)) +", Current Target: "+str(temp_order_target)) 
        print(" Actual Target: " + str(final_order_target)) 
        #Trial some with the current bias
        UW.setOrderTarget(temp_order_target) 
        UW.runUmbrellaTrials(trials_per_equilibration_iteration) 
        UW.printAcceptanceStatistics() 
        UW.resetAcceptanceStatistics() 

        #Force update? Maybe not needed
        #Update - statistical noise can damn you to hell
        run(UW.getTrajectoryLength()) 
        UW.forceUpdate() 

        curr_order[umbrellatrialnumber % time_average] = UW.getOrder() 
        umbrellatrialnumber += 1 

        #Check if we're close enough to call us done.
        if( (abs(final_order_target-numpy.mean(curr_order))) < close_enough_distance):
            equilibrated=True 
    #Restore state of the UW class.
    UW.setOrderTarget(final_order_target) 
    UW.resetAcceptanceStatistics() 
