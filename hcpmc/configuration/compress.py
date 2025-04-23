#yhyang
# update datetime by Jin Wang.
#hoomd-version=3.x/4.x
import hoomd
import numpy as np
import datetime


class hpmc_compress():
    def __init__(self, sim, mc):
        '''
        Args:
            sim: simulation
            mc: Monte Carlo integrater
        '''
        self.sim = sim
        self.mc = mc

    def box_compress(self, delta_d, run_step, start_time, particle_volume=1, particle_types=[], shear=True):
        """HPMC compress.

        Args:
            delta_d: Maximum size of displacement trial moves
            run_step: Relax steps after compress
            particle_volume: Volume of a single particle
            particle_types: types of particles that delta_d is setted

            

        This algorithm compress (or expand) system until acceptance rate reach 0.25-0.35.
        With a delta_d given, the acceptance rate of translational move depends on the displacement of two particles.
        The system will be compressed until the acceptance rate of translational move < 0.35 and vice versa (unitl P_acc > 0.25).
        Only compress of Lx, Ly and Lz supported.
        
        Warning: This algorithm need delta_d fixed. Turn off tuner of translational move before this method used.
        """
        
        
        #set delta_d for all types of particles
        if particle_types == []:
            particle_types = self.sim.state.particle_types
        #set delta_d for all types of particles
        for particle_type in particle_types:
            self.mc.d[particle_type] = delta_d
        N = self.sim.state.N_particles
        self.sim.run(100)
        
        # get acceptance
        acceptance = self.mc.translate_moves[0]/(self.mc.translate_moves[0] + self.mc.translate_moves[1])
        print('acceptance is ' + str(acceptance) + ', delta_d is ' + str(delta_d))
        
        #perform compress or expand
        while acceptance < 0.25 or acceptance > 0.35:
            #store the previous configuration
            snapshot = self.sim.state.get_snapshot()

            #scale the box
            randint = np.random.randint(3)
            if acceptance > 0.3:
                #compress
                scale_ratio = np.random.uniform(1.0 - 0.002*delta_d, 1)
            else:
                #expand
                scale_ratio = np.random.uniform(1, 1.0 + 0.002*delta_d)
            if shear:
                if randint == 0:
                    hoomd.update.BoxResize.update(self.sim.state, self.sim.state.box.scale((scale_ratio, 1, 1)), hoomd.filter.All())
                elif randint == 1:
                    hoomd.update.BoxResize.update(self.sim.state, self.sim.state.box.scale((1, scale_ratio, 1)), hoomd.filter.All())
                elif randint == 2:
                    hoomd.update.BoxResize.update(self.sim.state, self.sim.state.box.scale((1, 1, scale_ratio)), hoomd.filter.All())
            else:
                hoomd.update.BoxResize.update(self.sim.state, self.sim.state.box.scale((scale_ratio, scale_ratio, 1)), hoomd.filter.All())
            #callback the system if there is too much overlaps for compression
            if acceptance > 0.3:
                overlaps = self.mc.overlaps
                if overlaps*1.0/N > 0.002 and overlaps > 2:
                    self.sim.state.set_snapshot(snapshot)
                    #print('\rtoo much overlaps detected, callback the system. N_overlaps = ' + str(overlaps), end='', flush=True)
            
            #relax
            overlaps = self.mc.overlaps
            while overlaps >= 1:
                self.sim.run(50)
                overlaps = self.mc.overlaps
            self.sim.run(run_step)
            acceptance = self.mc.translate_moves[0]/(self.mc.translate_moves[0] + self.mc.translate_moves[1])
            box_volume = self.sim.state.box.volume
            phi = N * particle_volume / box_volume
            duration = datetime.datetime.now() - start_time
            formatted_duration = str(duration).split(".")[0]
            if acceptance > 0.35:
                print(f'{formatted_duration} acc: {acceptance:.3f} delta_d: {delta_d}, compress. box volume: {box_volume} phi: {phi}', flush=True)
            elif acceptance < 0.25:
                print(f'{formatted_duration} acc: {acceptance:.3f} delta_d: {delta_d}, expand.   box volume: {box_volume} phi: {phi}', flush=True)
        print(' ')

    '''
    def resize_wall(self, ratio, wall):
        #resize wall
        self.mc.external_potential.walls.remove(wall)
        if type(wall) == hoomd.wall.Sphere:
            wall_radius = wall.to_dict()['radius']
            wall = hoomd.wall.Sphere(radius=wall_radius*ratio)
        elif type(wall) == hoomd.wall.Plane:
            origin = np.array(wall.origin)*ratio
            normal = wall.normal
            wall = hoomd.wall.Plane(origin=origin, normal=normal)
        self.mc.external_potential.walls.append(wall)
        #resize particle position but leave box not changed
        old_box = self.sim.state.get_snapshot().configuration.box
        hoomd.update.BoxResize.update(self.sim.state, self.sim.state.box.scale(ratio))
        self.sim.state.set_box(old_box)
        return wall
    '''
    
    def resize_wall(self, ratio, walls):
        self.mc.external_potential.walls.clear()
        for i,wall in enumerate(walls):
            if type(wall) == hoomd.wall.Sphere:
                wall_radius = wall.to_dict()['radius']
                walls[i] = hoomd.wall.Sphere(radius=wall_radius*ratio)
                self.mc.external_potential.walls.append(walls[i])
            elif type(wall) == hoomd.wall.Plane:
                origin = np.array(wall.origin)*ratio
                normal = wall.normal
                walls[i] = hoomd.wall.Plane(origin=origin, normal=normal)
                self.mc.external_potential.walls.append(walls[i])
        old_box = self.sim.state.get_snapshot().configuration.box
        hoomd.update.BoxResize.update(self.sim.state, self.sim.state.box.scale(1*ratio))
        self.sim.state.set_box(old_box)


    def wall_compress(self, delta_d, run_step, walls, particle_types=[]):
        if particle_types == []:
            particle_types = self.sim.state.particle_types
        #set delta_d for all types of particles
        for particle_type in particle_types:
            self.mc.d[particle_type] = delta_d
        N = self.sim.state.N_particles
        self.sim.run(100)
        
        #get acceptance
        acceptance = self.mc.translate_moves[0]/(self.mc.translate_moves[0] + self.mc.translate_moves[1])
        if N > 500:
            if self.sim.state.box.dimensions == 2:
                N_near_boundary_ratio = (3*N)**(2/3)/N
            else:
                N_near_boundary_ratio = (4*N)**(3/4)/N
            reduced_acceptance = acceptance/(1 - N_near_boundary_ratio/2)
        else:
            reduced_acceptance = acceptance
        print('acceptance is ' + str(reduced_acceptance) + ', delta_d is ' + str(delta_d))
        #scale the wall
        
        while reduced_acceptance < 0.2 or reduced_acceptance > 0.3:
            if reduced_acceptance > 0.3:
                #compress
                scale_ratio = np.random.uniform(1.0 - 0.002*delta_d, 1)
            else:
                #expand
                scale_ratio = np.random.uniform(1, 1.0 + 0.002*delta_d)
                
            #resize particle position but leave box not changed
            old_box = self.sim.state.get_snapshot().configuration.box
            hoomd.update.BoxResize.update(self.sim.state, self.sim.state.box.scale(scale_ratio))
            self.sim.state.set_box(old_box)
            
            self.mc.external_potential.walls.clear()
            for i,wall in enumerate(walls):
                if type(wall) == hoomd.wall.Sphere:
                    wall_radius = wall.to_dict()['radius']
                    walls[i] = hoomd.wall.Sphere(radius=wall_radius*scale_ratio)
                    self.mc.external_potential.walls.append(walls[i])
                elif type(wall) == hoomd.wall.Plane:
                    origin = np.array(wall.origin)*scale_ratio
                    normal = wall.normal
                    walls[i] = hoomd.wall.Plane(origin=origin, normal=normal)
                    self.mc.external_potential.walls.append(walls[i])
            #print('scaled')
            #callback the system if there is too much overlaps for compression
            if reduced_acceptance > 0.3:
                overlaps = self.mc.overlaps
                if overlaps*1.0/N > 0.002 and overlaps > 2:
                    self.mc.external_potential.walls.clear()
                    for i,wall in enumerate(walls):
                        if type(wall) == hoomd.wall.Sphere:
                            wall_radius = wall.to_dict()['radius']
                            walls[i] = hoomd.wall.Sphere(radius=wall_radius/scale_ratio)
                            self.mc.external_potential.walls.append(walls[i])
                        elif type(wall) == hoomd.wall.Plane:
                            origin = np.array(wall.origin)/scale_ratio
                            normal = wall.normal
                            walls[i] = hoomd.wall.Plane(origin=origin, normal=normal)
                            self.mc.external_potential.walls.append(walls[i])
                    old_box = self.sim.state.get_snapshot().configuration.box
                    hoomd.update.BoxResize.update(self.sim.state, self.sim.state.box.scale(1/scale_ratio))
                    self.sim.state.set_box(old_box)
                    #print('\rtoo much overlaps detected, callback the system. N_overlaps = ' + str(overlaps), end='', flush=False)
                
            #relax
            overlaps = self.mc.overlaps
            while overlaps >= 1:
                self.sim.run(20)
                overlaps = self.mc.overlaps
            self.sim.run(run_step)
            acceptance = self.mc.translate_moves[0]/(self.mc.translate_moves[0] + self.mc.translate_moves[1])
            if N > 500:
                if self.sim.state.box.dimensions == 2:
                    N_near_boundary_ratio = (3*N)**(2/3)/N
                else:
                    N_near_boundary_ratio = (4*N)**(3/4)/N
                reduced_acceptance = acceptance/(1 - N_near_boundary_ratio/2)
            else:
                reduced_acceptance = acceptance
            if reduced_acceptance > 0.3:
                print('\racc: %.3f delta_d: %g, compress. '%(reduced_acceptance, delta_d), end='', flush=True)
            elif reduced_acceptance < 0.2:
                print('\racc: %.3f delta_d: %g, expand. '%(reduced_acceptance, delta_d), end='', flush=True)
        print(' ')