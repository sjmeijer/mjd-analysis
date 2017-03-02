#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demonstration of DNest4 in Python using the "StraightLine" example
"""
import sys, os, shutil
import numpy as np
import dnest4

from helpers import Waveform
from dns_model import Model

from pysiggen import Detector
from mpi4py import MPI

comm = MPI.COMM_WORLD   # get MPI communicator object
size = comm.size        # total number of processes
rank = comm.rank        # rank of this process
status = MPI.Status()   # get MPI status object

def main():
    fm = MPIFitManager()

    if rank == 0:
        model = Model(fm)
        fm.set_indices( model.get_indices() )
        fm.wait_for_all_workers_to_report()

        fm.fit(model, numLevels=100)

        fm.close_all_workers()
    else:
        fm.wait_and_process()

class MPIFitManager():
    def __init__(self,):
        self.max_sample_idx = 200
        self.fallPercentage = 0.97
        # self.doInitPlot = 1

        self.colors = ["red" ,"blue", "green", "purple", "orange", "cyan", "magenta", "goldenrod", "brown", "deeppink", "lightsteelblue", "maroon", "violet", "lawngreen", "grey", "chocolate" ]
        self.tags = self.enum('READY', 'EXIT', 'HOLE', 'ELECTRON', 'PROCESS', 'DONE_HOLE','DONE_ELECTRON', 'DONE_PROCESS')


        self.setup_waveforms("P42574A_24_spread.npz")
        self.setup_detector("P42574A_fields_impgrad_0.00000-0.00100.npz")

        self.charge_wf_task_len = 10
        self.extra_process_params = 13
        self.num_workers = size - 1

    def setup_waveforms(self, wfFileName):
        # doInitPlot = 0#self.doInitPlot

        if os.path.isfile(wfFileName):
            data = np.load(wfFileName)
            wfs = data['wfs']

            wfs = wfs[12:16]
            # wfs = wfs[wfidxs]

            self.wfs = wfs
            self.num_waveforms = wfs.size
        else:
          print "Saved waveform file %s not available" % wfFileName
          exit(0)

        wfLengths = np.empty(wfs.size)
        wfMaxes = np.empty(wfs.size)
        baselineLengths = np.empty(wfs.size)

        for (wf_idx,wf) in enumerate(wfs):
          wf.WindowWaveformAroundMax(fallPercentage=self.fallPercentage, rmsMult=2, earlySamples=self.max_sample_idx)
          baselineLengths[wf_idx] = wf.t0Guess

          if rank == 0:
              print "wf %d length %d (entry %d from run %d)" % (wf_idx, wf.wfLength, wf.entry_number, wf.runNumber)
          wfLengths[wf_idx] = wf.wfLength
          wfMaxes[wf_idx] = np.argmax(wf.windowedWf)

        #   if doInitPlot:
        #       if len(self.colors) < wfs.size:
        #           color = "red"
        #       else: color = self.colors[wf_idx]
        #       plt.plot(wf.windowedWf, color=color)
        #
        # if doInitPlot:
        #     plt.show()
        #     exit()

        self.baseline_origin_idx = np.amin(baselineLengths) - 30
        if self.baseline_origin_idx < 0:
            print "not enough baseline!!"
            exit(0)

        self.siggen_wf_length = (self.max_sample_idx - np.amin(baselineLengths) + 10)*10
        self.output_wf_length = np.amax(wfLengths) + 1

        if rank == 0:
            print "siggen_wf_length will be %d, output wf length will be %d" % (self.siggen_wf_length, self.output_wf_length)


    def setup_detector(self, fieldFileName):
        timeStepSize = 1 #ns
        detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
        det =  Detector(detName, timeStep=timeStepSize, numSteps=self.siggen_wf_length, maxWfOutputLength =self.output_wf_length, t0_padding=100 )
        det.LoadFieldsGrad(fieldFileName)

        self.detector = det

    def calc_likelihood(self, params):
        num_waveforms = self.num_waveforms
        detector = self.detector
        tags = self.tags
        calc_length = detector.calc_length

        (tf_first_idx, velo_first_idx, trap_idx, grad_idx, num_det_params) = self.indices
        tf_b, tf_c, tf_d, rc1, rc2, rcfrac = params[tf_first_idx:tf_first_idx+6]
        h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0 = params[velo_first_idx:velo_first_idx+6]
        charge_trapping = params[trap_idx]
        grad = np.int(params[grad_idx])

        r_arr, phi_arr, z_arr, scale_arr, t0_arr, smooth_arr, m_arr, b_arr = params[num_det_params:].reshape((8, num_waveforms))

        # b, c, dc, rc1, rc2, rcfrac = 1, -0.814072377576, 0.82162729751, 72.6, 1, 1
        # h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0 = 66333., 0.744, 181., 107270., 0.580, 100.
        # charge_trapping = 200
        # grad_idx = 0
        #
        # r_arr   = np.linspace(5, 30, num_waveforms)#np.ones(num_waveforms, dtype="f4") * 15
        # phi_arr = np.ones(num_waveforms, dtype="f4") * np.pi/8
        # z_arr =   np.ones(num_waveforms, dtype="f4") * 15
        # scale_arr=np.ones(num_waveforms, dtype="f4") * np.amax(self.wfs[0].windowedWf)
        # t0_arr   = np.ones(num_waveforms, dtype="f4") *self.max_sample_idx
        # smooth_arr=np.ones(num_waveforms, dtype="f4") * 15
        # m_arr     =np.zeros(num_waveforms, dtype="f4")
        # b_arr     =np.zeros(num_waveforms, dtype="f4")

        # Master process executes code below
        charge_tasks = np.empty((num_waveforms, self.charge_wf_task_len), dtype="f4")
        empty_charge_task = np.empty(self.charge_wf_task_len, dtype='f4')

        for i in range(num_waveforms):
            charge_tasks[i,:] = [r_arr[i], phi_arr[i], z_arr[i], h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0, grad_idx]

        task_index_hole = 0
        task_index_electron = 0
        task_index_process = 0

        num_workers = size - 1
        closed_workers = 0
        holes_and_electrons_received = 0

        workerToTaskMap = np.empty(size)

        #holders for storing results
        hole_wfs     = np.empty((num_waveforms, calc_length), dtype='f4')
        electron_wfs = np.empty((num_waveforms, calc_length), dtype='f4')
        wf_likes = np.empty(num_waveforms)

        #temporary holders to receive results into
        charge_wf = np.empty((calc_length), dtype='f4')
        wf_like = np.empty(1)

        #temporary holder to shove tasks into
        empty_process_task = np.empty(2*calc_length + self.extra_process_params, dtype='f4')

        #assume you start the function with all workers available.  Busy em all up
        ready_workers = range(1, num_workers+1)
        job_number = 0
        wfs_finished = 0
        worker_max_tasks = num_workers
        if num_workers > num_waveforms*2:
            worker_max_tasks = num_waveforms*2

        # print("Master starting with %d workers" % num_workers)
        while wfs_finished < num_waveforms:
            #Process whatever the last communication from the workers is
            #(only start looking after you've sent out work already)
            if job_number >= worker_max_tasks:
                comm.Probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                tag = status.Get_tag()
                source = status.Get_source()

                if tag == tags.READY:
                    comm.Recv(charge_wf, source=source, tag=tag, status=status)
                    ready_workers.append(source)
                    print "worker %d is ready & added to the stack" % source
                elif tag == tags.DONE_HOLE:
                    comm.Recv(charge_wf, source=source, tag=tag, status=status)
                    task_index_recd = workerToTaskMap[source]
                    hole_wfs[task_index_recd,:] = charge_wf
                    holes_and_electrons_received +=1
                    print("Got hole data for task %d from worker %d" % (task_index_recd, source))
                elif tag == tags.DONE_ELECTRON:
                    comm.Recv(charge_wf, source=source, tag=tag, status=status)
                    task_index_recd = workerToTaskMap[source]
                    electron_wfs[task_index_recd,:] = charge_wf
                    holes_and_electrons_received +=1
                    print("Got electron data for task %d from worker %d" % (task_index_recd, source))
                elif tag == tags.DONE_PROCESS:
                    comm.Recv(wf_like, source=source, tag=tag, status=status)
                    task_index_recd = workerToTaskMap[source]
                    if wf_like[0] == 999.:
                        wf_likes[task_index_recd] = -np.inf
                    else:
                        wf_likes[task_index_recd] = wf_like[0]
                    print("    Recv wf %d with ln like %f from worker %d" % (task_index_recd,wf_likes[task_index_recd], source))
                    wfs_finished +=1

            #wait around for a worker to ready itself
            if len(ready_workers) == 0: continue
            current_worker = ready_workers[0]

            if task_index_hole < num_waveforms:
                comm.Send([charge_tasks[task_index_hole,:], MPI.FLOAT], dest=current_worker, tag=tags.HOLE)
                print("Sending hole task %d to worker %d" % (task_index_hole, current_worker))
                workerToTaskMap[current_worker] = task_index_hole
                task_index_hole += 1
                job_number +=1
                ready_workers.pop(0)

            elif task_index_electron < num_waveforms:
                comm.Send([charge_tasks[task_index_electron,:], MPI.FLOAT], dest=current_worker, tag=tags.ELECTRON)
                print("Sending electron task %d to worker %d" % (task_index_electron, current_worker))
                workerToTaskMap[current_worker] = task_index_electron
                task_index_electron += 1
                job_number +=1
                ready_workers.pop(0)

            elif task_index_process < num_waveforms:
                if holes_and_electrons_received < num_waveforms * 2:
                    #tell it to wait?
                    continue

                empty_process_task[0:calc_length] =  hole_wfs[task_index_process,:]
                empty_process_task[calc_length:2*calc_length] =  electron_wfs[task_index_process,:]

                empty_process_task[2*calc_length:] = task_index_process, scale_arr[task_index_process], t0_arr[task_index_process], smooth_arr[task_index_process], m_arr[task_index_process], b_arr[task_index_process], tf_b, tf_c, tf_d, rc1, rc2, rcfrac, charge_trapping

                comm.Send([empty_process_task, MPI.FLOAT], dest=current_worker, tag=tags.PROCESS)
                print("Sending process task %d to worker %d" % (task_index_process, current_worker))
                workerToTaskMap[current_worker] = task_index_process
                task_index_process += 1
                ready_workers.pop(0)


        for idx in range(num_waveforms):
          print "    wf %d has ln like %f " % (idx, wf_likes[idx])
        #     print "Waveform %d: hole sum %f, electron sum %f, ln like %f" % (idx, np.sum(hole_wfs[idx,:]), np.sum(electron_wfs[idx,:]), wf_likes[idx] )
        print "  total lnlike is %f" % np.sum(wf_likes)
        print "remaining ready workers: " + str(ready_workers)
        print "\n-------------------------------------------------\n"

        return np.sum(wf_likes)

    def wait_for_all_workers_to_report(self):
        empty_charge_wf = np.empty(self.siggen_wf_length, dtype='f4')
        num_ready_workers = 0
        while num_ready_workers < self.num_workers:
            #Make sure they all tell us they're ready
            comm.Recv(empty_charge_wf, tag=self.tags.READY)
            num_ready_workers += 1

    def close_all_workers(self):
        empty_charge_task = np.empty(self.charge_wf_task_len, dtype='f4')
        empty_charge_wf = np.empty(self.siggen_wf_length, dtype='f4')

        workers = range(1, self.num_workers+1)
        for worker in workers:
            #Tell em to close
            comm.Send([empty_charge_task, MPI.FLOAT], dest=worker, tag=self.tags.EXIT)

        closed_workers = 0
        while closed_workers < self.num_workers:
            #Wait until they're told us all they're closed
            comm.Recv(empty_charge_wf, source=MPI.ANY_SOURCE, tag=self.tags.EXIT, status=status)
            source = status.Get_source()
            closed_workers += 1
            print("Worker %d exited. (%d of %d)" % (source, closed_workers, self.num_workers))
        print("Master finishing")

    def wait_and_process(self):
        detector = self.detector
        calc_length = self.siggen_wf_length
        tags = self.tags

        # Worker processes execute code below
        name = MPI.Get_processor_name()
        # print("I am a worker with rank %d on %s." % (rank, name))
        task = np.empty(self.charge_wf_task_len, dtype='f4')
        empty_wf = np.empty(calc_length, dtype='f4')
        process_task = np.empty(2*calc_length + self.extra_process_params, dtype='f4')

        empty_double = np.empty(1)

        while True:
            #figure out what operation I'm meant to do
            comm.Send([empty_wf,MPI.FLOAT], dest=0, tag=tags.READY)
            comm.Probe(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()

            if tag == tags.HOLE:
                comm.Recv([task,MPI.FLOAT], source=0, tag=tags.HOLE, status=status)
                wf = self.do_charge_calc_task(task, 1)
                comm.Send([wf, MPI.FLOAT], dest=0, tag=tags.DONE_HOLE)

            elif tag == tags.ELECTRON:
                comm.Recv([task,MPI.FLOAT], source=0, tag=tags.ELECTRON, status=status)
                wf = self.do_charge_calc_task(task, -1)
                comm.Send([wf, MPI.FLOAT], dest=0, tag=tags.DONE_ELECTRON)

            elif tag == tags.PROCESS:
                comm.Recv([process_task,MPI.FLOAT], source=0, tag=tags.PROCESS, status=status)
                ln_like = self.do_wf_process_task(process_task)
                empty_double[0] = ln_like
                comm.Send([empty_double, MPI.DOUBLE], dest=0, tag=tags.DONE_PROCESS)

            elif tag == tags.EXIT:
                break

        comm.Send([empty_wf,MPI.FLOAT], dest=0, tag=tags.EXIT)

    def do_charge_calc_task(self, charge_calc_task, charge):
        r, phi, z                          = charge_calc_task[:3]
        h_100_mu0, h_100_lnbeta, h_100_emu    = charge_calc_task[3:6]
        h_111_mu0, h_111_lnbeta, h_111_emu    = charge_calc_task[6:9]
        grad_idx                           = np.int(charge_calc_task[9])

        h_100_beta = 1./np.exp(h_100_lnbeta)
        h_111_beta = 1./np.exp(h_111_lnbeta)
        h_100_e0 = h_100_emu / h_100_mu0
        h_111_e0 = h_111_emu / h_111_mu0

        self.detector.SetFieldsGradIdx(grad_idx)
        self.detector.siggenInst.set_hole_params(h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0)
        wf = self.detector.MakeRawSiggenWaveform(r, phi, z, charge)

        return wf

    def do_wf_process_task(self, wf_proc_task):
        param_start_idx = 2*self.siggen_wf_length
        hole_wf        = wf_proc_task[0:self.siggen_wf_length]
        electron_wf    = wf_proc_task[self.siggen_wf_length:param_start_idx]

        wf_idx, energy, t0, smooth, m, b = wf_proc_task[param_start_idx:param_start_idx+6]
        d, tf_phi, tf_omega, rc1, rc2, rcfrac, charge_trapping = wf_proc_task[-7:]

        c = -d * np.cos(tf_omega)
        b_ov_a = c - np.tan(tf_phi) * np.sqrt(d**2-c**2)
        a = 1./(1+b_ov_a)
        tf_b = a * b_ov_a

        self.detector.SetTransferFunction(tf_b, c, d, rc1, rc2, rcfrac)
        self.detector.trapping_rc = charge_trapping

        wf = self.wfs[wf_idx]
        data = wf.windowedWf
        model_err = wf.baselineRMS
        data_len = len(data)

        model = self.detector.TurnChargesIntoSignal(hole_wf, electron_wf, energy, t0, data_len, h_smoothing=smooth, trapType="fullSignal", alignPoint="max", doMaxInterp=False)

        if model is None:
            return 999.
        if np.any(np.isnan(model)):
            return 999.

        start_idx = -self.baseline_origin_idx
        end_idx = data_len - self.baseline_origin_idx - 1
        baseline_trend = np.linspace(m*start_idx+b, m*end_idx+b, data_len)
        model += baseline_trend

        inv_sigma2 = 1.0/(model_err**2)
        ln_like = -0.5*(np.sum((data-model)**2*inv_sigma2 - np.log(inv_sigma2)))
        print "   source: wf %d has ln like %f" % (wf_idx, ln_like)
        return ln_like


    def enum(self, *sequential, **named):
        """Handy way to fake an enumerated type in Python
        http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
        """
        enums = dict(zip(sequential, range(len(sequential))), **named)
        return type('Enum', (), enums)

    def set_indices(self, indices):
        self.indices = indices

    def fit(self, model, numLevels, directory=""):

      sampler = dnest4.DNest4Sampler(model,
                                     backend=dnest4.backends.CSVBackend(basedir ="./" + directory,
                                                                        sep=" "))

      # Set up the sampler. The first argument is max_num_levels
      gen = sampler.sample(max_num_levels=numLevels, num_steps=200000, new_level_interval=10000,
                            num_per_step=1000, thread_steps=100,
                            num_particles=5, lam=10, beta=100, seed=1234)

      # Do the sampling (one iteration here = one particle save)
      for i, sample in enumerate(gen):
          print("# Saved {k} particles.".format(k=(i+1)))

      # Run the postprocessing
      # dnest4.postprocess()


if __name__=="__main__":
    main()
