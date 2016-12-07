/*
 * to compile: 
 *  gcc -o st signal_tester.c read_config.c point.c cyl_point.c calc_signal.c\
 *    fields.c detector_geometry.c signal_calc_util.c -lm -lreadline
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <readline/readline.h>
#include <readline/history.h>
#include <ctype.h>
#include <string.h>

#include "mjd_siggen.h"
#include "calc_signal.h"
#include "cyl_point.h"
#include "detector_geometry.h"
#include "fields.h"


int main(int argc, char **argv) {

  MJD_Siggen_Setup fSiggenData;

  read_config("conf/P42574A_grad0.05_pcrad2.50_pclen1.65.conf", &fSiggenData);
  field_setup(&fSiggenData);
  
  fSiggenData.velocity_type = 1;
  
  fSiggenData.dpath_e = (point *) malloc(fSiggenData.time_steps_calc*sizeof(point));
  fSiggenData.dpath_h = (point *) malloc(fSiggenData.time_steps_calc*sizeof(point));
  
  fSiggenData.v_params = (velocity_params *) malloc(sizeof(velocity_params));
  fSiggenData.v_params->h_100_mu0 =66333.;
  fSiggenData.v_params->h_100_beta = 0.744;
  fSiggenData.v_params->h_100_e0 = 181;
  fSiggenData.v_params->h_111_mu0 = 107270;
  fSiggenData.v_params->h_111_beta = 0.580;
  fSiggenData.v_params->h_111_e0 = 100;
  
  static float *signal;
  signal = (float *) malloc(fSiggenData.time_steps_calc*sizeof(*signal));
  
  point pt;
  pt.x = 15;
  pt.y = 0;
  pt.z = 15;
  
  make_signal(pt, signal, 1, &fSiggenData);
  
  

  printf("velocity type: %d\n", fSiggenData.velocity_type);
  return 0;
}