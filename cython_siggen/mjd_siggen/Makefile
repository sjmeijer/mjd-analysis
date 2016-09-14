# Makefile for signal generation from PPC detectors
#   - uses .c library codes by Karin Lagergren, heavily modified by David Radford
#
# [-lreadline option required for readline, addhistory...]

include $(MGDODIR)/buildTools/config.mk

CC = gcc
CPP = g++
CFLAGS = -O3 -Wall
RM = rm -f

# common files and headers
mk_signal_files = calc_signal.c cyl_point.c detector_geometry.c fields.c point.c read_config.c
mk_signal_headers = calc_signal.h cyl_point.h detector_geometry.h fields.h mjd_siggen.h point.h
OBJECTS = $(mk_signal_files:.c=.o)

All: $(SHLIB)


%: %.o $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJECTS) $(LIBFLAGS)

.cc.o:
	$(CC) $(CFLAGS) -c $<

%.o: %.C
	$(CC) $(CFLAGS) -c $<


$(SHLIB): $(OBJECTS)
	gcc -shared -o $(SHLIB) $(OBJECTS)  $(LIBFLAGS)

FORCE:

clean: 
	$(RM) *.o core* *[~%] *.trace *.dylib
	$(RM) stester mjd_fieldgen
