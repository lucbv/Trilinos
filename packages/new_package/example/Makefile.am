#@HEADER
# ************************************************************************
# 
#                     New_Package Example Package
#                 Copyright (2004) Sandia Corporation
# 
# Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
# license for use of this work by or on behalf of the U.S. Government.
# 
# This library is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation; either version 2.1 of the
# License, or (at your option) any later version.
#  
# This library is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#  
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301
# USA
# Questions? Contact Michael A. Heroux (maherou@sandia.gov) 
# 
# ************************************************************************
#@HEADER

include $(top_builddir)/Makefile.export.new_package

# The following line helps the test harness recover from build errors.

all-local: generate-sample-output.pl

generate-sample-output.pl : generate-sample-output.pl.stub
	cp generate-sample-output.pl.stub generate-sample-output.pl
	chmod a+x generate-sample-output.pl

EXEEXT = .exe

noinst_PROGRAMS = hello_test  RunValgrind

hello_test_SOURCES = $(srcdir)/cxx_main.cpp

RunValgrind_SOURCES = $(srcdir)/RunValgrind \
		$(srcdir)/In_Suppressions

#np# Specify locations of include files within Trilinos, but outside np
#np# Add packages that np is dependent on; remove epetra if np does not need it

AM_CPPFLAGS = $(NEW_PACKAGE_INCLUDES)

#np# - List libraries that np is dependent on.  The DEPENDENCIES primary is
#np# optional.
hello_test_DEPENDENCIES = \
	$(top_builddir)/../epetra/src/libepetra.a \
	$(top_builddir)/src/libnew_package.a 

#np# List libraries that np is dependent on.
hello_test_LDADD = $(NEW_PACKAGE_LIBS)

#
#  This is a crude but effective way to move files from the source tree to the build tree
#  We recommend renaming files because, although we recommend against the practice, some
#  users will configure and build in the same tree.
#
#  This requires at least one file to end in .exe (or whatever EXEEXT is set to).
#  The other files can have arbitrary names. (In this case Suppressions)
#
RunValgrind$(EXEEXT): $(srcdir)/RunValgrind
	cp  $(srcdir)/RunValgrind  RunValgrind$(EXEEXT)
	cp  $(srcdir)/In_Suppressions Suppressions

#np# Set "SUBDIRS= (list all subdirectories that are autotool'ed here)"
#np# List all subdirectories even if one or more are configured/built only
#np# conditionally.
#np# For packages whose examples are located in this directory, rather than
#np# in a subdirectory of this directory, use the Makefile.am located in
#np# example1/ rather than this file.
# SUBDIRS = example1
SUBDIRS = 
