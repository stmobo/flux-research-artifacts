# 2014 Cab Supercomputer Job Scheduling Traces

This dataset is historical metadata about jobs run on the now decomissioned,
unclassified, open LC system known as Cab.  The job metadata is the number of
Cab nodes requested, the requested runtime of the job, the actual runtime of the
job, the time the job was submitted, the time the job started running, the time
the job completed, the status of the job (did it complete successfully or exit
due to a failure), the exit code of the job (if it failed), and the queue that
the job was submitted to (i.e., debug or batch).

Two additional columns of metadata include the job ID and the user ID.  Both of
these have been anonymized.  The job ID has been replaced with numbers ranging
from 1 to N where N is the number of jobs in the file (i.e., 1, 2, 3, 4, 5 …).
The usernames have been replaced with numbers ranging from 1 to M (where M is
the number of unique users on the system), such that it is still possible to
determine which jobs were submitted by the same user, but the particular user is
completely anonymized.  As a concrete example, the username “sherbein” might
correspond to the number “45”.  All jobs submitted by “sherbein” will have their
username replaced with “45”.  The numbers used have no connection or correlation
with the Linux UIDs on the system.

## Example of Data Format

```csv
JobID,User,Partition,NNodes,NCPUS,Timelimit,State,Submit,Start,Elapsed,End,ExitCode
1,1,pbatch,6,96,12:00:00,COMPLETED,2014-08-01T02:22:58,2014-08-01T02:22:58,01:35:26,2014-08-01T03:58:24,0:0
2,1,pbatch,6,96,12:00:00,COMPLETED,2014-08-01T02:22:59,2014-08-01T02:22:59,01:32:55,2014-08-01T03:55:54,0:0
3,1,pbatch,4,64,12:00:00,COMPLETED,2014-08-01T13:33:50,2014-08-01T13:33:50,02:44:57,2014-08-01T16:18:47,0:0
4,1,pbatch,4,64,12:00:00,COMPLETED,2014-08-01T13:33:51,2014-08-01T13:33:51,01:33:13,2014-08-01T15:07:04,0:0
5,1,pbatch,4,64,12:00:00,COMPLETED,2014-08-01T13:33:51,2014-08-01T13:33:51,01:43:56,2014-08-01T15:17:47,0:0
6,1,pbatch,4,64,12:00:00,COMPLETED,2014-08-01T13:34:20,2014-08-01T13:34:20,01:21:25,2014-08-01T14:55:45,0:0
7,1,pbatch,4,64,12:00:00,COMPLETED,2014-08-01T13:36:41,2014-08-01T13:36:41,00:22:07,2014-08-01T13:58:48,0:0
8,1,pbatch,4,64,12:00:00,COMPLETED,2014-08-01T13:52:16,2014-08-01T13:52:17,01:09:35,2014-08-01T15:01:52,0:0
9,1,pbatch,4,64,12:00:00,COMPLETED,2014-08-01T13:55:24,2014-08-01T13:55:24,01:15:06,2014-08-01T15:10:30,0:0
```

# Citation Information

If you use these job traces in a publication, please cite them using the following information:

Flux Framwork Team (2020). 2014 Cab Supercomputer Job Scheduling Traces. Lawrence Livermore National Laboratory. https://doi.org/10.5281/zenodo.3908771

```bibtex
@misc{cab-traces,
    author = {Flux Framework Team},
    title = {2014 Cab Supercomputer Job Scheduling Traces},
    doi = {10.5281/zenodo.3908771},
    howpublished = {\url{https://doi.org/10.5281/zenodo.3908771}},
    year = 2020,
    organization = {Lawrence Livermore National Laboratory},
}
```

# Disclaimer

Released under LLNL-MI-811683. Prepared by LLNL under Contract DE-AC52-07NA27344.
