#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
useful functions for navigating through the folders and files
@author: vpeterson
"""
import os
import numpy as np

def get_subfolders(subject_path, Verbose=False):
    """
    Get subfolder from a given path. Useful for getting all patients list.

    given an address, provides all the subfolder included in such path.

    Parameters
    ----------
    subject_path : string
        address to the subject folder
    Verbose : boolean, optional

    Returns
    -------
    subfolders : list
    """
    subfolders = []
    for entry in os.listdir(subject_path):
        if os.path.isdir(os.path.join(subject_path, entry)):
            subfolders.append(entry)
            if Verbose:
                print(entry)

    return subfolders


def get_data_files(address, subfolder, endswith='.EDF',
                   endswith_annot='SZ-VK.TXT', annot_files: bool = None, Verbose=True):
    """
    Get file from a given subject.

    given an address to a subject folder and a list of subfolders,
    provides a list of all files matching the endswith.

    To access to a particular vhdr_file please see 'read_BIDS_file'.
    To get info from vhdr_file please see 'get_sess_run_subject'

    Parameters
    ----------
    subject_path : string
    subfolder : list
    endswith ; string
    Verbose : boolean, optional

    Returns
    -------
    iles : list
        list of addrress to access to a particular vhdr_file.
    """
    files = []
    session_path = address + '/' + subfolder + '/iEEG'
    # only data whose annotation exist will be loaded
    if annot_files is None:
        annot_files = get_annot_files(address, subfolder, endswith_annot,
                                  Verbose=False)
    if endswith == '.EDF':
        prestr = len(endswith)+1  # for checking pre-file extention string
        for f_name in os.listdir(session_path):
            if (f_name.endswith(endswith) or f_name.endswith('.edf')) and f_name[-prestr].isnumeric():
                aux = f_name[:-len(endswith)]  # check annot
                if any(aux in s for s in annot_files):
                    files.append(session_path + '/' + f_name)
                    if Verbose:
                        print(f_name)
    else:
        for idx, f_name in enumerate(os.listdir(session_path)):
            if f_name.endswith(endswith[idx]):
                aux = f_name[:-(len(endswith[idx])+5)]  # check annot
                if any(aux in s for s in annot_files):
                    files.append(session_path + '/' + f_name)
                    if Verbose:
                        print(f_name)
    files.sort()
    return files

def annot_files_mgh(PATH_ANNOTS: str, endwith="_EOF_AI.txt",):
    return [f for f in os.listdir(PATH_ANNOTS) if f.endswith(endwith)]

def get_annot_files(address, subfolder, endswith='.TXT', Verbose=True):
    """
    Get annot file from a given subject.

    given an address to a subject folder and a list of subfolders,
    provides a list of all files matching the endswith.

    To access to a particular vhdr_file please see 'read_BIDS_file'.
    To get info from vhdr_file please see 'get_sess_run_subject'

    Parameters
    ----------
    subject_path : string
    subfolder : list
    endswith ; string
    Verbose : boolean, optional

    Returns
    -------
    iles : list
        list of addrress to access to a particular vhdr_file.
    """
    files = []
    session_path = address + '/' + subfolder + '/iEEG'
    annot1_eof = 'SZ-VK.TXT'
    anno2_eof = 'SZ-NZ.TXT'  # this files contain both VK and NZ annots
    txt_files_1 = [f for f in os.listdir(session_path) if f.endswith(annot1_eof)]
    txt_files_2 = [f for f in os.listdir(session_path) if f.endswith(anno2_eof)]

    # anno2_eof files are annotation files of first order priority
    aux_prefix = np.repeat(session_path + '/', len(txt_files_2))
    files = [x + y for x, y in zip(aux_prefix, txt_files_2)]
    # now check if annot1 exists but not annot2
    for f, f_name in enumerate(txt_files_1):
        aux = f_name[:-len(annot1_eof)]  # aux name file which annot
        if any(aux in s for s in txt_files_1) and not any(aux in s for s in txt_files_2):
            files.append(session_path + '/' + f_name)
        if Verbose:
            print(f_name)
    # check files are not empty files
    files = [ f for f in files if  os.stat(f).st_size != 0]
    files.sort()
    return files


def get_patient_PE(data_file):
    """
    Given a the data_file string return the subject, and PE.

    Parameters
    ----------
        data_file (string): [description]

    Returns
    -------
        hops, subject, PE
    """
    hosp = data_file[data_file.find('iEEG/') + len('iEEG/'):
                     data_file.find('iEEG/') + len('iEEG/') + 3]
    to_find = '/iEEG/' + hosp + '-'

    subject = data_file[data_file.find(to_find)+len(to_find):
                        data_file.find(to_find)+7+len(to_find)]

    str_PE = data_file[data_file.find('PE'):]
    PE = str_PE[2:-4]

    return hosp, subject, PE
