import HpstrConf
import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
        '--sample', choices=['data','sim_bkgd','signal'],
        help='Signal which type of sample this is', required=True
)
parser.add_argument('-n',type=int,help='maximum event number to process',default=-1)
parser.add_argument('input_file',type=str,help='input ROOT file to process')
parser.add_argument('output_file',type=str,help='output ROOT file to write to')
args = parser.parse_args()

p = HpstrConf.Process()

p.run_mode = 1
p.skip_events = 0
p.max_events = args.n

p.add_library("libprocessors")

def file_in_hpstr(relpath):
    fullpath = os.environ['HPSTR_BASE']+"/"+relpath
    if not os.path.isfile(fullpath):
        raise ValueError(f'{fullpath} does not exist!')
    return fullpath

is_data = (args.sample == 'data')

preselect = HpstrConf.Processor('preselect', 'PreselectAndCategorize2021')
preselect.parameters["isData"] = 1 if is_data else 0
preselect.parameters["isSignal"] = 1 if (args.sample == 'signal') else 0
preselect.parameters["beamPosCfg"] = "" # has already been done for these samples
preselect.parameters["pSmearingFile"] = ""
#preselect.parameters["pSmearingFile"] = "" if is_data else file_in_hpstr(
#        "utils/data/fee_smearing/smearingFile_2016_all_20240620.root"
#)
preselect.parameters["v0ProjectionFitsCfg"] = ""
#preselect.parameters["v0ProjectionFitsCfg"] = file_in_hpstr(
#        'analysis/data/v0_projection_2016_config.json'
#        if is_data else
#        'analysis/data/v0_projection_2016_mc_config.json'
#)
preselect.parameters['trackBiasCfg'] = file_in_hpstr(
        'analysis/data/track_bias_corrections_data_2021.json'
        if is_data else
        'analysis/data/track_bias_corrections_tritrig_2021.json'
)

preselect.parameters['calTimeOffset'] = 37.4 if is_data else 0.#24.

p.sequence = [preselect]

p.input_files = [args.input_file]
p.output_files = [args.output_file]

p.printProcess()
