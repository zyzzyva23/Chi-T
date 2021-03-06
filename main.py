import argparse
from ortho_classes import Isoacceptor2, id_dict, synth_clean
from parallel import rnafold_in_parallel
import pandas as pd
from itertools import zip_longest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from pathlib import Path
import time
import re


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("file", help='Clean tRNADB-CE file')
    parser.add_argument("synth_file", help='File containing synthetase information')
    parser.add_argument("synth_name", help='Name of synthetase matching to entry in synth_file',
                        nargs='+')
    parser.add_argument("amino_acid", help='Amino Acid specified for tRNA generation')
    parser.add_argument('-o', '--output_directory', help='Directory to store output files', default='')
    parser.add_argument('-ip', '--id_part_change', help='Identity parts that should be chimerified (except ID element)',
                        nargs='+')
    parser.add_argument("-cp", "--cluster_parts", type=int,
                        help='Number of parts for each part type to cluster', default=200)
    parser.add_argument("-cm", "--cluster_min", type=int, help="Minimum number of remaining parts for clustering.",
                        default=60)
    parser.add_argument('-s', '--subtle', help="If true, only parts <=2 mutations away from the native sequence are chosen",
                        action='store_true')
    parser.add_argument('-r', '--reference', help='If using subtle mode, this is the reference tRNA')
    parser.add_argument("-l", "--length_filt", help='Filter chimeras if 79 nts or longer (or specified value)',
                        nargs='?', const=79, type=int)
    parser.add_argument("-cf", "--cervettini_filt", nargs='+', help='Parameters for Cervettini Filtering in order '
                                                                    'start_stringency, minimum_stringency, target '
                                                                    'number of chimeras, and step size',
                        type=float, default=[0.5, 0.2, 2500000, 0.05])
    parser.add_argument('-a', '--anticodons', nargs='+', help='Anticodons to iterate through')
    parser.add_argument('-f', '--frequency', help='Average frequency across anticodons', default=0.3, type=float)
    parser.add_argument('-d', '--diversity', help='Average diversity across anticodons', default=5.0, type=float)
    parser.add_argument('-n', '--num_iterations', help='Number of times to iterate through Chi-T per synthetase',
                        default=1, type=int)
    parser.add_argument('-m', '--automatic', help='No user input required', action='store_true')
    parser.add_argument('-i', '--initial', help='If true, save initial chimeras to csv (only if needed for analysis)',
                        action='store_true')
    parser.add_argument('-p', '--pattern', help='Specify a csv file with synth name and regex string column')
    parser.add_argument('-ham', '--ham', help='Go ham', action='store_true')
    parser.add_argument('-t', '--num_tRNAs', help='Number of designs to output', default=4, type=int)
    args = parser.parse_args()

    if args.cervettini_filt and len(args.cervettini_filt) > 4:
        raise Exception("Too many filtering parameters!")
    else:
        args.cervettini_filt = [(i if i is not None else j)
                                for i, j in zip_longest(args.cervettini_filt, [0.5, 0, 2500000, 0.05])]
    cf_start, cf_min, cf_targ, cf_ss = args.cervettini_filt

    first_ac = args.anticodons[0]

    Path(f'{args.output_directory}/folding').mkdir(parents=True, exist_ok=True)
    Path(f'{args.output_directory}/plots').mkdir(parents=True, exist_ok=True)

    log_file = f'{args.output_directory}/log_file.txt'
    with open(log_file, 'w') as f:
        f.write('Chi-T\n' +
                str(time.time()) + '\n' +
                str(args) + '\n\n')

    df = pd.read_csv(args.file, usecols=['seq_id', 'Amino Acid', 'tRNA1-7*', 'tRNA8-9*', 'tRNA10-13*', 'tRNA14-21*',
                                         'tRNA22-25*', 'tRNA26*', 'tRNA27-31*', 'tRNA32-38*', 'tRNA39-43*',
                                         'tRNA44-48*', 'tRNA49-53*', 'tRNA54-60*', 'tRNA61-65*', 'tRNA66-72*',
                                         'tRNA73-76*', 'tRNA14-21* aligned', 'tRNA1-7_66-72*', 'tRNA10-13_22-25*',
                                         'tRNA14-21_54-60*', 'tRNA14-21_54-60* aligned', 'tRNA26_44-48*',
                                         'tRNA27-31_39-43*', 'tRNA49-53_61-65*'],
                     dtype={'Amino Acid': 'category', 'tRNA8-9*': 'category', 'tRNA26*': 'category',
                            'tRNA73-76*': 'category'},
                     engine='c')

    trna_pattern = re.compile(
        '^\\({5,8}\\.{1,3}\\({4}\\.{5,}\\){4}\\.*\\({4,9}\\.{7}\\){4,9}.*\\.\\({5}\\.{2,}\\){5}\\){5,8}\\.{3,}$')
    ile_pattern = re.compile(
        '^\\({5,8}\\.{1,3}\\({3}\\.{5,}\\){3}\\.*\\({4,9}\\.{7}\\){4,9}.*\\.\\({5}\\.{2,}\\){5}\\){5,8}\\.{3,}$')
    pat_dict = {synth_name: (trna_pattern if args.amino_acid != 'Ile' else ile_pattern) for synth_name in args.synth_name}
    if args.pattern:
        pat_df = pd.read_csv(args.pattern, header=None)
        pat_dict_ = dict(zip(pat_df.iloc[:, 0], pat_df.iloc[:, 1]))
        for k, v in pat_dict_.items():
            pat_dict_[k] = v.replace('\\\\', '\\')
        pat_dict.update(pat_dict_)

    df = df[df['Amino Acid'] == args.amino_acid]
    synth_df = synth_clean(args.synth_file)
    # synth_df_ = synth_df[synth_df.synth in args.synth_name]
    # if not all([seq_id in df.seq_id for seq_id in synth_df_.trna_id]):
    #     raise Exception("One or more tRNA ID in synth file not found in database. Check file.")
    total_iter = len(args.synth_name)*args.num_iterations
    iso = Isoacceptor2(synth_df, id_dict, args.amino_acid, df, ac=first_ac, id_part_change=args.id_part_change,
                       num_iter=total_iter, reference=args.reference)

    for j, synth_name in enumerate(args.synth_name):
        pattern = re.compile(pat_dict[synth_name])
        iso.iter_trnas = {}
        with open(log_file, 'a') as f:
            f.write(f'##################################################\n\nChi-T Run for {synth_name}\n')

        for i in range(j*args.num_iterations, (j+1)*args.num_iterations):
            with open(log_file, 'a') as f:
                f.write(f'\n***************************************\n\nIteration {i+1} of {total_iter}\n')

            iso.cluster_parts(args.cluster_parts, synth_name=synth_name, clust_id_parts=False, log_file=log_file,
                              iteration=i+1, automatic=args.automatic, clust_size_min=args.cluster_min,
                              subtle=args.subtle, ham=args.ham)
            iso.chimera(synth_name, length_filt=args.length_filt, log_file=log_file, iteration=i+1)
            iso.cervettini_filter(args.output_directory, synth_name, iteration=i+1, start_stringency=cf_start,
                                  min_stringency=cf_min, target=cf_targ, step_size=cf_ss, log_file=log_file)
            if args.initial:
                iso.store_trnas(f'{args.output_directory}/{synth_name}_initial_iter{i+1}.csv')

            print('Folding...')
            rnafold_in_parallel(iso, f'{args.output_directory}/folding/{synth_name}_para_iter{i+1}', first_ac)
            iso.fold_filter(first_ac,
                            f'{args.output_directory}/folding/{synth_name}_para_iter{i+1}_{first_ac}_complete_fold.out',
                            args.output_directory, synth_name, pattern, iteration=i+1, log_file=log_file)

            for ac in args.anticodons[1:]:
                if not iso.trnas:
                    break
                iso.change_ac([ac], synth_name)
                rnafold_in_parallel(iso, f'{args.output_directory}/folding/{synth_name}_para_iter{i+1}', ac)

                iso.fold_filter(ac, f'{args.output_directory}/folding/{synth_name}_para_iter{i+1}_{ac}_complete_fold.out',
                                args.output_directory, synth_name, pattern, iteration=i+1, log_file=log_file)

            if iso.trnas:
                iso.final_filter(freq_thresh=args.frequency, div_thresh=args.diversity, percentile_out=0,
                                 log_file=log_file)
                iso.store_trnas(f'{args.output_directory}/{synth_name}_finalfold_iter{i+1}.csv')

        iso.select(synth_name, args.output_directory, log_file=log_file, automatic=args.automatic, subtle=args.subtle)
        iso.store_trnas(f'{args.output_directory}/{synth_name}_selected.csv')

        cluster = len(iso.trnas) > 40
        iso.cluster_select(cluster=cluster, num_seqs=args.num_tRNAs, log_file=log_file)

        try:
            driver = webdriver.Chrome()
            driver.maximize_window()
            url = 'http://rna.tbi.univie.ac.at/cgi-bin/RNAWebSuite/RNAfold.cgi'
            final_seqs = [trna.seq[first_ac] for trna in iso.final_trnas.values()]

            for i in range(3):
                driver.execute_script("window.open('');")

            for seq, handle in zip(final_seqs, driver.window_handles):
                driver.switch_to.window(handle)
                driver.get(url)
                text_box = driver.find_element(By.ID, 'SCREEN')
                text_box.send_keys(seq)
                submit = driver.find_element(By.CLASS_NAME, 'proceed')
                submit.click()

            handles = driver.window_handles
            handles_to_remove = []
            while True:
                success = True
                handles = [handle for handle in handles if handle not in handles_to_remove]
                if len(handles) == 0:
                    break
                for handle in handles:
                    time.sleep(0.25)
                    try:
                        driver.switch_to.window(handle)
                        driver.find_element(By.XPATH, '//*[@id="contentmain"]/h3[1]')
                        driver.execute_script("window.scrollTo(0, 300)")
                        handles_to_remove.append(handle)
                    except NoSuchElementException:
                        pass
        except:
            pass

        if not args.automatic:
            while True:
                accept = input('Use these tRNAs? (y)es or (n)o\n> ')
                if accept.lower() == 'y':
                    iso.trnas = iso.final_trnas
                    iso.store_trnas(f'{args.output_directory}/{synth_name}_final_four.csv')
                    iso.iter_trnas = {}
                    break
                elif accept.lower() == 'n':
                    print("Well I dunno man what do you want from me I'm just a machine make your own tRNAs")
                    iso.iter_trnas = {}
                    break
                else:
                    print('Innapropriate value!')
                    continue
        else:
            iso.trnas = iso.final_trnas
            iso.store_trnas(f'{args.output_directory}/{synth_name}_final_four.csv')
            iso.iter_trnas = {}
