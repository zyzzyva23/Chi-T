import pandas as pd
import numpy as np
import re


def alignment_dict(align_excel_file):
    d_align = pd.read_excel(align_excel_file, header=1)
    d_align = d_align.iloc[:, [0, 2]]
    d_align = d_align.set_index('Sequence of the D loop as annotated in the column "tRNA14-21*" of the tRNA-DB-CE database')
    d_al_dict = d_align.to_dict(orient='index')
    d_al_dict = {key: v2 for key, v1 in d_al_dict.items() for k2, v2 in v1.items()}
    return d_al_dict


def d_loop_extend(x):
    if len(x) == 6:
        return f'{x[0:2]}--{x[2:]}'
    elif len(x) == 7:
        return f'{x[0:3]}-{x[3:]}'
    else:
        return x


def d_loop_align(x):
    try:
        g_start = re.search("...GG.", x).span()[0]
        g_start += 3
        if g_start == 3:
            if x[g_start + 2] != 'G':
                return f'{x[0:3]}-{x[3:6]}{x[-1]}'
            else:
                return f'{x[0:7]}{x[-1]}'
        else:
            return f'{x[0:4]}{x[g_start:g_start + 3]}{x[-1]}'
    except AttributeError:
        return np.nan


def clean_from_trnadb(df, d_al_dict):
    df_ = df.drop(['Start position', 'End position', '1st Intron start position', '1st Intron end position',
            '1st Intron seq.', '2nd Intron start position', '2nd Intron end position', '2st Intron seq.',
            'Decision from Dr. Muto', 'Decision from Dr. Inokuchi', 'Decision from Dr. Yamada', 'Comment of Dr. Muto',
            'Comment of Dr. Inokuchi', 'Comment of Dr. Yamada', 'Final decision'], axis=1)
    df_ = df_.rename(columns={'#Sequence ID': 'seq_id', 'Genome ID': 'gen_id',
                           'tRNA73-76 (*We used universal position of cloverleaf structure)':'tRNA73-76*'})
    df_.seq_id = df_.seq_id.apply(lambda x: x.replace('>', ''))
    df_ = df_.dropna(how='any', subset=df_.columns[7:-3])

    df_['tRNA14-21* aligned'] = df_['tRNA14-21*'].map(d_al_dict).fillna(df_['tRNA14-21*'])
    df_['tRNA14-21* aligned'] = df_['tRNA14-21* aligned'].apply(d_loop_extend)
    df_['tRNA14-21* aligned'] = df_['tRNA14-21* aligned'].apply(d_loop_align)
    df_['tRNA73-76*'] = df_['tRNA73-76*'].apply(lambda x: x[0] + 'CCA')
    df_['tRNA32-38*'] = df_['tRNA32-38*'].apply(lambda x: x[0:2] + 'CTA' + x[-2:])
    for col_name in df_.loc[:, 'tRNA1-7*':'tRNA73-76*'].columns:
        df_[col_name] = df_[col_name].str.upper()
    return df_


def make_big_clean(df_files, d_al_dict):
    output = pd.DataFrame()
    for file in df_files:
        df = pd.read_csv(file, sep='\t', encoding='unicode_escape')
        df = clean_from_trnadb(df, d_al_dict)
        output = pd.concat([output, df])
    output = output.reset_index()
    output = output.drop(['index'], axis=1)
    return output


def merge_parts(df_):
    # Remove tRNAs whose D-loops don't align
    # May try and align these later
    df = df_
    df = df.dropna(subset=['tRNA14-21* aligned'])
    df.loc[:, 'tRNA1-7_66-72*'] = df.apply(lambda x: x['tRNA1-7*'] + '_' + x['tRNA66-72*'], axis=1)
    df.loc[:, 'tRNA10-13_22-25*'] = df.apply(lambda x: x['tRNA10-13*'] + '_' + x['tRNA22-25*'], axis=1)
    df.loc[:, 'tRNA14-21_54-60*'] = df.apply(lambda x: x['tRNA14-21*'] + '_' + x['tRNA54-60*'], axis=1)
    df.loc[:, 'tRNA14-21_54-60* aligned'] = df.apply(lambda x: x['tRNA14-21* aligned'] + '_' + x['tRNA54-60*'], axis=1)
    df.loc[:, 'tRNA26_44-48*'] = df.apply(lambda x: x['tRNA26*'] + '_' + x['tRNA44-48*'], axis=1)
    df.loc[:, 'tRNA27-31_39-43*'] = df.apply(lambda x: x['tRNA27-31*'] + '_' + x['tRNA39-43*'], axis=1)
    df.loc[:, 'tRNA49-53_61-65*'] = df.apply(lambda x: x['tRNA49-53*'] + '_' + x['tRNA61-65*'], axis=1)
    return df



if __name__ == '__main__':
    print(d_loop_align('AGTCCGGCTGA'))
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("trnas", nargs='+', help='Files from tRNADB-CE for cleaning')
#     parser.add_argument("-a", "--alignment", help="File used to align D-loops", default='d_align.xlsx')
#     time_string = ''.join([str(time) for time in time.localtime()[:6]])
#     parser.add_argument('-o', '--output', help='Output file', default='merged_'+time_string+'.csv')
#     args = parser.parse_args()
#
#     d_al_dict = alignment_dict(args.alignment)
#     large_df = make_big_clean(args.trnas, d_al_dict)
#     merge_df = merge_parts(large_df)
#
#     merge_df.to_csv(args.output, index=False)



