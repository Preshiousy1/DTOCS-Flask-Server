
from distutils.log import debug
from flask import Flask, session, request, render_template
import os
from werkzeug.utils import secure_filename

# DTOC IMPORTS
import networkx as nx
import pandas as pd
import datetime
import numpy as np
import math
from pathlib import Path

# Determine the current file path
file_path1 = Path(".").absolute()


# Create file path to create a folder and save results to said folder
UPLOAD_FOLDER = file_path1/"temp"
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Members API Route


@app.route("/")
def index():
    return render_template("index.html", token="DTOCSFlask")


@app.route("/upload_csv", methods=['POST'])
def upload_csv():
    target = os.path.join(UPLOAD_FOLDER, 'uploads')
    if not os.path.isdir(target):
        os.mkdir(target)
    file = request.files['file']
    filename = secure_filename(file.filename)
    destination = "/".join([target, filename])
    file.save(destination)
    # session['uploadFilePath'] = destination
    response = {"status": "success", "destination": destination}
    return response


@app.route("/send_data", methods=['POST'])
def process_data():

    # ###### Quality of life improvements
    # Determine save path. Option to further parametrize when solving multiple instances. For example, define parameters for scheduling horizon, discount rate, and scheduling fidelity. Vary the parameter values and create the files, with the option to call OMP from the script itself and store each instances' files in a separate folder.

    # In[2]:

    # Determine the current file path
    file_path = Path(".").absolute()

    print(file_path)

    # Create file path to create a folder and save results to said folder
    save_path = file_path/"temp/outputs"
    save_path.mkdir(exist_ok=True)

    print(save_path)

    # Set parameters for the Deswik file, schedule start date, scheduling horizon, schedule end date, and scheduling fidelity.

    # In[3]:

    # Data file
    deswik_file = request.json['destination']
    #__DPM_file__ = "DPM_params.csv""./temp/uploads/NMN_Reserve_Export.csv"

    # Start date for the schedule, i.e., when to start scheduling
    sched_start = "01-01-2022"
    sched_sm, sched_sd, sched_sy = sched_start.split('-')
    schedule_s = datetime.datetime(int(sched_sy), int(sched_sm), int(sched_sd))

    # Scheduling horizon in days (2 years for this project)
    schedule_horizon = 1824

    # Calculate Schedule End
    schedule_e = schedule_s + \
        datetime.timedelta(days=schedule_horizon) - \
        datetime.timedelta(seconds=1)

    # Calculate activity start date cut-off
    activity_cutoff = schedule_s + \
        datetime.timedelta(days=schedule_horizon, hours=1)

    # Number of shifts per day
    shift = 1

    # Number of quarter-shifts per shift
    quarter = 1

    # Populate a list of columns of interest from the Deswik mine plan. The following columns have to be compulsorily

    # In[4]:

    # Define a tuple of relevant columns
    # Column names refer to the actual Deswik column names
    columns = (
        'ID',
        'Duration hours',
        'GANANCIA TOTAL ($)',
        'Start',
        'DESARROLLO (m) BASAL',
        'DESARROLLO (m) ESTERIL',
        'DESARROLLO (m) RAMPA',
        'LVL_BACKFILL',
        'LVL_DEVELOPMENT (m)',
        'LVL_EXTRACCION_MINERAL (Tn)',
        'DE METROS PERFORADOS (m)',
        'Predecessor details',
        'Finish',
        'Rate',
        'NIVEL',
        'SOT_ACT_TYPE',
        'ID_LABOR',
        'Driving property',
        'RESBIN',
        'Resources'

    )

    # In[5]:

    # Read in the data file and clean up
    data = pd.read_csv(file_path/deswik_file,
                       usecols=columns,
                       index_col='ID',
                       dtype={'ID': str,
                              'Duration hours': float,
                              'GANANCIA TOTAL ($)': float,
                              'Start': str,
                              'DESARROLLO (m) ESTERIL': float,
                              'DESARROLLO (m) RAMPA': float,
                              'DESARROLLO (m) BASAL': float,
                              'LVL_BACKFILL': float,
                              'LVL_DEVELOPMENT (m)': float,
                              'LVL_EXTRACCION_MINERAL (Tn)': float,
                              'DE METROS PERFORADOS (m)': float,
                              'Predecessor details': str,
                              'Finish': str,
                              'Rate': str,
                              'NIVEL': str,
                              'SOT_ACT_TYPE': str,
                              'ID_LABOR': str,
                              'Driving property': str,
                              'RESBIN': str,
                              'Resources': str
                              },
                       parse_dates=['Start', 'Finish'],
                       na_filter=False
                       )

    print(f"Completed File Ingest {file_path/deswik_file}")

    # In[6]:

    # Inspect data
    data = pd.DataFrame(data)
    # print(data.head())

    print('Number of colums in Dataframe : ', len(data.columns))
    #print('Number of rows (activities) in Dataframe : ', len(data.index))

    # In[7]:

    # Rename columns
    data = data.rename(columns={
        'Duration hours': "duration",
        'GANANCIA TOTAL ($)': "profit",
        'DESARROLLO (m) BASAL': "prim_dev",
        'DESARROLLO (m) ESTERIL': "waste_dev",
        'DESARROLLO (m) RAMPA': "ramp_dev",
        'LVL_BACKFILL': "lvl_backfill",
        'LVL_DEVELOPMENT (m)': "lvl_dev",
        'LVL_EXTRACCION_MINERAL (Tn)': "lvl_ore_t",
        'DE METROS PERFORADOS (m)': "production_drill",
        'Predecessor details': "preds",
        'SOT_ACT_TYPE': "act_type_brd",
        'ID_LABOR': "act_type",
        'Start': "start",
        'Finish': "finish",
        'NIVEL': "Level",
        'Rate': "rate",
        'Driving property': "d_prop",
        'RESBIN': "min_reserve",
        'Resources': "resources"
    })

    print(data.head())

    # In[8]:

    # Trim data to include only those to be scheduled
    #data_trimmed = data.loc[(data.start >= schedule_s) & (data.start < activity_cutoff), :]
    data_ongoing_tasks = data.loc[(data.start < schedule_s) & (
        data.finish > schedule_s), :]
    data = data.loc[data['min_reserve'] == 'RESERVAS']
    data = data.loc[(data.start >= '2022-01-01'), :]

    #print("Activities filtered between " + str(schedule_s) + " and " + str(activity_cutoff))
    #print("Number of activities from short-term schedule: %d" % (data_trimmed.index.size))
    print("Number of ongoing activities (assumed scheduled): %d" %
          (data_ongoing_tasks.index.size))
    print('Number of rows (activities) in Dataframe : ', len(data.index))
    print(data.shape)
    data.head()

    #print("Unique activity types and their distribution: ", data_trimmed.act_type.value_counts())

    # In[9]:

    activities = data.loc[:, ('Level', 'start', 'finish')]
    levels = tuple(activities.Level.unique())

    active_levels = pd.DataFrame(0, index=pd.date_range(
        schedule_s, periods=1824).to_list(), columns=levels)

    for index, row in activities.iterrows():
        add = pd.date_range(start=row.start.date(),
                            end=row.finish.date(),
                            freq='1D')
        active_levels.loc[add, row.Level] += 1

    result = pd.DataFrame(np.select([active_levels > 0], [
                          1], 0)).sum(axis=1).to_frame()
    result[0].value_counts()

    # In[10]:

    # Define activity sets (Group the activity types into unique categories)
    # C_LH: Stope excavation; G_GB: primary development; V_CHV, V_RBV 3.1m, P_PS: vertical development;
    set_dev_advance = ("G_GB", "G_CX", "G_GT", "G_AC", "R_RP",
                       "E_EB", "E_EC", "E_EE", "E_ESE", "E_EV")
    set_vertical_advance = ("V_CHVT", "V_CHEMUG", "V_CHVTUG", "P_PS")
    set_backfill = ("C_LH")
    set_extraction = ("C_LH")
    set_prod_drill = ("C_LH")
    #set_pastfill    = ()

    # The sets above reflect activities that utilize particular equipment; only a limited number of actvities in a given set
    # can occur concurrently due to equipment availability and number

    # In[11]:

    # Split individual predecessors into components
    preds = data['preds']
    predecessor_details = preds.str.split(';', expand=True)
    for i in predecessor_details.columns:
        for index, row in predecessor_details.iterrows():
            if row[i]:
                row[i] = tuple(row[i].split(':'))
    # data_trimmed.drop(['Predecessor details'], axis=1, inplace=True)
    print("Predecessor Details processed.")
    predecessor_details.head(1)[0][0]

    # In[12]:

    # Create a Directed Graph
    G = nx.DiGraph()

    # Add nodes to DiGraph, divide into sub-nodes, assign activity attributes, and modify as required
    base_nodes = list()

    # In[13]:

    # Add nodes to DiGraph for the primary set of activities
    # Determine activity rate
    counter_TBS = 0
    for index, row in data.iterrows():
        counter_TBS += 1
        if row.rate.split('/')[1] == 'w':  # Activity rate
            row.rate = float(row.rate.split('/')[0].strip(row.d_prop)) / 7
        else:
            row.rate = float(row.rate.split('/')[0].strip(row.d_prop))

        # Add full activity node
        G.add_node(index,
                   act_type=row.act_type,
                   sot_type=row.act_type_brd,
                   resources=row.resources,
                   prim_dev=round(row.prim_dev, 2),
                   waste_dev=round(row.waste_dev, 2),
                   ramp_dev=round(row.ramp_dev, 2),
                   lvl_backfill=math.ceil(row.lvl_backfill),
                   lvl_dev=round(row.lvl_dev),
                   lvl_ore_t=math.ceil(row.lvl_ore_t),
                   lvl_proddrill_m=math.ceil(row.production_drill),
                   dev_advance=1 if row.act_type in set_dev_advance else 0,
                   vert_advance=1 if row.act_type in set_vertical_advance else 0,
                   backfill=1 if row.act_type in set_backfill else 0,
                   extraction=1 if row.act_type in set_extraction else 0,
                   prod_drill=1 if row.act_type in set_prod_drill else 0,
                   dur=math.ceil((row.duration / 24) * shift * quarter),
                   durh=row.duration,
                   # dpm=0.0,
                   start=row.start,
                   level=int(row.Level),
                   flag_complete=False,
                   es=0,
                   ef=math.ceil((row.duration / 24) * shift * quarter),
                   #ef=math.ceil((row.duration / 24) * shift * quarter),
                   obj=row.profit,
                   act_dur=(row.duration / 24) * shift * quarter,
                   flag_extra_act=True if row.start >= schedule_e else False
                   )
        base_nodes.append(index)
    print('Processed %d activities (to be scheduled) into the directed graph.' % counter_TBS)
    print("Nodes added to DiGraph.")

    # In[14]:

    # Add edges to DiGraph
    nodes_wo_preds = 0
    preds_not_handled = 0
    finished_prec = 0
    finishStart_count = 0
    startStart_count = 0
    percentOverlap_count = 0

    for index, row in predecessor_details.iterrows():
        if data.loc[index]['preds']:
            for i in predecessor_details.columns:
                if row[i]:
                    if G.has_node(row[i][0]):
                        if row[i][1] == "FinishStart":
                            G.add_edge(row[i][0], index,
                                       dur=float(row[i][2][:-2]) *
                                       shift * quarter,
                                       ptype="FinishStart")  # adding predecessor ID (i.e. row[i][0]) and duration (delay)
                            finishStart_count += 1
                        elif row[i][1] == "FinishFinish":
                            # adding predecessor ID (i.e. row[i][0]) and duration (delay)
                            G.add_edge(row[i][0], index, dur=0,
                                       ptype="FinishStart")
                        elif row[i][1] == "StartStart":
                            G.add_edge(row[i][0], index,
                                       dur=float(row[i][2][:-2]) *
                                       shift * quarter,
                                       ptype="StartStart")
                            startStart_count += 1
                        elif row[i][1] == "PercentOverlap":
                            G.add_edge(row[i][0], index,
                                       dur=-
                                       (G.nodes[row[i][0]]['dur'] * float(row[0]
                                        [2][:-2]) * .01) * shift * quarter,
                                       ptype="PercentOverlap")
                            percentOverlap_count += 1
                        else:
                            print(
                                "Encountered unknown precedence type " + row[i][1])
                            preds_not_handled += 1
                    else:
                        # Add a carryover precedence here
                        finished_prec += 1
                else:
                    break
        else:
            nodes_wo_preds += 1
            continue

    print('Nodes without predecessors: ' + str(nodes_wo_preds))
    print('Finished activities not present in Graph: ' + str(finished_prec))
    print('Precedences handled in Graph: FinishStart - %d, StartStart - %d, and PercentOverlap - %d' %
          (finishStart_count, startStart_count, percentOverlap_count))
    print('Unknown precedence type: ' + str(preds_not_handled))
    print("Finished constructing DiGraph G")

    # In[15]:

    # G.add_edge(row[i][0], index,
    #dur=float(row[i][2][:-2]) * shift * quarter,
    # ptype="StartStart")

    # In[16]:

    # Check if graph is acyclic
    if nx.is_directed_acyclic_graph(G):
        # Convert Start to time period
        print('DiGraph G is acyclic. Proceeding.')
        topo_sort = list(nx.topological_sort(G))
        for node in topo_sort:
            try:
                diff_starts = G.nodes[node]['start'] - schedule_s
                G.nodes[node]['tp'] = math.ceil(
                    (diff_starts.days + (diff_starts.seconds // 3600) / 24) * shift * quarter)
            except KeyError:
                print(
                    "Error! Not calculating the start time period correctly for: %s" % node)
        # Calculate and write the early starts for each activity
        with open(save_path/'EarlyStarts.txt', mode='w') as w:
            w.write('DESWIK_ID;ES;EF;TP\n')
            counter_es_false = 0
            for node in topo_sort:
                for pred in G.predecessors(node):
                    if G[pred][node]['ptype'] == "FinishStart" and (
                            G.nodes[pred]['ef'] + G[pred][node]['dur'] > G.nodes[node]['es']):
                        G.nodes[node]['es'] = G.nodes[pred]['ef'] + \
                            G[pred][node]['dur']
                    elif G[pred][node]['ptype'] == "StartStart" and (
                            G.nodes[pred]['es'] + G[pred][node]['dur'] > G.nodes[node]['es']):
                        G.node[node]['es'] = G.nodes[pred]['es'] + \
                            G[pred][node]['dur']
                    elif G[pred][node]['ptype'] == "PercentOverlap" and (
                            G.nodes[pred]['es'] + G[pred][node]['dur'] > G.nodes[node]['es']):
                        G.nodes[node]['es'] = G.nodes[pred]['es'] + \
                            G[pred][node]['dur']

                    G.nodes[node]['es'] = int(G.nodes[node]['es'])
                    G.nodes[node]['ef'] = int(G.nodes[node]['ef'])
                    G.nodes[node]['ef'] = G.nodes[node]['es'] + \
                        G.nodes[node]['dur']

            # Print Early Starts to file
            for node in topo_sort:
                w.write('{};{};{};{}\n'.format(node,
                                               G.nodes[node]['es'],
                                               G.nodes[node]['ef'],
                                               G.nodes[node]['tp']
                                               ))
            w.close()

        # Precedence Check
        with open(save_path/'PrecedenceCheck.txt', mode='w') as w:
            w.write('PRED;PRED EF;PRED DUR;DELAY;NODE;NODE ES;\n')
            for node in topo_sort:
                for pred in G.predecessors(node):
                    w.write('{};{};{};{};{};{}\n'.format(pred,
                                                         G.nodes[pred]['ef'],
                                                         G.nodes[pred]['dur'],
                                                         G[pred][node]['dur'],
                                                         node,
                                                         G.nodes[node]['es']))
            w.close()

    else:
        print('DiGraph G is cyclic. Terminating.')

    # In[17]:

    # Total number of time periods = scheduling horizon (in days) * shifts (per day) * quarters (per shift)
    time_p_c = schedule_horizon * shift * quarter
    print("Scheduling for %d time periods" % (time_p_c))

    # In[18]:

    # Delete activities with early start beyond scheduling horizon
    remove_nodes = list()
    removed_num = 0

    for node in tuple(G.nodes()):
        if G.nodes[node]['es'] >= time_p_c:
            remove_nodes.append(node)
            removed_num += 1
            # print(node, G.nodes[node]['es'], G.nodes[node]['tp'], G.nodes[node]['start'])
            if node in base_nodes:
                base_nodes.remove(node)
    G.remove_nodes_from(remove_nodes)
    if removed_num > 0:
        print('Warning! Removed activities outside scheduling period (early starts): %d ' % removed_num)
    else:
        print('All ingested activities within scheduling period.')

    # In[19]:

    set_muck = ("C_LH")
    set_ore_dev = ("G_GB", "G_CX", "G_AC", "E_EV", "E_EC")
    set_ore = ("G_GB", "G_CX", "G_AC", "E_EV", "E_EC", 'C_LH')
    set_dev = ("G_GB", "G_CX", "G_GT", "G_AC", "R_RP", "R_EC", "E_EB", "E_EC",
               "E_ESE", "E_EE", "E_EV", "V_CHVT", "V_CHEMUG", "V_CHVTUG", "P_PS")

    # In[20]:

    # Level,Loader,HaulTruck,ChargeTruck,Jumbo,Raise,DD,Prod Drill,Service
    eq_heat = {'loader': 235.06375,
               'truck': 514.2474,
               'charger': 87.961,
               'jumbo': 57.5,
               'raise': 112.73,
               'DD': 90,
               'prod_drill': 90,
               'service': 213.424}

    set_jclt = ["G_GB", "G_CX", "G_GT", "G_AC", "R_RP",
                "R_EC", "E_EB", "E_EC", "E_ESE", "E_EE", "E_EV"]
    set_raise = ["V_CHVT", "V_CHEMUG", "V_CHVTUG"]
    # set_DD = ['DDH', 'OCD', 'CRG']
    set_pd = ['C_LH']
    set_stp = ['C_LH']
    # set_srvc = ['SRC']
    set_rf = ['C_LH']
    # set_na = ['NA', 'VS', 'ASB', 'RBH', 'VL_S', 'VL_E']

    for node in G.nodes():
        n = G.nodes[node]
        n['heat'] = 0.0

        if n['act_type'] in set_jclt:
            if n['dur'] != 0:
                n['heat'] = ((eq_heat['jumbo']*0.2 + eq_heat['charger']*0.15 +
                             eq_heat['loader']*0.35 + eq_heat['truck']*0.3) * n['durh'] / 24)/n['dur']
            else:
                n['heat'] = 0.0
            continue
        elif n['act_type'] in set_raise:
            if n['dur'] != 0:
                n['heat'] = ((eq_heat['raise']*1.0) * n['durh'] / 24)/n['dur']
            else:
                n['heat'] = 0.0
            continue
        elif n['act_type'] in set_pd and n['sot_type'] == 'PROD_DRILL':
            if n['dur'] != 0:
                n['heat'] = ((eq_heat['prod_drill']*0.8 +
                             eq_heat['charger']*0.2) * n['durh'] / 24)/n['dur']
            else:
                n['heat'] = 0.0
            continue
        elif n['act_type'] in set_stp and n['sot_type'] == 'LH_STOPE':
            if n['dur'] != 0:
                n['heat'] = ((eq_heat['loader']*0.7 +
                             eq_heat['truck']*0.3) * n['durh'] / 24)/n['dur']
            else:
                n['heat'] = 0.0
            continue
        elif n['act_type'] in set_rf and n['sot_type'] == 'BACKFILL':
            if n['dur'] != 0:
                n['heat'] = ((eq_heat['truck']*1.0) * n['durh'] / 24)/n['dur']
            else:
                n['heat'] = 0.0
            continue
        else:
            n['heat'] = 0.0
            continue

    print("It's done!")

    # In[21]:

    # Write the Mapping file for DESWIK ID and OMP ID
    with open(save_path/'MNmapp.txt', mode='w') as w:
        w.write('%BZ_ID;DUR;DESWIK_ID;LEVEL')
        w.write('\n')
        count_bz = 0
        for node in G.nodes():
            n = G.nodes[node]
            G.nodes[node]['bz'] = count_bz
            # BZ ID
            w.write('{}'.format(G.nodes[node]['bz']))
            # Duration
            w.write(';{}'.format(n['dur']))
            # Deswik id
            w.write(';{}'.format(node))
            # X, Y, Z Coords
            #w.write(';{};{};{}'.format(round(n['x_cord'],2),round(n['y_cord'], 2),round(n['z_cord'], 2)))
            # Activity Level
            w.write(';{}'.format(n['level']))
            w.write('\n')
            count_bz += 1

    # In[22]:

    # Write the Early starts for each activity
    with open(save_path/'MNes.txt', mode='w') as w:
        w.write('Deswik ID;BZ ID;Early Start\n')
        for node in G:
            w.write('{};{};{}\n'.format(
                index, G.nodes[node]['bz'], G.nodes[node]['es']))

    # In[23]:

    # Write the Blocks File
    with open(save_path/'MNblocks.txt', mode='w') as w:
        w.write('%BZ_ID OBJECTIVE DURATION ORE_TONS PROD_DRILL_UNITS ACCESS_CX_UNITS BACKFILL_UNITS EXTRACTION_UNITS BACKFILL_TONNES ELEC_VENT_STATION_UNITS DRIFT_DEV_UNITS RAMP_DEV_UNITS DEVELOPMENT_METERS\n')

        # Define activity sets

        #set_pri_dev = ("G_GB","G_CX","G_GT","G_AC")
        # set_dab = ('DPT', 'FWD', 'RVA', 'SP.', 'FVA', 'SMP', 'SLT', 'ACC', 'DEC', 'ESC', 'EXP', 'LNK', 'MAG', 'OD.', 'OPA',
        #           'PS.', 'STA', 'SUB', 'TC.', 'TLB', 'WSH', 'GC.')
        # Service Hole, Return Air Raise, Fresh Air Raise, Charge, Slot Raise, Not Applicable, Asbuilt, Escape Raise, Rehab
        # Vent Level Start Milestone, Vent Level End Milestone
        #set_na = ('SRC', 'RAR', 'FAR', 'CRG', 'SR.', 'NA', 'ASB', 'ER.', 'RHB', 'V_LS', 'V_LE')

        # types = ['DD IND', 'Cc_DD OC', 'Cc_Dev Ore', 'Cc_Dev Waste', 'Cc_Drop Raise', 'Cc_ITH', 'Cc_Muck', 'Cc_Prod Drill',
        #          'Cc_Service', 'Cc_Slot', 'Raise', 'Sc_Dev Waste', '-']
        #set_lateral_advance = ("G_GB","G_CX","G_GT","G_AC","R_RP","E_EB","E_EC", "E_EE", "E_EV")
        #set_vertical_advance = ("V_CHV", "V_RBV 3.1m", "V_CHM")
        #set_backfill    = ("BACKFILL")
        #set_extraction  = ("LH_STOPE", "DES_LAT_MINERAL")

        count_predecessors = 0
        count_others = 0
        for node in G.nodes():
            if len(list(G.predecessors(node))) != 0:
                count_predecessors += len(list((G.predecessors(node))))
            n = G.nodes[node]
        # if n['scheduled'] == 1:
            # Write the Deswik ID and objective function value
            w.write('{}'.format(n['bz']))
            w.write(' {}'.format(round(n['obj'], 2)))

            # Write the Duration
            if n['dur'] == 0:
                w.write(' {}'.format(0))
            else:
                w.write(' {}'.format(n['dur']))

    ##################################CONSTRAINTS########################################
                # 1 ORE TONS col 0
            if n['act_type'] in set_ore:
                if not (n['dur'] == 0):
                    w.write(' {}'.format(round(n['lvl_ore_t']/n['dur'], 1)))
                else:
                    w.write(' {}'.format(round(n['lvl_ore_t'], 1)))
            else:
                w.write(' 0.0')

                #  TOTAL TKM HAULAGE
                # if not (n['dur'] == 0):
                #    w.write(' {}'.format(round(n['tkm']/n['dur'], 0)))
                # else:
                #    w.write(' {}'.format(round(n['tkm'], 0)))
                # MUCKING UNITS
                # if n['type'] == 'Cc_Muck':
                #     w.write(' 1')
                # else:
                #     w.write(' 0')
                # PRODUCTION DRILL METERS
    #         if n['act_type'] == 'OPS' or n['act_type'] == 'PDR':
    #             if not (n['dur'] == 0):
    #                 w.write(' {} 1'.format(round(n['pdrill']/n['dur'], 1)))
    #             else:
    #                 w.write(' {} 1'.format(round(n['pdrill'], 1)))
    #         else:
    #             w.write(' 0.0 0')
            # 2 PRODUCTION DRILL UNITS col 1
            if n['resources'] == 'MN_Perforaccion':
                w.write(' 1')
            else:
                w.write(' 0')

            # 3 ACCESS, CROSSCUT, TRANSPORT DRIFT UNITS col 2
            if n['resources'] == 'MN_Avance':
                w.write(' 1')
            else:
                w.write(' 0')

            # 4 BACKFILL UNITS col 3
            if n['resources'] == 'MN_Backfill':
                w.write(' 1')
            else:
                w.write(' 0')

            # 5 EXTRACTION UNITS col 4
            if n['resources'] == 'MN_Extraccion':
                w.write(' 1')
            else:
                w.write(' 0')

            # 6 DEVELOPMENT METERS col 5
            '''
            if n['act_type'] in set_dev:
                if not (n['dur'] == 0):
                    w.write(' {}'.format(round(n['lvl_dev']/n['dur'], 2)))
                else:
                    w.write(' {}'.format(round(n['lvl_dev'], 2)))
            else:
                w.write(' 0.0')
            '''
            # 7 LEVEL BACKFILL TONNES col 6
            if n['act_type'] == 'C_LH':
                if not (n['dur'] == 0):
                    w.write(' {}'.format(round(n['lvl_backfill']/n['dur'], 2)))
                else:
                    w.write(' {}'.format(round(n['lvl_backfill'], 2)))
            else:
                w.write(' 0.0')

            # 8 PRIMARY DEV col 7
            '''
            if n['act_type'] in set_dev:
                if not (n['dur'] == 0):
                    w.write(' {}'.format(round(n['prim_dev']/n['dur'], 2)))
                else:
                    w.write(' {}'.format(round(n['prim_dev'], 2)))
            else:
                w.write(' 0.0')
            '''
            # 9 WASTE DEV col 8
            '''
            if n['act_type'] in set_dev:
                if not (n['dur'] == 0):
                    w.write(' {}'.format(round(n['waste_dev']/n['dur'], 2)))
                else:
                    w.write(' {}'.format(round(n['waste_dev'], 2)))
            else:
                w.write(' 0.0')
            '''
            # 10 RAMP DEV col 9
            '''
            if n['act_type'] in set_dev:
                if not (n['dur'] == 0):
                    w.write(' {}'.format(round(n['ramp_dev']/n['dur'], 2)))
                else:
                    w.write(' {}'.format(round(n['ramp_dev'], 2)))
            else:
                w.write(' 0.0')
            '''

            # 11 ELECTRICAL STATION, PUMPING STATION, VENTILATION STATION, LOADING STATION UNITS COL 10
            if n['resources'] == 'MN_FRENTE MULTIPLE':
                w.write(' 1')
            else:
                w.write(' 0')

            # 12 DRIFT DEVELOPMENT UNITS COL 11
            if n['resources'] == 'MN_GGb':
                w.write(' 1')
            else:
                w.write(' 0')

            # 13 RAMP DEV'T UNITS COL 12
            if n['resources'] == 'MN_Rmp':
                w.write(' 1')
            else:
                w.write(' 0')

            #  DEVELOPMENT METERS COL
            if n['act_type'] in set_dev:
                if not (n['dur'] == 0):
                    w.write(' {}'.format(round(n['lvl_dev']/n['dur'], 2)))
                else:
                    w.write(' {}'.format(round(n['lvl_dev'], 2)))
            else:
                w.write(' 0.0')

            # Heat Emissions
            if n['level'] == 620:
                w.write(' {}{}'.format(round(n['heat'], 2), ' 0'*14))
            elif n['level'] == 600:
                w.write('{} {}{}'.format(' 0'*1, round(n['heat'], 2), ' 0'*13))
            elif n['level'] == 580:
                w.write('{} {}{}'.format(' 0'*2, round(n['heat'], 2), ' 0'*12))
            elif n['level'] == 550:
                w.write('{} {}{}'.format(' 0'*3, round(n['heat'], 2), ' 0'*11))
            elif n['level'] == 525:
                w.write('{} {}{}'.format(' 0'*4, round(n['heat'], 2), ' 0'*10))
            elif n['level'] == 500:
                w.write('{} {}{}'.format(' 0'*5, round(n['heat'], 2), ' 0'*9))
            elif n['level'] == 475:
                w.write('{} {}{}'.format(' 0'*6, round(n['heat'], 2), ' 0'*8))
            elif n['level'] == 450:
                w.write('{} {}{}'.format(' 0'*7, round(n['heat'], 2), ' 0'*7))
            elif n['level'] == 425:
                w.write('{} {}{}'.format(' 0'*8, round(n['heat'], 2), ' 0'*6))
            elif n['level'] == 400:
                w.write('{} {}{}'.format(' 0'*9, round(n['heat'], 2), ' 0'*5))
            elif n['level'] == 375:
                w.write('{} {}{}'.format(' 0'*10, round(n['heat'], 2), ' 0'*4))
            elif n['level'] == 350:
                w.write('{} {}{}'.format(' 0'*11, round(n['heat'], 2), ' 0'*3))
            elif n['level'] == 325:
                w.write('{} {}{}'.format(' 0'*12, round(n['heat'], 2), ' 0'*2))
            elif n['level'] == 300:
                w.write('{} {}{}'.format(' 0'*13, round(n['heat'], 2), ' 0'*1))
            elif n['level'] == 275:
                w.write(' {} {}'.format('0 '*14, round(n['heat'], 2)))

            # Ventilation Domains
            if n['level'] == 620:
                w.write(' {} {}'.format(1, '0 '*14))
            elif n['level'] == 600:
                w.write('{} {} {}'.format(' 0'*1, 1, '0 '*13))
            elif n['level'] == 580:
                w.write('{} {} {}'.format(' 0'*2, 1, '0 '*12))
            elif n['level'] == 550:
                w.write('{} {} {}'.format(' 0'*3, 1, '0 '*11))
            elif n['level'] == 525:
                w.write('{} {} {}'.format(' 0'*4, 1, '0 '*10))
            elif n['level'] == 500:
                w.write('{} {} {}'.format(' 0'*5, 1, '0 '*9))
            elif n['level'] == 475:
                w.write('{} {} {}'.format(' 0'*6, 1, '0 '*8))
            elif n['level'] == 450:
                w.write('{} {} {}'.format(' 0'*7, 1, '0 '*7))
            elif n['level'] == 425:
                w.write('{} {} {}'.format(' 0'*8, 1, '0 '*6))
            elif n['level'] == 400:
                w.write('{} {} {}'.format(' 0'*9, 1, '0 '*5))
            elif n['level'] == 375:
                w.write('{} {} {}'.format(' 0'*10, 1, '0 '*4))
            elif n['level'] == 350:
                w.write('{} {} {}'.format(' 0'*11, 1, '0 '*3))
            elif n['level'] == 325:
                w.write('{} {} {}'.format(' 0'*12, 1, '0 '*2))
            elif n['level'] == 300:
                w.write('{} {} {}'.format(' 0'*13, 1, '0 '*1))
            elif n['level'] == 275:
                w.write(' {} {}'.format('0 '*14, 1, 2))

            w.write('\n')

            #  DPM Emissions
            #w.write(' {}'.format(round(n['dpm'],2)))
        w.write('\n')
    print("Non-scheduled activities: " + str(count_others))

    # In[24]:

    # Write the Predecessor File (.prec or .fpp)
    with open(save_path/'MNfpp.txt', mode='w') as w:
        for node in G:
            w.write('{} {}'.format(
                G.nodes[node]['bz'], len(list(G.predecessors(node)))))
            for pred in G.predecessors(node):
                w.write(' {}'.format(G.nodes[pred]['bz']))
            w.write('\n')

    # In[25]:

    # Write the Delay File
    with open(save_path/'MNdelay.txt', mode='w') as w:
        for node in G:
            w.write('{} {}'.format(
                G.nodes[node]['bz'], len(list(G.predecessors(node)))))
            for pred in G.predecessors(node):
                w.write(' {}'.format(
                    math.ceil(G.nodes[pred]['dur'] + G[pred][node]['dur'])))
            w.write('\n')

    # In[26]:

    # Write the Problem File
    with open(save_path/'MNprob.txt', mode='w') as w:
        w.write('NDESTINATIONS: 1\n')
        w.write('NPERIODS: 1824\n')
        w.write('DISCOUNT_RATE: 0.0002\n')
        w.write('DURATION: 2\n')
        w.write('OBJECTIVE: 0 1\n')

        w.write('NCONSTRAINTS: 40\n')
        w.write('CONSTRAINT: 0 3 P * L 1200\n')
        w.write('CONSTRAINT: 1 4 P * L 2\n')
        w.write('CONSTRAINT: 2 5 P * L 2\n')
        w.write('CONSTRAINT: 3 6 P * L 2\n')
        w.write('CONSTRAINT: 4 7 P * L 2\n')
        w.write('CONSTRAINT: 5 8 P * L 900\n')
        w.write('CONSTRAINT: 6 9 P * L 1\n')
        w.write('CONSTRAINT: 7 10 P * L 1\n')
        w.write('CONSTRAINT: 8 11 P * L 1\n')
        w.write('CONSTRAINT: 9 12 P * L{} 1\n'.format(' 100'*1095))

        w.write('MTCONSTRAINT: 3 * L 1 2\n')

        # 620, 600, 580, 550, 525
        w.write('CONSTRAINT: 10 13 P * L 322\n')
        w.write('CONSTRAINT: 11 14 P * L 310\n')
        w.write('CONSTRAINT: 12 15 P * L 299\n')
        w.write('CONSTRAINT: 13 16 P * L 279\n')
        w.write('CONSTRAINT: 14 17 P * L 263\n')

        # 500, 475, 450, 425, 400
        w.write('CONSTRAINT: 15 18 P * L 246\n')
        w.write('CONSTRAINT: 16 19 P * L 230\n')
        w.write('CONSTRAINT: 17 20 P * L 213\n')
        w.write('CONSTRAINT: 18 21 P * L 195\n')
        w.write('CONSTRAINT: 19 22 P * L 178\n')

        # 375, 350, 325, 300, 275
        w.write('CONSTRAINT: 20 23 P * L 159\n')
        w.write('CONSTRAINT: 21 24 P * L 141\n')
        w.write('CONSTRAINT: 22 25 P * L 122\n')
        w.write('CONSTRAINT: 23 26 P * L 103\n')
        w.write('CONSTRAINT: 24 27 P * L 83\n')

        w.write('CONSTRAINT: 25 28 P * L 100\n')
        w.write('CONSTRAINT: 26 29 P * L 100\n')
        w.write('CONSTRAINT: 27 30 P * L 100\n')
        w.write('CONSTRAINT: 28 31 P * L 100\n')
        w.write('CONSTRAINT: 29 32 P * L 100\n')

        w.write('CONSTRAINT: 30 33 P * L 100\n')
        w.write('CONSTRAINT: 31 34 P * L 100\n')
        w.write('CONSTRAINT: 32 35 P * L 100\n')
        w.write('CONSTRAINT: 33 36 P * L 100\n')
        w.write('CONSTRAINT: 34 37 P * L 100\n')

        w.write('CONSTRAINT: 35 38 P * L 100\n')
        w.write('CONSTRAINT: 36 39 P * L 100\n')
        w.write('CONSTRAINT: 37 40 P * L 100\n')
        w.write('CONSTRAINT: 38 41 P * L 100\n')
        w.write('CONSTRAINT: 39 42 P * L 100\n')


if __name__ == "__main__":
    app.run(debug=True)
