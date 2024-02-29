import pm4py
import numpy as np
from pm4py.objects.log.obj import Event
from pm4py.objects.petri_net.obj import PetriNet

def addStartEndEvents(log):

    for trace in log:

        e_start = Event({'concept:name': 'START', 'org:resource': 'START', 'lifecycle:transition': 'complete', 'time:timestamp': trace[0]['time:timestamp']})
        e_end = Event({'concept:name': 'END', 'org:resource': 'END', 'lifecycle:transition': 'complete', 'time:timestamp': trace[-1]['time:timestamp']})

        trace.insert(0, e_start)
        trace.append(e_end)

    return log


def addStartEndTransitions(net, initial_marking, final_marking):
        
    N = len(net.transitions) + len(net.places) + 1
    t_start = PetriNet.Transition(name = 'n'+str(N), label='START')
    net.transitions.add(t_start)

    N = len(net.transitions) + len(net.places) + 1
    t_end = PetriNet.Transition(name = 'n'+str(N), label='END')
    net.transitions.add(t_end)

    N = len(net.transitions) + len(net.places) + 1
    new_place_start = PetriNet.Place(name = 'n'+str(N))
    net.places.add(new_place_start)

    N = len(net.transitions) + len(net.places) + 1
    new_place_end = PetriNet.Place(name = 'n'+str(N))
    net.places.add(new_place_end)

    places_from = list(initial_marking)
    places_to = list(final_marking)

    t_after_start = []
    for p in places_from:
        for arc in p.out_arcs:
            t_after_start.append(arc.target)
            net.arcs.remove(arc)
        p.out_arcs.clear()

    t_before_end = []
    for p in places_to:
        for arc in p.in_arcs:
            t_before_end.append(arc.source)
            net.arcs.remove(arc)
        p.in_arcs.clear()

    new_arcs = []

    for p in places_from:
        new_arcs.append(PetriNet.Arc(p, t_start))
        
    for p in places_to:
        new_arcs.append(PetriNet.Arc(t_end, p))

    new_arcs.append(PetriNet.Arc(t_start, new_place_start))
    new_arcs.append(PetriNet.Arc(new_place_end, t_end))

    for t in t_after_start:
        new_arcs.append(PetriNet.Arc(new_place_start, t))

    for t in t_before_end:
        new_arcs.append(PetriNet.Arc(t, new_place_end))

    for arc in new_arcs:
        net.arcs.add(arc)

    return net, initial_marking, final_marking
        


def splitRealLog(real, split_size = (0.6, 0.2, 0.2), split_temporal = True, save_to = ''):
    df_real = pm4py.convert_to_dataframe(real)

    cases = df_real['case:concept:name'].unique()
    np.random.seed(72)
    if not split_temporal:
        np.random.shuffle(cases)
    new_case_names = dict(zip(cases,range(len(cases))))
    df_real['CaseN'] = df_real['case:concept:name'].apply(lambda x: new_case_names[x])

    size_train = split_size[0]
    size_val = split_size[1]
    size_test = split_size[2]

    n_train = int(size_train * len(cases))
    n_val = int(size_val * len(cases))
    n_test = int(size_test * len(cases))

    df_train = df_real[df_real['CaseN'] < n_train]
    df_val = df_real[(df_real['CaseN'] >= n_train) & (df_real['CaseN'] < n_train + n_val)]
    df_test = df_real[df_real['CaseN'] >= n_train + n_val]

    del df_train['CaseN']
    del df_val['CaseN']
    del df_test['CaseN']

    real_train = pm4py.convert_to_event_log(df_train) 
    real_val = pm4py.convert_to_event_log(df_val) 
    real_test = pm4py.convert_to_event_log(df_test)

    if save_to:
        pm4py.write_xes(real_train, save_to + '/logTrain.xes')
        pm4py.write_xes(real_val, save_to + '/logVal.xes')
        pm4py.write_xes(real_test, save_to + '/logTest.xes')
    
    return real_train, real_val, real_test