from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.petri_net.utils import reachability_graph
from pm4py.objects.log.obj import EventLog
from collections import deque, Counter
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
import random
random.seed(72)


def search_transition(net, label):
    for t in list(net.transitions):
        if t.label == label:
            return t


def has_path_between_states(states1, states2):
    visited = set()

    def dfs(node):
        if node in states2:
            return True

        visited.add(node)

        next_nodes = [t.to_state for t in list(node.outgoing) if str(t)[-5:-1] == 'None']
        for next_node in next_nodes:
            if next_node not in visited:
                if dfs(next_node):
                    return True
        return False

    for start_node in states1:
        if start_node not in visited:
            if dfs(start_node):
                return True
            
    return False 


def ta_reach_tb(ts, t_a, t_b):

    t_a_reachgraph = [T for T in list(ts.transitions) if T.name == str(t_a)]
    t_b_reachgraph = [T for T in list(ts.transitions) if T.name == str(t_b)]
    states_after_a = [t.to_state for t in t_a_reachgraph]
    states_before_b = [t.from_state for t in t_b_reachgraph]
    
    return has_path_between_states(states_after_a, states_before_b)


def ta_skip_tb(ts, t_a, t_b):
    
    t_a_reachgraph = [T for T in list(ts.transitions) if T.name == str(t_a)]
    states_after_a = [t.to_state for t in t_a_reachgraph]
    t_b_reachgraph = [T for T in list(ts.transitions) if T.name == str(t_b)]
    for s in states_after_a:
        s_outgoing = list(s.outgoing)
        for t in s_outgoing:
            if t not in t_b_reachgraph:
                return True
            
    return False  


def shortest_path(start_node, end_node):
    visited = set()
    queue = deque([(start_node, [])])

    while queue:
        current_node, path = queue.popleft()

        if current_node == end_node:
            return path + [current_node]

        if current_node in visited:
            continue

        visited.add(current_node)

        current_node_outgoing = current_node.outgoing
        for arc in current_node_outgoing:
            next_node = arc.to_state
            queue.append((next_node, path + [current_node]))

    return None

def find_shortest_path(nodes_set1, nodes_set2):
    shortest_distance = float('inf')
    shortest_path_nodes = None

    for node1 in nodes_set1:
        for node2 in nodes_set2:
            path = shortest_path(node1, node2)

            if path and len(path) < shortest_distance:
                shortest_distance = len(path)
                shortest_path_nodes = (node1, node2)

    return shortest_path_nodes


def filter_event_log_activities(log, l):
    
    traces_l = []

    for trace in log:
        for event in trace:
            if event['concept:name'] in l:
                traces_l.append(trace)
    
    return traces_l

def return_most_common_places_alignments_fp(log, net, im, fm, ts, a, b):    

    traces_l = filter_event_log_activities(log, [a, b])

    try:
        alignments_ = alignments.apply_log(EventLog(traces_l), net, im, fm, parameters={"ret_tuple_as_trans_desc": True})
    except:
        alignments_ = []
        for i in range(len(traces_l)):
            alignments_.append(alignments.apply_trace(traces_l[i], net, im, fm, parameters={"ret_tuple_as_trans_desc": True}))

    aligned_traces = [[y[0] for y in x['alignment'] if y[0][1]!='>>'] for x in alignments_]


    places_after = []
    places_before = []

    for aligned_trace in aligned_traces:
        
        align_transitions_ts = []
        for al_event in aligned_trace:
            for T in list(ts.transitions):
                if al_event[1] == str(T).split(',')[0][1:]:
                    align_transitions_ts.append(T)
                    break
        
        align_nodes_ts_to = []
        for al_t in align_transitions_ts:
            align_nodes_ts_to.append(al_t.to_state)

        align_nodes_ts_from = []
        for al_t in align_transitions_ts:
            align_nodes_ts_from.append(al_t.from_state)

        for i in range(len(align_transitions_ts)):
            if align_transitions_ts[i].name.split(", ")[1][1:-2] == a:
                places_after.append(align_nodes_ts_to[i])
            if align_transitions_ts[i].name.split(", ")[1][1:-2] == b:
                places_before.append(align_nodes_ts_from[i])


    count_after = Counter(places_after)
    most_common_after = count_after.most_common(1)[0][0]

    count_before = Counter(places_before)
    most_common_before = count_before.most_common(1)[0][0]

    return most_common_after, most_common_before


def add_skip(net, initial_marking, a, b, method='greedy', log=None, final_marking=None):
    
    t_a = search_transition(net, a)
    t_b = search_transition(net, b)
    
    ts = reachability_graph.construct_reachability_graph(net, initial_marking)

    if ta_reach_tb(ts, t_a, t_b):
        return net, False

    # find transition objects in ts
    s = 'START'
    t_start = search_transition(net, s)
    ts_trans = list(ts.transitions)
    for T_s in ts_trans:
        if T_s.name == str(t_start):
            break
    e = 'END'
    t_end = search_transition(net, e)
    for T_e in ts_trans:
        if T_e.name == str(t_end):
            break

    if method != 'alignments':
        t_a_reachgraph = [T for T in list(ts.transitions) if T.name == str(t_a)]
        reach_nodes_a = [t.to_state for t in t_a_reachgraph]
        t_b_reachgraph = [T for T in list(ts.transitions) if T.name == str(t_b)]
        reach_nodes_b = [t.from_state for t in t_b_reachgraph]

    if method == 'greedy':
        reach_nodes = [find_shortest_path([T_s.from_state], reach_nodes_a)[1]]
        list_new_sources = [['n'+x[:-1] for x in r.name.split('n')[1:]]for r in reach_nodes]
        reach_nodes = [find_shortest_path(reach_nodes_b, [T_e.to_state])[0]]
        list_new_targets = [['n'+x[:-1] for x in r.name.split('n')[1:]]for r in reach_nodes]
    if method == 'complete':
        list_new_sources = [['n'+x[:-1] for x in r.name.split('n')[1:]]for r in reach_nodes_a]
        list_new_targets = [['n'+x[:-1] for x in r.name.split('n')[1:]]for r in reach_nodes_b]
    if method == 'alignments':
        most_common_after, most_common_before = return_most_common_places_alignments_fp(log, net, initial_marking, final_marking, ts, a, b)
        list_new_sources = [['n'+x[:-1] for x in r.name.split('n')[1:]]for r in [most_common_after]]
        list_new_targets = [['n'+x[:-1] for x in r.name.split('n')[1:]]for r in [most_common_before]]

    for str_new_sources in list_new_sources:
        new_sources = [p for p in list(net.places) if p.name in str_new_sources]
        for str_new_target in list_new_targets:
            new_target = [p for p in list(net.places) if p.name in str_new_target]
            N = len(net.transitions) + len(net.places) + 1
            inv_t = PetriNet.Transition(name = 'n'+str(N))
            net.transitions.add(inv_t)

            for s in new_sources:
                arc = PetriNet.Arc(s, inv_t)
                net.arcs.add(arc)
                s.out_arcs.add(arc)
                inv_t.in_arcs.add(arc)

            for t in new_target:
                arc = PetriNet.Arc(inv_t, t)
                net.arcs.add(arc)
                inv_t.out_arcs.add(arc)
                t.in_arcs.add(arc)
    
    return net, True


def add_skip_b(net, initial_marking, a, b):
    
    t_a = search_transition(net, a)
    t_b = search_transition(net, b)

    ts = reachability_graph.construct_reachability_graph(net, initial_marking)
    
    if ta_skip_tb(ts, t_a, t_b):
        return net, False
                
    N = len(net.transitions) + len(net.places) + 1
    inv_t = PetriNet.Transition(name = 'n'+str(N))
    net.transitions.add(inv_t)
    new_sources = [arc.target for arc in list(t_a.out_arcs)]
    new_target = [arc.target for arc in list(t_b.out_arcs)]

    for s in new_sources:
        arc = PetriNet.Arc(s, inv_t)
        net.arcs.add(arc)
        s.out_arcs.add(arc)
        inv_t.in_arcs.add(arc)

    for t in new_target:
        arc = PetriNet.Arc(inv_t, t)
        net.arcs.add(arc)
        inv_t.out_arcs.add(arc)
        t.in_arcs.add(arc)
    
    return net, True


def add_new_transition_a(net, a, b):

    t_b = search_transition(net, b)

    places_from = [arc.source for arc in list(t_b.in_arcs)]

    for arc in t_b.in_arcs:
        net.arcs.remove(arc)

    t_b.in_arcs.clear()

    N = len(net.transitions) + len(net.places) + 1
    t_a = PetriNet.Transition(name = 'n'+str(N), label=a)
    net.transitions.add(t_a)

    N = len(net.transitions) + len(net.places) + 1
    inv_t = PetriNet.Transition(name = 'n'+str(N))
    net.transitions.add(inv_t)

    N = len(net.transitions) + len(net.places) + 1
    new_place = PetriNet.Place(name = 'n'+str(N))
    net.places.add(new_place)

    arc = PetriNet.Arc(new_place, t_b)
    net.arcs.add(arc)
    new_place.out_arcs.add(arc)
    t_b.in_arcs.add(arc)

    arc = PetriNet.Arc(inv_t, new_place)
    net.arcs.add(arc)
    inv_t.out_arcs.add(arc)
    new_place.in_arcs.add(arc)

    arc = PetriNet.Arc(t_a, new_place)
    net.arcs.add(arc)
    t_a.out_arcs.add(arc)
    new_place.in_arcs.add(arc)

    for s in places_from:

        arc = PetriNet.Arc(s, t_a)
        net.arcs.add(arc)
        s.out_arcs.add(arc)
        t_a.in_arcs.add(arc)

        arc = PetriNet.Arc(s, inv_t)
        net.arcs.add(arc)
        s.out_arcs.add(arc)
        inv_t.in_arcs.add(arc)

    return net

def add_new_transition_b(net, a, b):

    t_a = search_transition(net, a)

    places_to = [arc.target for arc in list(t_a.out_arcs)]

    for arc in t_a.out_arcs:
        net.arcs.remove(arc)

    t_a.out_arcs.clear()

    N = len(net.transitions) + len(net.places) + 1
    t_b = PetriNet.Transition(name = 'n'+str(N), label=b)
    net.transitions.add(t_b)

    N = len(net.transitions) + len(net.places) + 1
    inv_t = PetriNet.Transition(name = 'n'+str(N))
    net.transitions.add(inv_t)

    N = len(net.transitions) + len(net.places) + 1
    new_place = PetriNet.Place(name = 'n'+str(N))
    net.places.add(new_place)

    arc = PetriNet.Arc(t_a, new_place)
    net.arcs.add(arc)
    t_a.out_arcs.add(arc)
    new_place.in_arcs.add(arc)

    arc = PetriNet.Arc(new_place, inv_t)
    net.arcs.add(arc)
    new_place.out_arcs.add(arc)
    inv_t.in_arcs.add(arc)

    arc = PetriNet.Arc(new_place, t_b)
    net.arcs.add(arc)
    new_place.out_arcs.add(arc)
    t_b.in_arcs.add(arc)

    for s in places_to:

        arc = PetriNet.Arc(t_b, s)
        net.arcs.add(arc)
        t_b.out_arcs.add(arc)
        s.in_arcs.add(arc)

        arc = PetriNet.Arc(inv_t, s)
        net.arcs.add(arc)
        inv_t.out_arcs.add(arc)
        s.in_arcs.add(arc)

    return net


def updateModel(net, initial_marking, recc, method='greedy', top_n_recc = 0, log=None, final_marking=None):

    model_updates = [False]

    if top_n_recc > 0:
        recc = dict(list(recc.items())[:top_n_recc])

    for rel in recc.keys():
        transition_labels = [x.label for x in list(net.transitions)]

        a = rel.split(' -> ')[0]
        b = rel.split(' -> ')[1]

        if recc[rel][1] == 'skip':
            if (a not in transition_labels) and (b not in transition_labels):
                continue
            if a not in transition_labels:
                net = add_new_transition_a(net, a, b)
                model_updates.append(True)
            if b not in transition_labels:
                net = add_new_transition_b(net, a, b)
                model_updates.append(True)
            if (a in transition_labels) and (b in transition_labels):
                net, model_updated = add_skip(net, initial_marking, a, b, method, log, final_marking)
                model_updates.append(model_updated)

        elif recc[rel][1] == 'skip_b':
            if (a not in transition_labels) or (b not in transition_labels):
                continue
            net, model_updated = add_skip_b(net, initial_marking, a, b)
            model_updates.append(model_updated)
    
    model_updated = sum(model_updates) > 0

    return net, model_updated