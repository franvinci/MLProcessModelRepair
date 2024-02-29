from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.petri_net.utils import reachability_graph
import random
random.seed(72)


def search_transition(net, label):
    for t in list(net.transitions):
        if t.label == label:
            return t


def ta_reach_tb(ts, t_a, t_b, trans_done = []):
    trans_done.append(t_a)
    t_a_reachgraph = [T for T in list(ts.transitions) if T.name == str(t_a)]
    t_b_reachgraph = [T for T in list(ts.transitions) if T.name == str(t_b)]

    reach_nodes_a = [t.to_state for t in t_a_reachgraph]
    next_t = [x for n in reach_nodes_a for x in list(n.outgoing)]
    for t in t_b_reachgraph:
        if t in next_t:
            return True
    
    next_invt = [t for t in next_t if len(t.name.split("'")) == 1 and t not in trans_done]
    if len(next_invt) > 0:
        if True in [ta_reach_tb(ts, invt, t_b, trans_done) for invt in next_invt]:
            return True
    
    return False


def add_skip(net, initial_marking, a, b):
    
    t_a = search_transition(net, a)
    t_b = search_transition(net, b)
    
    ts = reachability_graph.construct_reachability_graph(net, initial_marking)

    if ta_reach_tb(ts, t_a, t_b, []):
        return net, False

    t_a_reachgraph = [T for T in list(ts.transitions) if T.name == str(t_a)]
    reach_nodes = [t.to_state for t in t_a_reachgraph]
    list_new_sources = [['n'+x[:-1] for x in r.name.split('n')[1:]]for r in reach_nodes]

    t_b_reachgraph = [T for T in list(ts.transitions) if T.name == str(t_b)]
    reach_nodes = [t.from_state for t in t_b_reachgraph]
    list_new_targets = [['n'+x[:-1] for x in r.name.split('n')[1:]]for r in reach_nodes]

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


def add_skip_b(net, a, b):
    
    t_a = search_transition(net, a)
    t_b = search_transition(net, b)
    
    # check if it exists
    for arc in list(t_a.out_arcs):
        p = arc.target
        for p_arc in list(p.out_arcs):
            if not p_arc.target.label:
                return net, False
                
    # if not
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


def updateModel(net, initial_marking, final_marking, recc, top_n_recc = 0):

    model_updates = [False]

    if top_n_recc > 0:
        recc = dict(list(recc.items())[:top_n_recc])

    for rel in recc.keys():
        print(rel)
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
                net, model_updated = add_skip(net, initial_marking, a, b)
                model_updates.append(model_updated)

        elif recc[rel][1] == 'skip_b':
            if (a not in transition_labels) or (b not in transition_labels):
                continue
            net, model_updated = add_skip_b(net, a, b)
            model_updates.append(model_updated)
    
    model_updated = sum(model_updates) > 0

    return net, model_updated