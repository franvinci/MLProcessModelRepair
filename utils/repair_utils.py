from pm4py.objects.petri_net.obj import PetriNet


def search_transition(net, label):
    for t in list(net.transitions):
        if t.label == label:
            return t


def a_follow_b(t_a, b):
    for s in list(t_a.out_arcs):
        p = s.target
        for arc in list(p.out_arcs):
            if arc.target.label == b:
                return True
            if not arc.target.label:
                if a_follow_b(arc.target, b):
                    return True
    return False


def a_follow_b_in_net(net, a, b):
    t_a = search_transition(net, a)
    return a_follow_b(t_a, b)


def net_start_a(net, initial_marking, a):
    for p in list(initial_marking):
        for arc in list(net.arcs):
            if arc.source == p:
                if arc.target.label == a:
                    return True
                if not arc.target.label:
                    if a_follow_b(arc.target, a):
                        return True                    
    return False


def net_end_b(net, final_marking, b):
    for p in list(final_marking):
        for arc in list(net.arcs):
            if arc.target == p:
                if arc.source.label == b:
                    return True
                if not arc.source.label:
                    t_b = search_transition(net, b)
                    if a_follow_b(t_b, arc.target):
                        return True
    return False


def add_start(net, initial_marking, a):

    if net_start_a(net, initial_marking, a):
        return net, False

    t_a = search_transition(net, a)
    N = len(net.transitions) + len(net.places) + 1
    inv_t = PetriNet.Transition(name = 'n'+str(N))
    net.transitions.add(inv_t)
    new_sources = list(initial_marking)
    new_target = [arc.source for arc in list(t_a.in_arcs)]
    new_arcs = []

    for s in new_sources:
        new_arcs.append(PetriNet.Arc(s, inv_t))

    for t in new_target:
        new_arcs.append(PetriNet.Arc(inv_t, t))

    for a in new_arcs:
        net.arcs.add(a)
    
    return net, True


def add_end(net, final_marking, b):

    if net_end_b(net, final_marking, b):
        return net, False
    
    t_b = search_transition(net, b)
    N = len(net.transitions) + len(net.places) + 1
    inv_t = PetriNet.Transition(name = 'n'+str(N))
    net.transitions.add(inv_t)
    new_sources = [arc.target for arc in list(t_b.out_arcs)]
    new_target = list(final_marking)
    new_arcs = []

    for s in new_sources:
        new_arcs.append(PetriNet.Arc(s, inv_t))

    for t in new_target:
        new_arcs.append(PetriNet.Arc(inv_t, t))

    for a in new_arcs:
        net.arcs.add(a)
    
    return net, True


def add_skip(net, initial_marking, final_marking, a, b):

    if a == '<start>':
        return add_start(net, initial_marking, b)
    
    if b == '<end>':
        return add_end(net, final_marking, a)

    if a_follow_b_in_net(net, a, b):
        return net, False
    
    t_a = search_transition(net, a)
    t_b = search_transition(net, b)
    N = len(net.transitions) + len(net.places) + 1
    inv_t = PetriNet.Transition(name = 'n'+str(N))
    net.transitions.add(inv_t)
    new_sources = [arc.target for arc in list(t_a.out_arcs)]
    new_target = [arc.source for arc in list(t_b.in_arcs)]
    new_arcs = []

    for s in new_sources:
        new_arcs.append(PetriNet.Arc(s, inv_t))

    for t in new_target:
        new_arcs.append(PetriNet.Arc(inv_t, t))

    for a in new_arcs:
        net.arcs.add(a)
    
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
    new_arcs = []

    for s in new_sources:
        new_arcs.append(PetriNet.Arc(s, inv_t))

    for t in new_target:
        new_arcs.append(PetriNet.Arc(inv_t, t))

    for a in new_arcs:
        net.arcs.add(a)
    
    return net, True


def updateModel(net, initial_marking, final_marking, recc, top_n_recc = 0):

    model_updates = [False]

    if top_n_recc > 0:
        recc = dict(list(recc.items())[:top_n_recc])

    for rel in recc.keys():

        a = rel.split(' -> ')[0]
        b = rel.split(' -> ')[1]

        if recc[rel][1] == 'skip':
            net, model_updated = add_skip(net, initial_marking, final_marking, a, b)
            model_updates.append(model_updated)

        elif recc[rel][1] == 'skip_b':
            if a == '<start>' or b == '<end>':
                continue
            net, model_updated = add_skip_b(net, a, b)
            model_updates.append(model_updated)
    
    model_updated = sum(model_updates) > 0

    return net, model_updated